"""Layout-aware manuscript extraction: turn a PDF's structured blocks into clean per-page
body segments, so citation claims are free of headings, headers/footers, captions and
column-bleed. Pure functions over TextBlock (unit-testable); only extract_blocks touches fitz."""
import re
from collections import Counter
from dataclasses import dataclass

from probatio.manuscript import _REFS_HEADING, strip_line_numbers


@dataclass(frozen=True)
class TextBlock:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 (page pixels)
    size: float                               # dominant span font size
    bold: bool                                # dominant span weight
    page: int
    page_height: float


def estimate_body_size(blocks: list[TextBlock]) -> float:
    """Length-weighted modal font size = 'normal' body text (headings are few + short)."""
    weighted: Counter[float] = Counter()
    for b in blocks:
        t = b.text.strip()
        if t:
            weighted[round(b.size, 1)] += len(t)
    return weighted.most_common(1)[0][0] if weighted else 0.0


# Caption: "Figure|Fig|Table|Tbl|Scheme" + space + digits + period-or-colon (real captions only).
# A sentence like "Table 1 lists …" has no period/colon after the number → stays body.
_CAPTION = re.compile(
    r"^\s*(figure|fig|table|tbl|scheme)\s*\d+\s*[.:]",
    re.I,
)

# Numbered section heading: "2.", "2.6", "2.6.5 Extra Trees", optionally trailing dot
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")

# Math operators that mark a real equation (excludes bare < > which appear in stats/markup)
_MATH_OPS = re.compile(r"[=∑∫√±×÷≈≤≥∝]")


def _norm_repeat(text: str) -> str:
    """Normalise text for repetition matching.

    Only collapse *isolated* all-digit tokens (i.e. whole words that are pure digits)
    to "#", so that standalone page numbers like "3" or "42" are normalised but embedded
    digits in words like "Patient1" or inline numbers inside sentences ("Patient 1 improved")
    are preserved per-token.  This prevents two distinct sentences that differ only in an
    embedded number from being treated as the same repeated header.
    """
    return re.sub(r"(?<!\S)\d+(?!\S)", "#", text.strip().lower())


# Page-number shape: pure Arabic, Roman numerals, or "page N [of M]" (all case-insensitive).
_PAGE_NUMBER = re.compile(
    r"^(?:\d{1,4}|[ivxlcdm]+|(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?)$",
    re.I,
)


def _is_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER.match(text.strip()))


def _centroid_in_margin(bbox: tuple[float, float, float, float], page_height: float) -> bool:
    """Return True iff the block's vertical centroid falls in the outer 8% margin band.

    Guards against page_height <= 0 by returning False (treat as body).
    """
    if page_height <= 0:
        return False
    cy = (bbox[1] + bbox[3]) / 2.0
    return cy < 0.08 * page_height or cy > 0.92 * page_height


def drop_margins(blocks: list[TextBlock]) -> list[TextBlock]:
    """Remove page-number blocks and repeated running headers/footers.

    BIAS TOWARD KEEPING.  A margin-band block is removed only when it is EITHER:
      1. page-number-shaped (pure digits, Roman numerals, or "page N [of M]"); OR
      2. repeated (same normalised text) in the margin band on >=2 distinct pages.

    "In margin" is determined by the block's vertical CENTROID, not its edges, so a
    paragraph whose bottom merely crosses 92 % but whose centre is in the body zone
    is never considered a margin block.
    """
    if not blocks:
        return []

    # Build repetition counts only from margin-band blocks, keyed by normalised text.
    # Track (normalised_text → set of page numbers) so we can test ">=2 distinct pages".
    margin_pages: dict[str, set[int]] = {}
    for b in blocks:
        t = b.text.strip()
        if not t:
            continue
        if _centroid_in_margin(b.bbox, b.page_height):
            key = _norm_repeat(t)
            margin_pages.setdefault(key, set()).add(b.page)

    repeated = {k for k, pages in margin_pages.items() if len(pages) >= 2}

    out: list[TextBlock] = []
    for b in blocks:
        t = b.text.strip()
        if not t:
            out.append(b)
            continue
        if _centroid_in_margin(b.bbox, b.page_height):
            if _is_page_number(t) or _norm_repeat(t) in repeated:
                continue  # drop it
        out.append(b)
    return out


def classify(block: TextBlock, body_size: float) -> str:
    r"""Classify a text block as heading|caption|equation|body.

    BIAS TOWARD BODY: a non-body classification causes the block's text to be dropped
    downstream. Only classify as heading/caption/equation when confident.
    Missing a real heading is harmless; misclassifying claim text is a bug.

    Classification rules:
    - caption: matches "(figure|fig|table|tbl|scheme) <digits> <period|colon>" — the number
        must be immediately followed by '.' or ':' as in real captions. A sentence such as
        "Table 1 lists the primer sequences [5]." stays body.
    - heading: short (<=100 chars), single-line, does NOT end with sentence punctuation
        (.!?), AND has a heading signal:
        * numbered title ^\d+(\.\d+)*\.?\s+[A-Z], OR
        * bold, OR
        * size > body_size * 1.15  (only when body_size > 0 — zero-guard), OR
        * ALL-CAPS title-like: t.upper()==t, contains letters, AND (>=2 words OR
          alphabetic-length >= 6) — so short acronyms like "DNA"/"IV" stay body.
    - equation: short, single-line, contains a real math operator (= ∑ ∫ √ ± × ÷ ≈ ≤ ≥ ∝)
        AND has a low alphabetic ratio (<0.5). Bare citation markers "[12]", page numbers,
        stat markers "p < 0.05", or "***" have no math operator → body.
    - body: everything else.
    """
    t = block.text.strip()
    if not t:
        return "body"

    # A genuine heading/caption/equation NEVER contains an in-text citation marker.
    # Guard here so a short fragment like "5 Gy reduced viability [3]" is never
    # mislabelled as heading (via the numbered pattern) and silently dropped.
    from probatio.manuscript import find_markers  # noqa: PLC0415 — function-level import avoids circular-import risk
    if find_markers(t):
        return "body"

    # --- caption ---
    if _CAPTION.match(t):
        return "caption"

    single_line = "\n" not in t
    short = len(t) <= 100

    if short and single_line:
        # Headings must NOT end with sentence punctuation (.!?)
        ends_like_sentence = t[-1] in ".!?"
        letters = [c for c in t if c.isalpha()]
        alpha_len = len(letters)

        if not ends_like_sentence:
            # numbered section heading
            if _NUMBERED_HEADING.match(t):
                return "heading"
            # bold
            if block.bold:
                return "heading"
            # larger font (zero-guard: only when body_size > 0)
            if body_size > 0 and block.size > body_size * 1.15:
                return "heading"
            # ALL-CAPS title-like: must have letters AND (multi-word OR long enough)
            if letters and t.upper() == t:
                words = t.split()
                if len(words) >= 2 or alpha_len >= 6:
                    return "heading"

        # equation: requires a real math operator AND low alpha ratio
        if _MATH_OPS.search(t):
            if alpha_len / len(t) < 0.5:
                return "equation"

    return "body"


def order_by_column(page_blocks: list[TextBlock]) -> list[TextBlock]:
    """Order blocks on a single page: if an unambiguous 2-column layout is detected,
    return left column top-to-bottom then right column top-to-bottom; otherwise return
    all blocks top-to-bottom (y0, then x0 as tie-breaker).

    Column detection uses a gap/gutter test rather than page midpoint, so that
    indented single-column blocks are never false-split into two columns:

    1. <= 1 block -> return as-is.
    2. Compute each block's center-x.  Sort the centers and find the largest gap
       between consecutive values.  Treat as two columns ONLY when ALL of:
         a. gap > 0.15 * content_width  (content_width = max(x1) - min(x0))
         b. the gap splits blocks into a left group AND a right group each with >= 2 blocks
         c. left group's max x1 <= gutter_x AND right group's min x0 >= gutter_x
            (no horizontal overlap across the gutter; gutter_x = midpoint of the gap)
         d. no block spans the gutter (x0 < gutter_x AND x1 > gutter_x)
    3. If two-column: left (by y0, x0) + right (by y0, x0).
    4. Otherwise (single-column or ambiguous): all blocks sorted by (y0, x0).

    The primary input is single-column submission manuscripts; we must never scramble those.
    Only switch to 2-column when there is an unambiguous vertical gutter.
    """
    if len(page_blocks) <= 1:
        return list(page_blocks)

    def by_y(blocks: list[TextBlock]) -> list[TextBlock]:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    # Compute center-x for each block.
    centers = sorted((b.bbox[0] + b.bbox[2]) / 2.0 for b in page_blocks)

    # Find the largest gap between consecutive center-x values.
    max_gap = 0.0
    gutter_x = 0.0
    for i in range(1, len(centers)):
        gap = centers[i] - centers[i - 1]
        if gap > max_gap:
            max_gap = gap
            gutter_x = (centers[i - 1] + centers[i]) / 2.0

    content_width = max(b.bbox[2] for b in page_blocks) - min(b.bbox[0] for b in page_blocks)

    # Guard against degenerate content (zero-width).
    if content_width <= 0:
        return by_y(page_blocks)

    # Condition (a): gap must be large relative to content width.
    if max_gap <= 0.15 * content_width:
        return by_y(page_blocks)

    # Split blocks at gutter_x.
    left = [b for b in page_blocks if (b.bbox[0] + b.bbox[2]) / 2.0 < gutter_x]
    right = [b for b in page_blocks if (b.bbox[0] + b.bbox[2]) / 2.0 >= gutter_x]

    # Condition (b): each group must have >= 2 blocks.
    if len(left) < 2 or len(right) < 2:
        return by_y(page_blocks)

    # Condition (c): groups must not overlap across the gutter.
    left_max_x1 = max(b.bbox[2] for b in left)
    right_min_x0 = min(b.bbox[0] for b in right)
    epsilon = 1.0  # 1 pt tolerance
    if left_max_x1 > gutter_x + epsilon or right_min_x0 < gutter_x - epsilon:
        return by_y(page_blocks)

    # Condition (d): no block may span the gutter.
    for b in page_blocks:
        if b.bbox[0] < gutter_x and b.bbox[2] > gutter_x:
            return by_y(page_blocks)

    return by_y(left) + by_y(right)


def page_segments(ordered_blocks: list[TextBlock], body_size: float) -> list[str]:
    """Walk already-ordered blocks from a single page; accumulate consecutive body blocks
    (joined by spaces) into a segment; end the current segment at any non-body block
    (heading/caption/equation), whose text is dropped. Returns the list of non-empty
    body segments.

    Heading isolation is only as good as classify() — a heading that classify() leaves as
    body (e.g., same-size, non-bold, non-numbered text) will NOT create a boundary. This
    is the accepted conservative trade-off: we bias toward keeping text.

    CALLER CONTRACT: ordered_blocks must be a SINGLE page's blocks, already column-ordered
    (the orchestrator groups by page before calling). Multi-page input would space-join
    across the page break.
    """
    segments: list[str] = []
    current: list[str] = []
    for b in ordered_blocks:
        if classify(b, body_size) == "body":
            t = b.text.strip()
            if t:
                current.append(t)
        elif current:
            segments.append(" ".join(current))
            current = []
    if current:
        segments.append(" ".join(current))
    return segments


def _parse_block(blk: dict, page: int, page_height: float) -> "TextBlock | None":
    """Parse a single PyMuPDF block dict into a TextBlock, or return None.

    Never raises.  Returns None when:
    - blk is not a dict
    - lines key is absent or null (no text to extract)
    - assembled text is empty or there are no spans
    - bbox is absent, null, or has fewer than 4 elements

    Null-guards per field:
    - lines:         blk.get("lines") or []
    - span flags:    dom.get("flags") or 0
    - span size:     float(dom.get("size") or 0.0)
    - span font:     str(dom.get("font") or "")
    - bbox:          checked for None and len >= 4
    """
    if not isinstance(blk, dict):
        return None
    lines = blk.get("lines") or []
    spans = [s for ln in lines for s in (ln.get("spans") or [])]
    text = strip_line_numbers("\n".join(
        "".join(s.get("text", "") for s in (ln.get("spans") or []))
        for ln in lines
    ).strip())
    if not text or not spans:
        return None
    bbox_raw = blk.get("bbox")
    if bbox_raw is None or len(bbox_raw) < 4:
        return None
    bbox: tuple[float, float, float, float] = tuple(  # type: ignore[assignment]
        float(x) for x in bbox_raw[:4]
    )
    dom = max(spans, key=lambda s: len(s.get("text", "")))
    size = float(dom.get("size") or 0.0)
    bold = bool((dom.get("flags") or 0) & 16) or "bold" in str(dom.get("font") or "").lower()
    return TextBlock(text=text, bbox=bbox, size=size, bold=bold, page=page, page_height=page_height)


def extract_blocks(doc) -> list[TextBlock]:  # type: ignore[no-untyped-def]
    """Extract TextBlock objects from a fitz document — the ONLY fitz-touching function.

    Reads page.get_text("dict") per page.  Per block, delegates to _parse_block which
    is null-safe and never raises; None results (malformed/empty blocks) are skipped.
    One bad block never aborts the whole document.
    """
    import fitz  # type: ignore[import-untyped]  # noqa: F401 — only import inside this function

    blocks: list[TextBlock] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        ph = float(page.rect.height)
        data = page.get_text("dict")
        for blk in data.get("blocks", []):
            tb = _parse_block(blk, page=pno + 1, page_height=ph)
            if tb is not None:
                blocks.append(tb)
    return blocks


def extract_clean_pages(blocks: list[TextBlock]) -> list[tuple[int, list[str]]]:
    """Extract body segments per page, stopping at the first references heading.

    Pipeline per call:
      1. drop_margins   — remove running headers/footers and page numbers
      2. Group by page, then for each page (in sorted order):
         a. order_by_column — reorder blocks for 2-column layouts
         b. Walk blocks; stop at (and exclude) the first block whose text contains
            a standalone references-heading line (_REFS_HEADING.search); collect
            the preceding blocks as pre-refs content.
         c. If the cutoff fired, stop processing all subsequent pages.
      3. estimate_body_size — computed over pre-refs blocks only (so a long
         same-font bibliography cannot shift the modal size).
      4. page_segments   — walk each page's pre-refs blocks and emit body segments.

    Returns list[tuple[int, list[str]]]: each tuple is (real 1-based PDF page number,
    per-page segment list).  Pages that produce no body segments are OMITTED so that
    the real PDF page numbers are preserved exactly.  Stops at (and excludes) the first
    references heading and all pages that follow it.  body_size is measured exclusively
    over pre-references content.
    """
    kept = drop_margins(blocks)

    # Group blocks by page
    by_page: dict[int, list[TextBlock]] = {}
    for b in kept:
        by_page.setdefault(b.page, []).append(b)

    # Pass 1: collect pre-refs blocks per page, respecting cutoff across pages.
    # Use _REFS_HEADING.search so a heading embedded in a multi-line block is found.
    pre_refs_by_page: list[tuple[int, list[TextBlock]]] = []
    stop = False
    for page in sorted(by_page):
        if stop:
            break
        ordered = order_by_column(by_page[page])
        before_refs: list[TextBlock] = []
        for b in ordered:
            if _REFS_HEADING.search(b.text):
                stop = True
                break
            before_refs.append(b)
        pre_refs_by_page.append((page, before_refs))

    # body_size estimated only over pre-refs blocks.
    all_pre_refs = [b for _, pg_blocks in pre_refs_by_page for b in pg_blocks]
    body_size = estimate_body_size(all_pre_refs)

    # Pass 2: produce segments for each page using the pre-refs blocks.
    # Preserve real PDF page numbers; omit pages with no surviving segments.
    result: list[tuple[int, list[str]]] = []
    for page_num, pg_blocks in pre_refs_by_page:
        segs = page_segments(pg_blocks, body_size)
        if segs:
            result.append((page_num, segs))
    return result
