import json
import logging
import re
from pathlib import Path
from typing import Any
from probatio.interfaces import LLMClient
from probatio.models import Citation, Reference

# --- citation-marker + structure regexes -------------------------------------
_NUM = re.compile(r"\[\s*(\d+(?:\s*[–-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[–-]\s*\d+)?)*)\s*\]")
_AY = re.compile(r"\(([^()]*\b(?:19|20)\d{2}[a-z]?[^()]*)\)")
_REFS_HEADING = re.compile(
    r"(?im)^\s*(references|bibliography|works cited|literature cited)\s*:?\s*$")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z])")


def _expand_numeric(inside: str) -> list[str]:
    keys: list[str] = []
    for part in inside.split(","):
        part = part.strip()
        rng = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            if 0 <= b - a < 100:                 # sanity cap on range size
                keys.extend(str(i) for i in range(a, b + 1))
        elif part.isdigit():
            keys.append(part)
    return keys


def _parse_authoryear(inside: str) -> list[str]:
    keys: list[str] = []
    for part in inside.split(";"):
        m = re.search(r"([A-Z][A-Za-z'’\-]+).*?\b((?:19|20)\d{2})[a-z]?", part)
        if m:
            keys.append(f"{m.group(1)} {m.group(2)}")
    return keys


def find_markers(text: str) -> list[list[str]]:
    """Citation marker groups in document order; each group is a list of reference keys.

    Numeric: '[3,4,5]' -> ['3','4','5']; '[3–5]' -> ['3','4','5'].
    Author-year: '(Smith et al., 2020; Jones, 2019)' -> ['Smith 2020','Jones 2019'].
    """
    found: list[tuple[int, list[str]]] = []
    for m in _NUM.finditer(text):
        found.append((m.start(), _expand_numeric(m.group(1))))
    for m in _AY.finditer(text):
        found.append((m.start(), _parse_authoryear(m.group(1))))
    found.sort(key=lambda t: t[0])
    return [keys for _, keys in found if keys]


def strip_markers(text: str) -> str:
    """The claim sentence with its citation markers removed and whitespace normalised."""
    t = _AY.sub("", _NUM.sub("", text))
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)   # drop the space a removed marker left before punctuation
    return re.sub(r"\s+", " ", t).strip()


_LINE_NUMBER = re.compile(r"\s*\d{1,4}\s*")


def strip_line_numbers(text: str) -> str:
    """Drop reviewer-PDF margin line-numbers, which PyMuPDF emits as standalone lines.

    A body-text line that is ONLY a 1–4 digit integer is a margin line-number (or the
    left-margin '1 2 3 …' column), never prose: inline numbers — years, quantities, ranges,
    '[3]' markers — are part of a larger line and survive. Applied to the claim body only,
    NOT the reference list, where a bare number on its own line can be a citation key.
    """
    return "\n".join(ln for ln in text.splitlines() if not _LINE_NUMBER.fullmatch(ln))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def references_block(full_text: str) -> str:
    """Text from the LAST References/Bibliography heading to the end ('' if none)."""
    last = None
    for m in _REFS_HEADING.finditer(full_text):
        last = m
    return full_text[last.end():] if last else ""


# A reference-list entry begins with a numeric marker: [12] / (12) / 12. / 12)
_NUM_ENTRY = re.compile(r"^\s*(?:\[\s*\d+\s*\]|\(\s*\d+\s*\)|\d+[.)])\s+")
# A publication year — used to tell a real author-year entry from a stray continuation line.
_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")


def _marker_number(line: str) -> int | None:
    """The integer of a leading reference marker, or None if the line has no marker."""
    m = _NUM_ENTRY.match(line)
    if not m:
        return None
    d = re.search(r"\d+", m.group(0))
    return int(d.group(0)) if d else None


def _entries_by_marker(lines: list[str]) -> list[str]:
    """Group lines into entries, each opening at an *ascending* numeric-marker line.

    A marker line opens a new entry only when its number advances the enumeration
    (the first marker, then +1 each time). So a hard-wrapped continuation line that
    happens to start with a number — a year ``2020.`` or a DOI — does NOT falsely
    break an entry (its number is not last+1), keeping the title/DOI/year glued to
    the marker they belong to.
    """
    entries: list[str] = []
    cur: list[str] = []
    last_num: int | None = None
    for ln in lines:
        num = _marker_number(ln)
        is_start = num is not None and (last_num is None or num == last_num + 1)
        if is_start and cur:
            entries.append("\n".join(cur))
            cur = []
        if is_start:
            last_num = num
        cur.append(ln)
    if cur:
        entries.append("\n".join(cur))
    return [e for e in entries if e.strip()]


def _looks_numbered(nonblank: list[str]) -> bool:
    """True when >=3 lines open with numeric markers that ascend like an enumeration.

    Keys off the marker *sequence*, not its share of physical lines, so a bibliography
    whose entries hard-wrap across several lines (markers sparse) is still recognised —
    this is the common PyMuPDF case and the previous fraction-of-lines test missed it.
    """
    nums = [n for n in (_marker_number(ln) for ln in nonblank) if n is not None]
    if len(nums) < 3:
        return False
    ascending = sum(1 for a, b in zip(nums, nums[1:]) if b > a)
    return ascending >= 0.7 * (len(nums) - 1)


def _entries_by_blankline(block: str) -> list[str] | None:
    """Split an author-year list on blank lines, re-gluing any fragment that lacks a
    year (a stray internal blank inside one entry) back onto the previous entry.
    Returns None when there is no real blank-line structure to split on.
    """
    parts = [e for e in re.split(r"\n\s*\n", block) if e.strip()]
    if len(parts) <= 1:
        return None
    merged: list[str] = []
    for p in parts:
        if merged and not _YEAR.search(p):
            merged[-1] = merged[-1] + "\n\n" + p        # bias toward keeping text together
        else:
            merged.append(p)
    return merged


def _split_into_entries(block: str) -> list[str]:
    """Best-effort split of a bibliography into individual entries.

    Numbered lists split at ascending marker lines; author-year lists split on blank
    lines (year-aware); with neither structure, every non-blank line is its own entry
    so the packer can still bound chunk size. Always covers the whole block.
    """
    lines = block.split("\n")
    nonblank = [ln for ln in lines if ln.strip()]
    if not nonblank:
        return []
    if _looks_numbered(nonblank):
        return _entries_by_marker(lines)
    by_blank = _entries_by_blankline(block)
    if by_blank is not None:
        return by_blank
    return nonblank                                  # last resort: one entry per line


def split_reference_block(block: str, *, budget: int = 6000) -> list[str]:
    """Split a bibliography into ordered, LLM-sized chunks at entry boundaries.

    A block within ``budget`` returns as a single chunk (preserving the prior
    single-call behaviour). Whole entries are greedily packed up to ``budget``; an
    entry larger than ``budget`` becomes its own (over-budget) chunk rather than
    being cut. The concatenation of chunks covers the whole block.
    """
    if not block.strip():
        return []
    if len(block) <= budget:
        return [block.strip("\n")]
    entries = _split_into_entries(block)
    if not entries:
        return [block.strip("\n")]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for e in entries:
        e_len = len(e) + 1
        if cur and cur_len + e_len > budget:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(e)
        cur_len += e_len
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _loads(raw: str) -> Any:
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    try:
        return json.loads(m.group(1) if m else raw)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def _year(v: Any) -> int | None:
    try:
        return int(str(v)[:4])
    except (TypeError, ValueError):
        return None


_REFS_SYSTEM = (
    "Parse this academic reference list into structured JSON. Return ONLY a JSON array; each "
    'element {"key":"...","authors":["..."],"year":2020,"title":"...","doi":"...","raw":"..."}. '
    'key = the entry number ("12") for numbered lists, else "Surname Year". '
    "Use null for unknown fields; keep raw = the entry's original text."
)
_SCOPE_SYSTEM = (
    "Each line is 'INDEX: claim' from a manuscript. Label each claim 'empirical' (a factual/"
    "quantitative assertion to verify against a cited source) or 'non_checkable' (a method "
    "attribution, a 'see review' pointer, or general background credit). "
    'Return ONLY a JSON array of {"i":INDEX,"kind":"empirical|non_checkable"}.'
)


class PyMuPDFManuscriptParser:
    """Hybrid: regex for markers/sentences/heading, LLM for the fuzzy bits — structuring
    the reference list and tagging each claim's checkability."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def parse_references(self, manuscript_pdf: Path) -> list[Reference]:
        """Parse ONLY the bibliography (for acquisition): read the PDF text, isolate the
        references block, and structure it. No claim mining, no scope-tag LLM call."""
        import fitz  # type: ignore[import-untyped]  # PyMuPDF; lazy so pure helpers stay dep-free
        doc = fitz.open(str(manuscript_pdf))
        plain_text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
        doc.close()
        return await self._parse_refs(references_block(plain_text))

    async def parse(self, manuscript_pdf: Path) -> tuple[list[Citation], list[Reference]]:
        import fitz  # type: ignore[import-untyped]  # PyMuPDF; lazy so pure helpers stay dep-free
        from probatio.manuscript_layout import extract_blocks, extract_clean_pages
        doc = fitz.open(str(manuscript_pdf))
        plain_pages = [doc[i].get_text() for i in range(doc.page_count)]
        try:
            clean_pages = extract_clean_pages(extract_blocks(doc))
        except Exception as e:  # noqa: BLE001 — layout extraction must never abort the parse
            logging.getLogger(__name__).warning(
                "layout extraction failed, falling back to plain text: %s", e)
            clean_pages = []
        doc.close()

        references = await self._parse_refs(references_block("\n".join(plain_pages)))
        # Gate fallback on CITATION COUNT: mine layout first; only fall back if zero citations
        # found (e.g. all segments were junk with no markers), not merely on segment presence.
        citations = self._mine_segments(clean_pages)
        if not citations:
            citations = self._mine_citations(plain_pages)
        await self._tag_scope(citations)
        return citations, references

    def _mine_segments(self, clean_pages: list[tuple[int, list[str]]]) -> list[Citation]:
        """Mine citations from layout-aware segments.

        clean_pages is list[tuple[int, list[str]]]: each tuple is (real 1-based PDF page
        number, segments).  manuscript_page is stamped with the real PDF page number, not
        a re-enumerated index, so citations from page 3 (even when pages 1–2 had no
        surviving segments) correctly carry manuscript_page=3.
        """
        citations: list[Citation] = []
        n = 0
        for page_num, segments in clean_pages:
            for segment in segments:
                for sent in split_sentences(strip_line_numbers(segment)):
                    for group in find_markers(sent):
                        n += 1
                        citations.append(Citation(
                            id=f"c{n:04d}", claim=strip_markers(sent),
                            ref_keys=group, manuscript_page=page_num,
                        ))
        return citations

    def _mine_citations(self, pages: list[str]) -> list[Citation]:
        citations: list[Citation] = []
        n = 0
        for pno, ptext in enumerate(pages, 1):
            head = _REFS_HEADING.search(ptext)
            body = ptext[: head.start()] if head else ptext
            for sent in split_sentences(strip_line_numbers(body)):
                for group in find_markers(sent):
                    n += 1
                    citations.append(Citation(
                        id=f"c{n:04d}", claim=strip_markers(sent),
                        ref_keys=group, manuscript_page=pno))
            if head:
                break  # everything past the bibliography heading is references, not claims
        return citations

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        """A failure worth retrying: timeouts, transport errors, and retryable HTTP
        statuses — notably ollama's 503 'model is loading' on a cold model. A 4xx
        (bad request / model-not-found / auth) is permanent and fails fast."""
        import httpx
        if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
            return True
        return (isinstance(e, httpx.HTTPStatusError)
                and e.response.status_code in (429, 500, 502, 503, 504))

    async def _complete_with_retry(self, *, system: str, user: str, think: bool,
                                   attempts: int = 2) -> str:
        """Call the LLM, retrying only TRANSIENT failures (see _is_transient); a permanent
        error fails fast rather than burning another full timeout. Returns '' when every
        attempt fails — the caller treats '' as a failed chunk and skips it, never aborting
        the whole parse."""
        log = logging.getLogger(__name__)
        for attempt in range(1, attempts + 1):
            try:
                return await self.llm.complete(system=system, user=user, think=think)
            except Exception as e:  # noqa: BLE001 - classify, then retry transient or skip chunk
                if self._is_transient(e) and attempt < attempts:
                    log.warning("reference-chunk transient error (attempt %d/%d): %s",
                                attempt, attempts, e)
                    continue
                kind = "transient, retries exhausted" if self._is_transient(e) else "permanent"
                log.warning("reference-chunk call failed (%s); skipping chunk: %s", kind, e)
                return ""
        return ""

    @staticmethod
    def _to_reference(e: Any) -> Reference | None:
        if not isinstance(e, dict):
            return None
        key = str(e.get("key", "")).strip()
        if not key:
            return None
        raw_authors = e.get("authors")
        authors = [str(a) for a in raw_authors if a] if isinstance(raw_authors, list) else []
        return Reference(
            key=key, raw=str(e.get("raw", ""))[:1000],
            title=(str(e["title"]) if e.get("title") else None),
            doi=(str(e["doi"]) if e.get("doi") else None),
            authors=authors[:20], year=_year(e.get("year")))

    @staticmethod
    def _dedupe(refs: list[Reference]) -> list[Reference]:
        """Collapse duplicate keys, keeping first position but upgrading to the richer
        record (more populated fields) when a later duplicate carries more — so a sparse
        record seen first does not shadow a fuller one from a later chunk."""
        def populated(r: Reference) -> int:
            return sum(bool(x) for x in (r.title, r.doi, r.authors, r.year))
        by_key: dict[str, Reference] = {}
        order: list[str] = []
        for r in refs:
            if r.key not in by_key:
                by_key[r.key] = r
                order.append(r.key)
            elif populated(r) > populated(by_key[r.key]):
                by_key[r.key] = r
        return [by_key[k] for k in order]

    async def _parse_refs(self, block: str) -> list[Reference]:
        if not block.strip():
            return []
        chunks = split_reference_block(block)
        log = logging.getLogger(__name__)
        refs: list[Reference] = []
        for idx, chunk in enumerate(chunks, 1):
            # think=False: a reference list is a huge structured-output call; reasoning
            # tokens make it run unbounded (and add no value to mechanical parsing).
            data = _loads(await self._complete_with_retry(
                system=_REFS_SYSTEM, user=chunk, think=False))
            if not isinstance(data, list):
                log.warning("reference chunk %d/%d did not parse; its entries are skipped",
                            idx, len(chunks))
                continue
            before = len(refs)
            refs.extend(r for r in (self._to_reference(e) for e in data) if r is not None)
            if len(refs) == before:
                log.warning("reference chunk %d/%d yielded 0 references", idx, len(chunks))
        return self._dedupe(refs)

    async def _tag_scope(self, citations: list[Citation]) -> None:
        if not citations:
            return
        numbered = "\n".join(f"{i}: {c.claim}" for i, c in enumerate(citations))
        data = _loads(await self.llm.complete(system=_SCOPE_SYSTEM, user=numbered[:12000]))
        if not isinstance(data, list):
            return  # default stays 'empirical' for every claim
        for item in data:
            if isinstance(item, dict) and item.get("kind") == "non_checkable":
                i = item.get("i")
                if isinstance(i, int) and 0 <= i < len(citations):
                    citations[i].kind = "non_checkable"
