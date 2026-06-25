import re
from pathlib import Path
import fitz  # type: ignore[import-untyped]  # PyMuPDF
from probatio.interfaces import HighlightRect


def _fragments(snippet: str, min_len: int = 30, cap: int = 20) -> list[str]:
    """Sentence-ish fragments of a snippet, long enough to match a unique span."""
    out: list[str] = []
    seen: set[str] = set()
    for p in re.split(r"(?<=[.;:])\s+|\n+", snippet):
        p = p.strip()
        if len(p) >= min_len and p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= cap:
            break
    return out


class PyMuPDFHighlighter:
    """snippet -> rectangles: whole-snippet match, else highlight every fragment found."""

    def locate(self, pdf_path: Path, snippet: str, page: int | None = None) -> list[HighlightRect]:
        doc = fitz.open(pdf_path)
        try:
            page_indices = [page - 1] if page is not None else list(range(doc.page_count))
            rects = self._search(doc, page_indices, snippet)        # whole snippet verbatim
            if rects:
                return rects
            frags = _fragments(snippet)                             # else highlight the whole
            rects = self._search_many(doc, page_indices, frags)     # passage, fragment by fragment
            if not rects:                                           # last resort: every page
                rects = self._search_many(doc, list(range(doc.page_count)), frags)
            return rects
        finally:
            doc.close()

    def _search_many(self, doc, page_indices, needles: list[str]) -> list[HighlightRect]:
        out: list[HighlightRect] = []
        for n in needles:
            out += self._search(doc, page_indices, n)
        return out

    def _search(self, doc, page_indices, needle: str) -> list[HighlightRect]:
        out = []
        for pi in page_indices:
            if pi < 0 or pi >= doc.page_count:
                continue
            for q in doc[pi].search_for(needle):
                out.append(HighlightRect(page=pi + 1, x0=q.x0, y0=q.y0, x1=q.x1, y1=q.y1))
        return out

    def render_page_png(self, pdf_path: Path, snippet: str, page: int | None = None,
                        dpi: int = 120,
                        color: tuple[float, float, float] | None = None
                        ) -> tuple[bytes, int] | None:
        """Render the page containing the snippet to a PNG, with the snippet highlighted.

        Highlights are drawn by PyMuPDF in the PDF's own coordinate space, so the
        rendered image is always correct — no client-side coordinate math. ``color``
        is an optional RGB triple (0..1) for the highlight stroke (e.g. green for a
        supported verdict, red for unsupported); the viewer default (yellow) is used
        when omitted. Returns (png_bytes, page_number), or None if the PDF has no pages.
        """
        rects = self.locate(pdf_path, snippet, page)
        doc = fitz.open(pdf_path)
        try:
            if doc.page_count == 0:
                return None
            target = rects[0].page if rects else (page or 1)
            target = min(max(target, 1), doc.page_count)
            pg = doc[target - 1]
            for r in rects:
                if r.page == target:
                    annot = pg.add_highlight_annot(fitz.Rect(r.x0, r.y0, r.x1, r.y1))
                    if color is not None:
                        annot.set_colors(stroke=color)
                        annot.update()
            return pg.get_pixmap(dpi=dpi).tobytes("png"), target
        finally:
            doc.close()
