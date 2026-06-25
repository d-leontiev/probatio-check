import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional
from probatio.models import Citation, Reference, CitationCheck, ResolutionStatus

_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# path -> (doi, title); injected so the resolver is testable without real PDFs.
MetaFn = Callable[[Path], tuple[Optional[str], Optional[str]]]


def norm_doi(doi: str) -> str:
    d = doi.strip().lower().rstrip(".")
    return d.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "").strip()


_DOI_SLUG_BAD = re.compile(r"[^a-z0-9._-]+")


def doi_slug(s: str) -> str:
    """Filesystem-safe slug of a DOI/id: lowercase, any non-[a-z0-9._-] char -> '_'.

    The SINGLE source of truth used both to NAME an acquired PDF (probatio.acquire) and to
    MATCH it here, so a DOI's mandatory '/' can never make the two diverge. A bare DOI like
    '10.1038/s41586-020-2649-2' can never appear verbatim in a filename (no '/'), so the
    filename match must compare slugged forms, not the raw normalized DOI."""
    return _DOI_SLUG_BAD.sub("_", s.lower()).strip("_") or "ref"


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", t.lower())


def _title_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, " ".join(_norm_title(a).split()),
                           " ".join(_norm_title(b).split())).ratio()


def _default_extract_meta(pdf_path: Path) -> tuple[Optional[str], Optional[str]]:
    """DOI (regex over first pages) + a best-effort title (PDF metadata, else first long line)."""
    import fitz  # type: ignore[import-untyped]  # PyMuPDF
    try:
        doc = fitz.open(str(pdf_path))
        text = "".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
        meta_title = ((doc.metadata or {}).get("title") or "").strip()
        doc.close()
    except Exception:  # noqa: BLE001 - one corrupt/scanned PDF must not abort the whole folder
        return None, None
    m = _DOI.search(text)
    doi = m.group(0) if m else None
    title: Optional[str] = meta_title if (
        len(meta_title) > 10 and "microsoft word" not in meta_title.lower()) else None
    if not title:
        for line in text.splitlines():
            if len(line.strip()) >= 20:
                title = line.strip()
                break
    return doi, title


class CitationResolver:
    """Resolve each (citation, ref_key) to its source PDF. Confidence-gated: when a match is
    uncertain it reports 'ambiguous' (for a human pick) rather than guessing wrong."""

    def __init__(self, *, extract_meta: MetaFn | None = None,
                 title_threshold: float = 0.62, ambiguous_margin: float = 0.08):
        self._extract = extract_meta or _default_extract_meta
        self.title_threshold = title_threshold
        self.ambiguous_margin = ambiguous_margin

    async def resolve(self, citations: list[Citation], references: list[Reference],
                      refs_dir: Path) -> list[CitationCheck]:
        refs_by_key = {r.key: r for r in references}
        pdf_meta = {p: self._extract(p) for p in self._pdfs(refs_dir)}
        checks: list[CitationCheck] = []
        for cit in citations:
            for key in cit.ref_keys:
                ref = self._find_reference(key, refs_by_key, references)
                if ref is None:
                    checks.append(CitationCheck(
                        citation=cit, ref_key=key, resolution="unresolved_marker"))
                    continue
                pdf, status = self._match_pdf(ref, pdf_meta)
                checks.append(CitationCheck(
                    citation=cit, ref_key=key, reference=ref,
                    source_pdf=pdf, resolution=status))
        return checks

    @staticmethod
    def _pdfs(refs_dir: Path) -> list[Path]:
        return sorted(p for p in Path(refs_dir).rglob("*") if p.suffix.lower() == ".pdf")

    @staticmethod
    def _find_reference(key: str, refs_by_key: dict[str, Reference],
                        references: list[Reference]) -> Optional[Reference]:
        if key in refs_by_key:
            return refs_by_key[key]
        m = re.fullmatch(r"(.+?)\s+((?:19|20)\d{2})", key)   # author-year fallback "Surname YYYY"
        if m:
            surname, year = m.group(1).lower(), int(m.group(2))
            for r in references:
                if r.year == year and r.authors and surname in r.authors[0].lower():
                    return r
        return None

    def _match_pdf(self, ref: Reference,
                   pdf_meta: dict[Path, tuple[Optional[str], Optional[str]]]
                   ) -> tuple[Optional[Path], ResolutionStatus]:
        # 1. DOI exact (PDF text/metadata, or DOI embedded in the filename)
        if ref.doi:
            rd = norm_doi(ref.doi)
            # Pass 1: an exact embedded DOI wins across ALL files (never lose to a substring).
            for path, (doi, _t) in pdf_meta.items():
                if doi and norm_doi(doi) == rd:
                    return path, "resolved"
            # Pass 2: the DOI slug as a DELIMITED filename token (acquire names files
            # "<doi_slug>__<key>.pdf"/"<doi_slug>.pdf"). Anchored — never a bare substring —
            # so DOI 10.1/abc cannot match a sibling file for DOI 10.1/abc.123.
            slug = doi_slug(rd)
            for path in pdf_meta:
                stem = path.name.lower().removesuffix(".pdf")
                if stem == slug or stem.startswith(slug + "__"):
                    return path, "resolved"
        # 2. title fuzzy, with ambiguity guard
        if ref.title:
            scored = sorted(
                ((_title_sim(ref.title, t), p) for p, (_d, t) in pdf_meta.items() if t),
                key=lambda s: s[0], reverse=True)
            if scored and scored[0][0] >= self.title_threshold:
                best = scored[0][0]
                second = scored[1][0] if len(scored) > 1 else 0.0
                if second >= self.title_threshold and (best - second) < self.ambiguous_margin:
                    return None, "ambiguous"
                return scored[0][1], "resolved"
        return None, "no_pdf"
