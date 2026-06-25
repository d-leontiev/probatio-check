from pathlib import Path
from typing import Optional, cast
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from probatio.highlight import PyMuPDFHighlighter
from probatio.models import CitationReport, CitationVerdict, CitationCheck

# --- "check" mode: citation-audit UI -----------------------------------------
_STATIC = Path(__file__).parent / "static"

# Highlight colour per displayed verdict (PyMuPDF RGB, 0..1): green=supported,
# yellow=partially, orange=overstated, red=unsupported. Others fall back to the
# viewer default (yellow) by passing color=None.
_HL_COLOR = {
    "supported": (0.40, 0.80, 0.45),
    "partially": (1.00, 0.86, 0.30),
    "overstated": (1.00, 0.62, 0.25),
    "unsupported": (1.00, 0.45, 0.45),
}


class OverrideBody(BaseModel):
    id: str
    verdict: Optional[str] = None
    note: str = ""
    reviewed: Optional[bool] = None
    clear_override: bool = False


def create_check_app(*, report: CitationReport, refs_dir: str | Path,
                     out_dir: Optional[Path] = None) -> FastAPI:
    """Citation-audit UI: problems-first list, click → highlighted source passage + verdict,
    with human override that persists back to citations.json. Reuses the highlight endpoint."""
    from probatio.report import _STATUS_ORDER, _status, write_citation_sidecar
    app = FastAPI(title="probatio-check")
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    rdir = Path(refs_dir)
    _passage = {p.id: (p, chk) for chk in report.checks for p in chk.passages}
    _ALLOWED = {"supported", "partially", "overstated", "unsupported",
                "not_found", "not_a_claim", "unchecked"}

    def _cid(c: CitationCheck) -> str:
        return f"{c.citation.id}:{c.ref_key}"

    def _pdf_for(chk: CitationCheck) -> Path:
        """Resolve a passage's source PDF from the check's own ``source_pdf``.

        Uses the real filename (``Path.name`` strips directory components, so it
        cannot escape refs_dir), falling back to the stored absolute path. We never
        reconstruct the path from a sanitized paper id — which is why real filenames
        containing spaces resolve correctly.
        """
        if chk.source_pdf is None:
            raise HTTPException(404, "source pdf unknown")
        target = rdir / Path(chk.source_pdf).name
        if target.is_file():
            return target
        if Path(chk.source_pdf).is_file():
            return Path(chk.source_pdf)
        raise HTTPException(404, "pdf not found")

    def _row(c: CitationCheck) -> dict:
        r = c.reference
        return {
            "id": _cid(c), "status": _status(c), "verdict": c.verdict,
            "human_override": c.human_override, "reviewed": c.reviewed,
            "confidence": c.confidence, "rationale": c.rationale, "note": c.note,
            "claim": c.citation.claim, "ref_key": c.ref_key,
            "section": c.citation.section,
            "manuscript_page": c.citation.manuscript_page,
            "kind": c.citation.kind,
            "reference": ({
                "key": r.key, "title": r.title, "authors": r.authors,
                "year": r.year, "doi": r.doi, "raw": r.raw,
            } if r else None),
            "source": c.source_pdf.name if c.source_pdf else None,
            "resolution": c.resolution,
            "passages": [{
                "id": p.id, "page": p.page, "score": p.score,
                "snippet": p.snippet, "rcs_summary": p.rcs_summary,
            } for p in c.passages],
            "passage_id": c.passages[0].id if c.passages else None,
            "page": c.passages[0].page if c.passages else None,
        }

    @app.get("/api/citations")
    def citations():
        rows = sorted(report.checks, key=lambda c: _STATUS_ORDER.get(_status(c), 99))
        return {"manuscript": report.manuscript, "coverage": report.coverage,
                "checks": [_row(c) for c in rows]}

    @app.get("/api/page-image/{eid}")
    def page_image(eid: str):
        pc = _passage.get(eid)
        if pc is None:
            raise HTTPException(404, "passage not found")
        ctx, chk = pc
        rendered = PyMuPDFHighlighter().render_page_png(
            _pdf_for(chk), ctx.snippet, ctx.page, color=_HL_COLOR.get(_status(chk)))
        if rendered is None:
            raise HTTPException(404, "page not found")
        png, _page = rendered
        return Response(content=png, media_type="image/png")

    @app.post("/api/override")
    def override(body: OverrideBody):
        for c in report.checks:
            if _cid(c) == body.id:
                if body.clear_override:
                    c.human_override = None
                elif body.verdict is not None:
                    if body.verdict not in _ALLOWED:
                        raise HTTPException(400, "invalid verdict")
                    c.human_override = cast(CitationVerdict, body.verdict)
                    c.reviewed = True
                if body.reviewed is not None:
                    c.reviewed = body.reviewed
                c.note = body.note
                if out_dir is not None:
                    write_citation_sidecar(report, Path(out_dir))
                return {"ok": True, "status": _status(c), "reviewed": c.reviewed}
        raise HTTPException(404, "check not found")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    return app
