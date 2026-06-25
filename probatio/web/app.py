import asyncio
from pathlib import Path
from typing import Optional, cast
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from probatio.config import (Settings, assert_local_only, ConfidentialityError,
                             make_verify_client)
from probatio.acquire import acquire_open_access, UnpaywallOpenAlexClient
from probatio.manuscript import PyMuPDFManuscriptParser
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


class RunState:
    """Mutable state for one web run, shared across phases of the two-step flow.

    Phases: idle|acquiring|awaiting_refs|checking|done|error.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
        self.manuscript: Path | None = None
        self.refs_dir: Path | None = None
        self.out_dir: Path | None = None
        self.acquire_report = None      # AcquisitionReport | None
        self.check_report: CitationReport | None = None
        self.phase: str = "idle"        # idle|acquiring|awaiting_refs|checking|done|error
        self.progress: dict = {"step": "", "i": 0, "n": 0}
        self.error: str = ""


def _build_app(run: RunState) -> FastAPI:
    """Citation-audit UI: problems-first list, click → highlighted source passage + verdict,
    with human override that persists back to citations.json. Reuses the highlight endpoint.

    All endpoints read mutable state from ``run`` so a single app serves every phase."""
    from probatio.report import _STATUS_ORDER, _status, write_citation_sidecar
    app = FastAPI(title="probatio-check")
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    _ALLOWED = {"supported", "partially", "overstated", "unsupported",
                "not_found", "not_a_claim", "unchecked"}

    def _report() -> CitationReport:
        if run.check_report is None:
            raise HTTPException(409, "no check report yet")
        return run.check_report

    def _passage_index():
        rep = run.check_report
        return {p.id: (p, chk) for chk in (rep.checks if rep else []) for p in chk.passages}

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
        rdir = Path(run.refs_dir) if run.refs_dir is not None else None
        if rdir is not None:
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

    @app.get("/api/guard")
    def guard():
        s = run.settings or Settings()
        try:
            assert_local_only(s)
            local, detail = True, "all components local"
        except ConfidentialityError as e:
            local, detail = False, str(e)
        return {"local": local, "detail": detail, "verify_model": s.verify_model,
                "embedding_model": s.embedding_model, "ollama_api_base": s.ollama_api_base}

    @app.get("/api/run-status")
    def run_status():
        return {"phase": run.phase, "step": run.progress.get("step", ""),
                "i": run.progress.get("i", 0), "n": run.progress.get("n", 0),
                "done": run.phase == "done", "error": run.error}

    class AcquireBody(BaseModel):
        manuscript_path: str
        refs_dir: str | None = None

    def _set_progress(step: str, i: int, n: int) -> None:
        run.progress = {"step": step, "i": i, "n": n}

    @app.post("/api/acquire", status_code=202)
    async def acquire(body: AcquireBody):
        s = run.settings or Settings()
        try:
            assert_local_only(s)
        except ConfidentialityError as e:
            raise HTTPException(409, str(e))
        man = Path(body.manuscript_path).expanduser()
        if not man.is_file() or man.suffix.lower() != ".pdf":
            raise HTTPException(400, "manuscript must be an existing .pdf path")
        refs_dir = (Path(body.refs_dir).expanduser() if body.refs_dir
                    else man.with_name(f"{man.stem}-refs"))
        refs_dir.mkdir(parents=True, exist_ok=True)
        run.manuscript, run.refs_dir, run.error = man, refs_dir, ""
        run.phase = "acquiring"
        _set_progress("parsing", 0, 0)

        async def job():
            try:
                parser = PyMuPDFManuscriptParser(make_verify_client(s))
                refs = await parser.parse_references(man)
                client = UnpaywallOpenAlexClient(email=s.unpaywall_email)
                run.acquire_report = await acquire_open_access(
                    refs, refs_dir, client=client, on_progress=_set_progress)
                run.phase = "awaiting_refs"
            except Exception as e:  # noqa: BLE001 - surface failure to the UI, don't crash the server
                run.error, run.phase = str(e), "error"
        asyncio.create_task(job())
        return {"phase": run.phase}

    @app.get("/api/references")
    def references():
        rep = run.acquire_report
        if rep is None:
            raise HTTPException(409, "no acquisition yet")
        order = {"error": 0, "not_found": 1, "paywalled": 2, "fetched": 3, "already_present": 4}
        rows = sorted(rep.results, key=lambda r: order.get(r.status, 9))
        return {"summary": rep.summary, "results": [
            {"ref_key": r.ref_key, "status": r.status, "doi": r.doi,
             "title": r.title, "pdf_path": r.pdf_path} for r in rows]}

    from fastapi import UploadFile, File

    @app.post("/api/drop-refs")
    async def drop_refs(files: list[UploadFile] = File(...)):
        if run.refs_dir is None:
            raise HTTPException(409, "no refs folder yet — start an acquisition first")
        added = 0
        for f in files:
            data = await f.read()
            if not data.startswith(b"%PDF"):
                continue
            name = Path(f.filename or "ref.pdf").name        # strip any path components
            (Path(run.refs_dir) / name).write_bytes(data)
            added += 1
        return {"added": added}

    @app.get("/api/citations")
    def citations():
        rep = _report()
        rows = sorted(rep.checks, key=lambda c: _STATUS_ORDER.get(_status(c), 99))
        return {"manuscript": rep.manuscript, "coverage": rep.coverage,
                "checks": [_row(c) for c in rows]}

    @app.get("/api/page-image/{eid}")
    def page_image(eid: str):
        pc = _passage_index().get(eid)
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
        rep = _report()
        for c in rep.checks:
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
                if run.out_dir is not None:
                    write_citation_sidecar(rep, Path(run.out_dir))
                return {"ok": True, "status": _status(c), "reviewed": c.reviewed}
        raise HTTPException(404, "check not found")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    return app


def create_app(settings: Settings) -> FastAPI:
    """Launcher app: an empty ``RunState`` so the browser can drive acquire + check."""
    return _build_app(RunState(settings))


def create_check_app(*, report: CitationReport, refs_dir: str | Path,
                     out_dir: Optional[Path] = None) -> FastAPI:
    """Audit-only app: a pre-populated ``RunState`` (phase="done"). Backward-compatible
    wrapper over ``_build_app`` — signature and behaviour unchanged."""
    run = RunState()
    run.check_report = report
    run.refs_dir = Path(refs_dir)
    run.out_dir = Path(out_dir) if out_dir is not None else None
    run.phase = "done"
    return _build_app(run)
