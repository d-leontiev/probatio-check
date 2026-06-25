import fitz
from fastapi.testclient import TestClient
import probatio.web.app as webapp
from probatio.web.app import create_check_app, create_app
from probatio.config import Settings
from probatio.web.serve import load_check_report
from probatio.report import write_citation_sidecar
from probatio.models import (AcquisitionReport, AcquisitionResult, Citation,
                             CitationCheck, CitationReport, EvidenceContext,
                             Reference)


def _pdf(path, text):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


async def _aiter(v):
    return v


def _report(refs):
    _pdf(refs / "a.pdf", "The melting point is 150 C as reported in our study.")
    p1 = EvidenceContext(id="a::p1::check::0", paper_id="a",
                         snippet="The melting point is 150 C", page=1, score=0.91)
    p2 = EvidenceContext(id="a::p2::check::1", paper_id="a",
                         snippet="Measured at standard pressure", page=1,
                         score=0.50, rcs_summary="states the melting point")
    return CitationReport(
        manuscript="m.pdf", coverage={"supported": 1, "unsupported": 1},
        checks=[
            CitationCheck(
                citation=Citation(id="c1", claim="melting point is 150 C",
                                  ref_keys=["1"], section="Results",
                                  manuscript_page=4, kind="empirical"),
                ref_key="1", source_pdf=refs / "a.pdf", resolution="resolved",
                reference=Reference(key="1", raw="Smith 2020", title="Melting study",
                                    authors=["Smith", "Doe"], year=2020, doi="10.1/x"),
                verdict="supported", confidence=0.9, passages=[p1, p2]),
            CitationCheck(
                citation=Citation(id="c2", claim="it boils at 9000 C", ref_keys=["2"]),
                ref_key="2", source_pdf=refs / "a.pdf", resolution="resolved",
                verdict="unsupported", rationale="not stated in source"),
        ])


def test_citations_problems_first_and_coverage(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs, out_dir=tmp_path))
    body = client.get("/api/citations").json()
    assert body["coverage"]["supported"] == 1
    assert body["checks"][0]["verdict"] == "unsupported"   # problems first
    assert body["checks"][0]["claim"] == "it boils at 9000 C"


def test_override_persists(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs, out_dir=tmp_path))
    r = client.post("/api/override", json={"id": "c1:1", "verdict": "overstated", "note": "hedged"})
    assert r.status_code == 200 and r.json()["status"] == "overstated"
    again = client.get("/api/citations").json()
    c1 = next(c for c in again["checks"] if c["id"] == "c1:1")
    assert c1["human_override"] == "overstated" and c1["note"] == "hedged"
    assert (tmp_path / "citations.json").exists()          # persisted to disk


def test_override_rejects_bad_verdict(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs))
    assert client.post("/api/override", json={"id": "c1:1", "verdict": "great"}).status_code == 400


def test_page_image_highlights_source(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs))
    img = client.get("/api/page-image/a::p1::check::0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_page_image_resolves_source_with_spaces(tmp_path):
    # Real EndNote PDFs have spaces in their filenames; the page image must still
    # render. The endpoint resolves via the check's source_pdf, not a sanitized id.
    refs = tmp_path / "refs"
    refs.mkdir()
    _pdf(refs / "Random Forests.pdf", "Random forests are an ensemble of decision trees.")
    passage = EvidenceContext(id="Random Forests::p1::check::0", paper_id="Random Forests",
                              snippet="ensemble of decision trees", page=1)
    rep = CitationReport(
        manuscript="m.pdf", coverage={},
        checks=[CitationCheck(
            citation=Citation(id="c1", claim="RF is an ensemble", ref_keys=["1"]),
            ref_key="1", source_pdf=refs / "Random Forests.pdf", resolution="resolved",
            verdict="supported", passages=[passage])])
    client = TestClient(create_check_app(report=rep, refs_dir=refs))
    img = client.get("/api/page-image/Random%20Forests::p1::check::0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_index_served(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs))
    assert client.get("/").status_code == 200


def test_citations_enriched_fields(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs))
    body = client.get("/api/citations").json()
    c1 = next(c for c in body["checks"] if c["id"] == "c1:1")
    assert c1["reviewed"] is False
    assert c1["confidence"] == 0.9
    assert c1["section"] == "Results" and c1["manuscript_page"] == 4
    assert c1["kind"] == "empirical" and c1["resolution"] == "resolved"
    assert c1["reference"]["title"] == "Melting study"
    assert c1["reference"]["authors"] == ["Smith", "Doe"]
    assert c1["reference"]["year"] == 2020 and c1["reference"]["doi"] == "10.1/x"
    assert {p["id"] for p in c1["passages"]} == {"a::p1::check::0", "a::p2::check::1"}
    assert any(p["rcs_summary"] == "states the melting point" for p in c1["passages"])
    assert c1["passage_id"] == "a::p1::check::0" and c1["page"] == 1
    assert c1["reference"]["key"] == "1" and c1["reference"]["raw"] == "Smith 2020"
    c2 = next(c for c in body["checks"] if c["id"] == "c2:2")
    assert c2["reference"] is None and c2["passages"] == []
    assert c2["passage_id"] is None


def test_static_assets_served(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs))
    assert client.get("/").status_code == 200
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    css = client.get("/static/app.css")
    assert css.status_code == 200 and "css" in css.headers["content-type"]


def test_load_check_report_roundtrip(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    rep = _report(refs)
    write_citation_sidecar(rep, tmp_path)
    loaded = load_check_report(tmp_path)
    assert loaded.manuscript == "m.pdf" and len(loaded.checks) == 2
    assert loaded.checks[0].source_pdf.name == "a.pdf"


def test_override_verdict_marks_reviewed(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs,
                                         out_dir=tmp_path))
    r = client.post("/api/override", json={"id": "c1:1", "verdict": "overstated"})
    assert r.status_code == 200 and r.json()["reviewed"] is True
    c1 = next(c for c in client.get("/api/citations").json()["checks"]
              if c["id"] == "c1:1")
    assert c1["reviewed"] is True and c1["human_override"] == "overstated"


def test_override_reviewed_without_verdict_keeps_override_null(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs,
                                         out_dir=tmp_path))
    assert client.post("/api/override",
                       json={"id": "c1:1", "reviewed": True}).json()["reviewed"] is True
    c1 = next(c for c in client.get("/api/citations").json()["checks"]
              if c["id"] == "c1:1")
    assert c1["reviewed"] is True and c1["human_override"] is None


def test_override_clear_wins_when_verdict_also_sent(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs,
                                         out_dir=tmp_path))
    # clear_override takes precedence over a simultaneously-sent verdict (the elif)
    r = client.post("/api/override",
                    json={"id": "c1:1", "verdict": "overstated", "clear_override": True})
    assert r.status_code == 200
    c1 = next(c for c in client.get("/api/citations").json()["checks"]
              if c["id"] == "c1:1")
    assert c1["human_override"] is None


def test_override_clear_keeps_reviewed(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    client = TestClient(create_check_app(report=_report(refs), refs_dir=refs,
                                         out_dir=tmp_path))
    client.post("/api/override", json={"id": "c1:1", "verdict": "overstated"})
    r = client.post("/api/override", json={"id": "c1:1", "clear_override": True})
    assert r.status_code == 200 and r.json()["status"] == "supported"
    c1 = next(c for c in client.get("/api/citations").json()["checks"]
              if c["id"] == "c1:1")
    assert c1["human_override"] is None and c1["reviewed"] is True


def test_launcher_serves_index_and_guard(tmp_path):
    client = TestClient(create_app(Settings()))
    assert client.get("/").status_code == 200
    g = client.get("/api/guard").json()
    assert set(g) >= {"local", "verify_model", "embedding_model", "ollama_api_base"}
    assert g["local"] is True                 # default Settings are local ollama


def test_launcher_status_idle(tmp_path):
    client = TestClient(create_app(Settings()))
    assert client.get("/api/run-status").json()["phase"] == "idle"


def test_acquire_fail_closed(tmp_path, monkeypatch):
    from probatio.config import Settings
    s = Settings(verify_model="claude-haiku-4-5-20251001")    # cloud -> refused
    client = TestClient(create_app(s))
    pdf = tmp_path / "m.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    r = client.post("/api/acquire", json={"manuscript_path": str(pdf)})
    assert r.status_code == 409


def test_acquire_background_then_references(tmp_path, monkeypatch):
    from probatio.config import Settings
    pdf = tmp_path / "m.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    async def fake_parse_refs(self, p):
        from probatio.models import Reference
        return [Reference(key="1", raw="1", doi="10.1/x")]
    monkeypatch.setattr("probatio.manuscript.PyMuPDFManuscriptParser.parse_references",
                        fake_parse_refs)

    async def fake_acquire(refs, refs_dir, *, client, max_concurrency=4, on_progress=None):
        if on_progress:
            on_progress("fetching", 1, 1)
        return AcquisitionReport(results=[AcquisitionResult(ref_key="1", status="paywalled")],
                                 summary={"paywalled": 1})
    monkeypatch.setattr(webapp, "acquire_open_access", fake_acquire)
    monkeypatch.setattr(webapp, "UnpaywallOpenAlexClient", lambda **k: object())

    client = TestClient(create_app(Settings()))
    r = client.post("/api/acquire", json={"manuscript_path": str(pdf)})
    assert r.status_code == 202
    # background task runs on the TestClient event loop; poll until awaiting_refs
    for _ in range(50):
        if client.get("/api/run-status").json()["phase"] == "awaiting_refs":
            break
    refs = client.get("/api/references").json()
    assert refs["summary"]["paywalled"] == 1 and refs["results"][0]["ref_key"] == "1"


def test_acquire_rejects_bad_manuscript(tmp_path):
    from probatio.config import Settings
    client = TestClient(create_app(Settings()))
    r = client.post("/api/acquire", json={"manuscript_path": str(tmp_path / "nope.pdf")})
    assert r.status_code == 400


def test_drop_refs_after_acquire(tmp_path, monkeypatch):
    from probatio.config import Settings
    from probatio.models import Reference, AcquisitionReport
    pdf = tmp_path / "m.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr("probatio.manuscript.PyMuPDFManuscriptParser.parse_references",
                        lambda self, p: _aiter([Reference(key="1", raw="1")]))
    import probatio.web.app as webapp

    async def fake_acquire(refs, refs_dir, *, client, max_concurrency=4, on_progress=None):
        return AcquisitionReport(results=[], summary={})
    monkeypatch.setattr(webapp, "acquire_open_access", fake_acquire)
    monkeypatch.setattr(webapp, "UnpaywallOpenAlexClient", lambda **k: object())
    client = TestClient(create_app(Settings()))
    client.post("/api/acquire", json={"manuscript_path": str(pdf)})
    for _ in range(50):
        if client.get("/api/run-status").json()["phase"] == "awaiting_refs":
            break
    files = [("files", ("Good Paper.pdf", b"%PDF-1.7 body", "application/pdf")),
             ("files", ("bad.pdf", b"<html>no</html>", "application/pdf"))]
    r = client.post("/api/drop-refs", files=files)
    assert r.status_code == 200 and r.json()["added"] == 1
    assert (pdf.with_name("m-refs") / "Good Paper.pdf").exists()
