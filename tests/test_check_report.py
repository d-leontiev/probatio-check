import json
from pathlib import Path
from probatio.report import write_citation_sidecar
from probatio.models import Citation, CitationCheck, CitationReport, EvidenceContext


def test_citation_sidecar_roundtrip_and_problems_first(tmp_path):
    rep = CitationReport(
        manuscript="paper.pdf",
        coverage={"unsupported": 1, "supported": 1},
        checks=[
            CitationCheck(
                citation=Citation(id="c1", claim="good claim here", ref_keys=["1"]),
                ref_key="1", source_pdf=Path("/refs/a.pdf"), resolution="resolved",
                verdict="supported", rationale="ok",
                passages=[EvidenceContext(id="e", paper_id="a", snippet="the source text", page=2)]),
            CitationCheck(
                citation=Citation(id="c2", claim="bad claim here", ref_keys=["2"]),
                ref_key="2", source_pdf=Path("/refs/b.pdf"), resolution="resolved",
                verdict="unsupported", rationale="not in source"),
        ])
    json_path, md_path = write_citation_sidecar(rep, tmp_path)

    data = json.loads(json_path.read_text())
    assert data["manuscript"] == "paper.pdf" and len(data["checks"]) == 2
    assert data["checks"][0]["verdict"] == "supported"

    md = md_path.read_text()
    assert "Coverage:" in md
    assert md.index("bad claim here") < md.index("good claim here")  # unsupported before supported
    assert "the source text" in md


def test_sidecar_shows_override(tmp_path):
    rep = CitationReport(
        manuscript="m.pdf", coverage={},
        checks=[CitationCheck(
            citation=Citation(id="c1", claim="c", ref_keys=["1"]), ref_key="1",
            resolution="resolved", verdict="unsupported", human_override="supported")])
    _, md_path = write_citation_sidecar(rep, tmp_path)
    assert "[supported]" in md_path.read_text()   # override wins over the model verdict
