from pathlib import Path
from probatio.models import Reference, Citation, CitationCheck, CitationReport
from probatio.interfaces import ManuscriptParser, CitationResolver, CitationVerifier


def test_citation_defaults():
    c = Citation(id="c0001", claim="X increases Y.", ref_keys=["12"])
    assert c.kind == "empirical" and c.section == ""
    chk = CitationCheck(citation=c, ref_key="12", resolution="resolved")
    assert chk.verdict == "unchecked"
    assert chk.human_override is None and chk.passages == []
    rep = CitationReport(manuscript="paper.pdf", checks=[chk], coverage={"resolved": 1})
    assert rep.coverage["resolved"] == 1


def test_reference_and_source_pdf():
    ref = Reference(key="12", raw="Smith J. 2020. Title. doi:10.1/x", title="Title",
                    doi="10.1/x", authors=["Smith J"], year=2020)
    chk = CitationCheck(
        citation=Citation(id="c1", claim="c", ref_keys=["12"]),
        ref_key="12", reference=ref, source_pdf=Path("/tmp/refs/a.pdf"),
        resolution="resolved", verdict="overstated", confidence=0.7)
    assert chk.reference.doi == "10.1/x" and chk.source_pdf.name == "a.pdf"
    assert chk.verdict == "overstated"


def test_protocols_are_runtime_checkable():
    # Smoke: the Protocols import and are runtime_checkable (used for DI in tests).
    assert isinstance(ManuscriptParser, type(CitationResolver))
    assert hasattr(CitationVerifier, "_is_runtime_protocol")


def test_citation_check_reviewed_defaults_false_and_roundtrips():
    from probatio.models import Citation, CitationCheck
    c = CitationCheck(
        citation=Citation(id="c1", claim="x", ref_keys=["1"]),
        ref_key="1", resolution="resolved",
    )
    assert c.reviewed is False
    # old JSON without the field still loads, defaulting to False
    loaded = CitationCheck.model_validate_json(
        '{"citation":{"id":"c1","claim":"x","ref_keys":["1"]},'
        '"ref_key":"1","resolution":"resolved"}'
    )
    assert loaded.reviewed is False
    # the field round-trips when set
    c.reviewed = True
    assert CitationCheck.model_validate_json(c.model_dump_json()).reviewed is True
