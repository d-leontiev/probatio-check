from probatio.models import (
    EvidenceContext,
)


def test_evidence_context_roundtrip():
    e = EvidenceContext(id="e1", paper_id="smith2020", snippet="Cells divide.",
                        page=3, score=0.8, rcs_summary="cells divide in mitosis")
    assert e.page == 3 and e.snippet == "Cells divide."
