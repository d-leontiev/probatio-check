import pytest
from probatio.check import check_pipeline
from probatio.models import Citation, CitationCheck, EvidenceContext


class StubParser:
    async def parse(self, pdf):
        return [], []          # resolver stub supplies the checks directly


class StubResolver:
    def __init__(self, checks):
        self._checks = checks

    async def resolve(self, citations, references, refs_dir):
        return self._checks


class StubRetriever:
    def __init__(self, by_name):
        self._by = by_name

    async def passages_for(self, pdf_path, claim, k=3):
        if pdf_path.name == "corrupt.pdf":
            raise ValueError("unreadable PDF")
        return self._by.get(pdf_path.name, [])


class StubVerifier:
    async def judge(self, claim, passages):
        return "supported", "matches the source", 0.9


def _cit(cid, kind="empirical"):
    return Citation(id=cid, claim=f"claim {cid}", ref_keys=["1"], kind=kind)


@pytest.mark.asyncio
async def test_pipeline_buckets_every_citation(tmp_path):
    checks = [
        CitationCheck(citation=_cit("c1"), ref_key="1", source_pdf=tmp_path / "a.pdf", resolution="resolved"),
        CitationCheck(citation=_cit("c2"), ref_key="1", source_pdf=tmp_path / "b.pdf", resolution="resolved"),
        CitationCheck(citation=_cit("c3", "non_checkable"), ref_key="1", source_pdf=tmp_path / "a.pdf", resolution="resolved"),
        CitationCheck(citation=_cit("c4"), ref_key="9", resolution="no_pdf"),
        CitationCheck(citation=_cit("c5"), ref_key="1", source_pdf=tmp_path / "corrupt.pdf", resolution="resolved"),
    ]
    retr = StubRetriever({"a.pdf": [EvidenceContext(id="e", paper_id="a", snippet="src", page=1)]})  # b.pdf -> []
    report = await check_pipeline(
        manuscript=tmp_path / "m.pdf", refs_dir=tmp_path,
        parser=StubParser(), resolver=StubResolver(checks),
        retriever=retr, verifier=StubVerifier())

    by_id = {c.citation.id: c for c in report.checks}
    assert by_id["c1"].verdict == "supported" and by_id["c1"].passages          # judged
    assert by_id["c2"].verdict == "not_found"                                    # retrieval empty
    assert by_id["c3"].verdict == "not_a_claim" and by_id["c3"].passages == []   # non-checkable, not judged
    assert by_id["c4"].verdict == "unchecked"                                    # no_pdf, not judged
    assert by_id["c5"].resolution == "unreadable_source"                         # retriever raised
    assert report.coverage == {
        "supported": 1, "not_found": 1, "not_a_claim": 1, "no_pdf": 1, "unreadable_source": 1}
