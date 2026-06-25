import pytest
from probatio.check_retrieval import RefRetriever
from probatio.config import Settings
from probatio.interfaces import IndexHandle
from probatio.models import EvidenceContext


class _FakeGatherer:
    def __init__(self):
        self.seen: list[str] = []

    async def gather(self, index, subq):
        self.seen.append(subq.text)
        return [
            EvidenceContext(id="e1", paper_id="a", snippet="verbatim source text", page=2, score=0.9),
            EvidenceContext(id="e2", paper_id="a", snippet="second passage", page=3, score=0.4),
        ]


@pytest.mark.asyncio
async def test_caches_index_per_pdf_and_caps_k(tmp_path):
    builds: list = []

    async def fake_build(p):
        builds.append(p)
        return IndexHandle(docs=object(), settings=object(), pdf_dir=p.parent)

    g = _FakeGatherer()
    rr = RefRetriever(Settings(), build_index=fake_build, gatherer=g)
    pdf = tmp_path / "a.pdf"

    p1 = await rr.passages_for(pdf, "claim one", k=1)
    p2 = await rr.passages_for(pdf, "claim two", k=2)

    assert len(builds) == 1                       # embedded once, then cached
    assert g.seen == ["claim one", "claim two"]   # but retrieval runs per claim
    assert len(p1) == 1 and p1[0].snippet == "verbatim source text" and p1[0].page == 2
    assert len(p2) == 2


@pytest.mark.asyncio
async def test_skip_summary_forced_on(tmp_path):
    rr = RefRetriever(Settings(), build_index=None, gatherer=_FakeGatherer())
    assert rr._pq.answer.evidence_skip_summary is True   # verbatim, no RCS
