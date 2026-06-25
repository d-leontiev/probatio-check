import pytest
from pathlib import Path
from types import SimpleNamespace
from probatio.gather import PaperQAGatherer
from probatio.interfaces import Gatherer, IndexHandle
from probatio.models import SubQuestion

def _fake_session():
    # Mimic paper-qa session.contexts: each has .text.text, .text.name,
    # .context (RCS summary), .score, and page info in the chunk name.
    ctx = SimpleNamespace(
        context="ROCK inhibitors raise phagocytosis.",
        score=8,
        text=SimpleNamespace(
            text="ROCK inhibitors increased phagocytosis in cell culture.",
            name="smith2020 pages 2-2", doc=SimpleNamespace(dockey="smith2020")),
    )
    return SimpleNamespace(contexts=[ctx])

@pytest.mark.asyncio
async def test_gather_maps_contexts_to_evidence(monkeypatch):
    handle = IndexHandle(docs=SimpleNamespace(), settings=SimpleNamespace(), pdf_dir=Path("."))
    g = PaperQAGatherer()
    assert isinstance(g, Gatherer)
    async def fake_aget_evidence(query, settings=None): return _fake_session()
    handle.docs.aget_evidence = fake_aget_evidence
    out = await g.gather(handle, SubQuestion(id="q1", text="Do ROCK inhibitors help?", order=0))
    assert len(out) == 1
    e = out[0]
    assert e.paper_id == "smith2020"
    assert e.page == 2
    assert e.snippet.startswith("ROCK inhibitors increased")
    assert e.rcs_summary == "ROCK inhibitors raise phagocytosis."
    assert e.id == "smith2020::p2::q1::0"  # includes subq.id for cross-question uniqueness
