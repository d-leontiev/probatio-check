import pytest
from pathlib import Path
from probatio.interfaces import LLMClient
from tests.fakes import FakeLLMClient


@pytest.mark.asyncio
async def test_fake_llm_returns_scripted_response():
    client = FakeLLMClient(responses=["hello"])
    assert isinstance(client, LLMClient)
    out = await client.complete(system="s", user="u")
    assert out == "hello"


def test_tiny_corpus_fixture_has_pdfs(tiny_corpus):
    pdfs = list(Path(tiny_corpus).glob("*.pdf"))
    assert len(pdfs) >= 2
