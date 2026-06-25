import pytest
from probatio.config import (
    Settings, paperqa_settings, make_verify_client,
)
from probatio.interfaces import LLMClient


def test_settings_defaults(monkeypatch):
    # env override wins; the remaining models fall back to the local-first defaults
    monkeypatch.setenv("PROBATIO_VERIFY_MODEL", "ollama/custom-judge")
    s = Settings()
    assert s.verify_model == "ollama/custom-judge"        # from env
    assert s.summary_model == "ollama/gemma4:31b"         # local-first default
    assert s.embedding_model == "ollama/embeddinggemma"   # local-first default
    assert s.ollama_api_base == "http://localhost:11434"  # local ollama out-of-the-box


def test_paperqa_settings_uses_summary_model():
    s = Settings(summary_model="claude-haiku-4-5-20251001",
                 embedding_model="text-embedding-3-small")
    pq = paperqa_settings(s)
    assert pq.summary_llm == "claude-haiku-4-5-20251001"
    assert pq.embedding == "text-embedding-3-small"


def test_make_verify_client_satisfies_protocol():
    assert isinstance(make_verify_client(Settings()), LLMClient)


def test_skip_summary_wires_into_paperqa():
    assert paperqa_settings(Settings(skip_paperqa_summary=True)).answer.evidence_skip_summary is True
    assert paperqa_settings(Settings()).answer.evidence_skip_summary is False


def _chunk_chars(pq):
    rc = pq.parsing.reader_config
    return rc["chunk_chars"] if isinstance(rc, dict) else rc.chunk_chars


def test_paperqa_doc_details_disabled():
    # we supply our own bibliography; paper-qa must not call an LLM to infer citations
    pq = paperqa_settings(Settings())
    assert pq.parsing.use_doc_details is False
    assert pq.parsing.multimodal is False   # text-only: no image extraction (no pillow needed)
    assert _chunk_chars(pq) == 1500          # small focused chunks by default


def test_chunk_chars_is_configurable():
    assert _chunk_chars(paperqa_settings(Settings(chunk_chars=800))) == 800


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown PROBATIO_LLM_PROVIDER"):
        make_verify_client(Settings(llm_provider="bogus"))


def test_verify_client_honors_ollama_api_base():
    # the judge (LiteLLMClient -> ollama /api/chat) must route to PROBATIO_OLLAMA_API_BASE,
    # not just litellm's OLLAMA_API_BASE env — so one setting controls judge AND embeddings.
    c = make_verify_client(Settings(verify_model="ollama/gemma4:31b",
                                    ollama_api_base="http://remote-ollama:1234"))
    assert c.ollama_api_base == "http://remote-ollama:1234"
