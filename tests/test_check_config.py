import pytest
from probatio.config import Settings, assert_local_only, ConfidentialityError, paperqa_settings


def _local(**kw):
    base = dict(llm_provider="litellm", verify_model="ollama/gemma4:31b",
                embedding_model="ollama/embeddinggemma",
                ollama_api_base="http://localhost:11435")
    base.update(kw)
    return Settings(**base)


def test_check_passages_default():
    # The judge sees the top-N retrieved passages; default is evidence_k (every retrieved passage)
    # so a supporting passage isn't dropped when it ranks just below the top of the retrieval.
    assert Settings().check_passages == 10


def test_local_only_passes():
    assert assert_local_only(_local()) is None


def test_cloud_verify_model_blocked():
    with pytest.raises(ConfidentialityError):
        assert_local_only(_local(verify_model="claude-haiku-4-5-20251001"))


def test_cloud_embedding_blocked():
    with pytest.raises(ConfidentialityError):
        assert_local_only(_local(embedding_model="text-embedding-3-small"))


def test_claude_agent_provider_blocked():
    with pytest.raises(ConfidentialityError):
        assert_local_only(_local(llm_provider="claude-agent"))


def test_empty_api_base_blocked():
    with pytest.raises(ConfidentialityError):
        assert_local_only(_local(ollama_api_base=""))


def test_public_api_base_blocked():
    with pytest.raises(ConfidentialityError):
        assert_local_only(_local(ollama_api_base="https://api.some-cloud.com:443"))


def test_tailscale_and_hostname_allowed():
    assert assert_local_only(_local(ollama_api_base="http://spark-a2c2:11434")) is None
    assert assert_local_only(_local(ollama_api_base="http://100.125.212.122:11434")) is None
    assert assert_local_only(_local(ollama_api_base="http://dmleon-system:11434")) is None


def test_embedding_api_base_plumbed_for_ollama():
    # A remote/Tailscale ollama (allowed by assert_local_only) must route EMBEDDINGS there too,
    # not just the judge — else paper-qa silently falls back to localhost:11434.
    pq = paperqa_settings(_local(embedding_model="ollama/embeddinggemma",
                                 ollama_api_base="http://100.125.212.122:11434"))
    assert pq.embedding_config == {"kwargs": {"api_base": "http://100.125.212.122:11434"}}


def test_embedding_api_base_not_set_for_local_st():
    # sentence-transformers embeddings run in-process; there is no api_base to inject.
    pq = paperqa_settings(_local(embedding_model="st-all-MiniLM-L6-v2",
                                 ollama_api_base="http://localhost:11435"))
    assert pq.embedding_config is None


def test_default_settings_pass_local_only():
    # local-first: with no .env and no PROBATIO_* overrides, check mode is allowed to run
    # out-of-the-box (the confidential local path is the default, not a cloud config).
    assert assert_local_only(Settings()) is None
