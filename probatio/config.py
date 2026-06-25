from pydantic_settings import BaseSettings, SettingsConfigDict
from paperqa import Settings as PQASettings
from probatio.interfaces import LLMClient


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROBATIO_", env_file=".env", extra="ignore")
    # "litellm" — routes ollama/* models to a local ollama (default) or any litellm backend.
    llm_provider: str = "litellm"
    # Defaults are LOCAL-FIRST: check mode runs fully on-machine out-of-the-box against a local
    # ollama (pull gemma4:31b + embeddinggemma), so an embargoed manuscript never leaves the box.
    # Override via .env / PROBATIO_* only for a remote ollama host, or for non-confidential cloud
    # use (assert_local_only fail-closes if check mode is pointed at any cloud model/endpoint).
    summary_model: str = "ollama/gemma4:31b"      # RCS model; unused in check mode (RCS is skipped)
    verify_model: str = "ollama/gemma4:31b"       # the citation judge
    embedding_model: str = "ollama/embeddinggemma"
    ollama_api_base: str = "http://localhost:11434"   # local ollama; routes judge + embeddings (+ RCS)
    # Per-call timeout (s) for the ollama summary/RCS model. litellm's Ollama default is
    # 60s — too tight for a cold 31B load + a long-passage summary. Only applied to
    # ollama/* summary models (cloud models keep their own defaults).
    summary_timeout: int = 300
    evidence_k: int = 10
    # check mode: how many top retrieved passages the citation judge sees per claim. Default 10
    # (= evidence_k, i.e. every retrieved passage) so a supporting passage isn't missed when it
    # ranks just below the top — empirically the single biggest lever against false "unsupported".
    check_passages: int = 10
    chunk_chars: int = 1500   # small, focused chunks -> on-topic snippets + precise highlights
    # When True, paper-qa skips its contextual-summary LLM pass -> retrieval-only,
    # zero LLM calls (pair with a local embedding model for a $0-metered run).
    skip_paperqa_summary: bool = False
    unpaywall_email: str = "leon.dmytro@gmail.com"   # contact for Unpaywall/OpenAlex polite pool


def paperqa_settings(s: Settings) -> PQASettings:
    """PaperQA2 is used for EVIDENCE only; the summary_llm does the RCS pass."""
    pq = PQASettings()
    pq.llm = s.summary_model
    pq.summary_llm = s.summary_model
    pq.embedding = s.embedding_model
    # Raise the per-call timeout for an ollama summary/RCS model (litellm's Ollama default
    # of 60s is too tight for a cold 31B load + a long-passage summary) and pin its
    # api_base so retrieval (embedding) and RCS can live on different ollama hosts. Format
    # is the litellm router model_list dict, which lmi/LiteLLMModel passes straight through.
    if s.summary_timeout and s.summary_model.startswith("ollama/"):
        litellm_params = {"model": s.summary_model, "timeout": s.summary_timeout}
        if s.ollama_api_base:
            litellm_params["api_base"] = s.ollama_api_base
        cfg = {"model_list": [{"model_name": s.summary_model, "litellm_params": litellm_params}]}
        pq.summary_llm_config = cfg
        pq.llm_config = cfg
    # Pin the EMBEDDING model's api_base too. assert_local_only permits a remote/Tailscale
    # ollama host, but without this paper-qa's embedding call falls back to litellm's default
    # ollama host (localhost:11434) and silently ignores ollama_api_base — so retrieval would
    # hit the wrong (or no) ollama. embedding_config -> LiteLLMEmbeddingModel.config, whose
    # "kwargs" are forwarded to litellm.aembedding (where api_base routes ollama/* models).
    if s.ollama_api_base and s.embedding_model.startswith("ollama/"):
        pq.embedding_config = {"kwargs": {"api_base": s.ollama_api_base}}
    pq.answer.evidence_k = s.evidence_k
    pq.answer.answer_max_sources = s.evidence_k
    pq.answer.evidence_skip_summary = s.skip_paperqa_summary
    # We supply our own bibliography (from the ingestor), so don't let paper-qa call
    # an LLM / metadata service to infer each PDF's citation during aadd. Combined with
    # skip_summary + local embeddings this makes paper-qa fully LLM-free (key-free).
    pq.parsing.use_doc_details = False
    # Text-only pipeline: don't extract images from PDFs (avoids the pillow dependency
    # and needless work — we ground on text snippets, not figures).
    pq.parsing.multimodal = False
    # Smaller chunks so each evidence snippet is a single focused passage rather than a
    # 5000-char multi-topic blob (keeps snippets on-topic and the PDF highlight precise).
    rc = pq.parsing.reader_config
    overlap = min(150, s.chunk_chars // 8)
    if isinstance(rc, dict):
        rc["chunk_chars"], rc["overlap"] = s.chunk_chars, overlap
    else:
        rc.chunk_chars, rc.overlap = s.chunk_chars, overlap
    return pq


def _client_for(s: Settings, model: str) -> LLMClient:
    if s.llm_provider == "litellm":
        from probatio.llm import LiteLLMClient
        return LiteLLMClient(default_model=model, ollama_api_base=s.ollama_api_base)
    raise ValueError(f"unknown PROBATIO_LLM_PROVIDER: {s.llm_provider!r} (use 'litellm')")


def make_verify_client(s: Settings) -> LLMClient:
    return _client_for(s, s.verify_model)


# --- "check" mode: fail-closed confidentiality guard --------------------------
class ConfidentialityError(RuntimeError):
    """Raised when check mode would send the manuscript or its sources off the machine."""


def _is_local_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("ollama/") or m.startswith("st-")


def _is_local_host(api_base: str) -> bool:
    if not api_base:
        return False
    host = api_base.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    if host in ("localhost", "0.0.0.0") or host.endswith(".local"):
        return True
    if host.startswith(("127.", "10.", "192.168.", "100.")):  # loopback + RFC1918 + Tailscale CGNAT
        return True
    if host.startswith("172.") and host.split(".")[1].isdigit() and 16 <= int(host.split(".")[1]) <= 31:
        return True
    return "." not in host  # a single-label hostname is a LAN/Tailscale name, not a public FQDN


def assert_local_only(s: Settings) -> None:
    """Refuse to proceed unless every model + endpoint check mode uses is local. Fail-closed:
    the manuscript under review is embargoed, so nothing may leave the machine."""
    problems: list[str] = []
    if s.llm_provider != "litellm":
        problems.append(
            f"llm_provider={s.llm_provider!r} routes to the cloud — use 'litellm' (-> ollama)")
    for label, model in (("verify_model", s.verify_model), ("embedding_model", s.embedding_model)):
        if not _is_local_model(model):
            problems.append(f"{label}={model!r} is not local (use an ollama/* or st-* model)")
    if not _is_local_host(s.ollama_api_base):
        problems.append(
            f"ollama_api_base={s.ollama_api_base!r} is empty or not a local/Tailscale host")
    if problems:
        raise ConfidentialityError(
            "check mode is confidential and refuses to run with non-local components:\n  - "
            + "\n  - ".join(problems))
