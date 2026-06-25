import json
import pytest
from probatio.verify_citations import LLMCitationVerifier, _SYSTEM
from probatio.interfaces import CitationVerifier
from probatio.models import EvidenceContext
from tests.fakes import FakeLLMClient

_EV = [EvidenceContext(id="e1", paper_id="p", snippet="may slightly increase viscosity")]


@pytest.mark.asyncio
async def test_graded_verdict_and_rationale():
    fake = FakeLLMClient([json.dumps(
        {"verdict": "overstated", "rationale": "source hedges with 'may'", "confidence": 0.8})])
    v = LLMCitationVerifier(fake)
    assert isinstance(v, CitationVerifier)
    verdict, why, conf = await v.judge("X greatly increases viscosity.", _EV)
    assert verdict == "overstated" and "hedge" in why.lower() and conf == 0.8


@pytest.mark.asyncio
async def test_not_found_short_circuits_without_llm_call():
    fake = FakeLLMClient([])
    verdict, why, conf = await LLMCitationVerifier(fake).judge("any claim", [])
    assert verdict == "not_found" and conf == 0.0
    assert fake.calls == []  # never call the model when there's nothing to judge


@pytest.mark.asyncio
async def test_malformed_json_is_unsupported_not_crash():
    verdict, why, conf = await LLMCitationVerifier(
        FakeLLMClient(["not json at all"])).judge("c", _EV)
    assert verdict == "unsupported" and conf == 0.0


@pytest.mark.asyncio
async def test_unknown_verdict_defaults_unsupported_and_conf_clamped():
    fake = FakeLLMClient([json.dumps({"verdict": "great", "confidence": 5})])
    verdict, why, conf = await LLMCitationVerifier(fake).judge("c", _EV)
    assert verdict == "unsupported" and conf == 1.0  # clamped to [0,1]


def test_system_prompt_contains_calibration_tokens():
    """_SYSTEM must contain the key tokens introduced by the judge-calibration edit."""
    for token in ("partially", "overstated", "compound", "substantive", "CONTRADICT", "supported"):
        assert token in _SYSTEM, f"_SYSTEM missing expected token: {token!r}"


@pytest.mark.asyncio
async def test_judge_sees_only_claim_and_passage():
    fake = FakeLLMClient([json.dumps({"verdict": "supported", "confidence": 0.9})])
    await LLMCitationVerifier(fake).judge("CLAIM TEXT", _EV)
    user = fake.calls[0]["user"]
    assert "CLAIM TEXT" in user and "may slightly increase viscosity" in user
