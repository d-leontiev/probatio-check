import json
import re
from probatio.interfaces import LLMClient
from probatio.models import CitationVerdict, EvidenceContext

_SYSTEM = (
    "You are a strict citation-faithfulness checker for peer review. Given ONE claim from a "
    "manuscript and the exact verbatim passage(s) retrieved from the CITED source, decide how "
    "well the source supports the claim. Judge ONLY against the passage(s) shown — never use "
    "outside knowledge.\n\n"
    "A passage gives \"substantive support\" for a part of the claim only if it bears on the "
    "SAME entity, relation, direction, and scope. Mere topic or vocabulary overlap is NOT "
    "support.\n\n"
    "Grades:\n"
    "- supported: EVERY substantive assertion in the claim is stated by the passage(s) "
    "(or a clear paraphrase).\n"
    "- partially: the claim's CENTRAL assertion is substantively supported, but one or more "
    "secondary parts are not (a compound claim), OR a single assertion is supported only in "
    "part (e.g. the trend but not the magnitude).\n"
    "- overstated: the passage(s) support a NARROWER or WEAKER statement than the claim — "
    "the claim generalizes or strengthens beyond what the source shows "
    "(e.g. one dataset/result -> \"widely used\" / \"in general\").\n"
    "- unsupported: NO substantive part of the claim is supported, OR the passage(s) "
    "CONTRADICT the claim, OR they are off-topic / about a different entity.\n\n"
    "Decision rule (in order):\n"
    "1. If the passage(s) CONTRADICT the claim, grade 'unsupported' "
    "(this overrides everything below).\n"
    "2. Else if every substantive part is supported, grade 'supported'.\n"
    "3. Else if some substantive part is supported: grade 'overstated' when the claim "
    "generalizes or strengthens the supported part beyond the source's scope; otherwise "
    "'partially' (central part supported, secondary parts absent).\n"
    "4. Else (no substantive part supported, or only topic/vocabulary overlap), "
    "grade 'unsupported'.\n\n"
    "Examples:\n"
    "1. Claim: \"Random forests reduce variance by averaging many trees.\" Passage: "
    "\"Averaging over many trees lowers the variance of the ensemble.\" -> "
    "{\"verdict\":\"supported\"} (the whole claim is stated).\n"
    "2. Claim: \"Gradient boosting lowers bias and is the default choice for production ML "
    "systems.\" Passage: \"Gradient boosting sequentially fits learners to reduce bias.\" -> "
    "{\"verdict\":\"partially\"} (bias reduction supported; \"default choice for production\" "
    "is absent).\n"
    "3. Claim: \"This method achieves the best accuracy across all benchmarks.\" Passage: "
    "\"On the MNIST benchmark, the method outperformed the baselines.\" -> "
    "{\"verdict\":\"overstated\"} (one benchmark; the claim generalizes to all).\n"
    "4. Claim: \"Drug D increases cell viability.\" Passage: \"Treatment with drug D "
    "significantly reduced cell viability.\" -> {\"verdict\":\"unsupported\"} "
    "(the passage contradicts the claim).\n"
    "5. Claim: \"Compound A is highly soluble in supercritical CO2.\" Passage: \"Compound B's "
    "solubility was measured at 313 K.\" -> {\"verdict\":\"unsupported\"} "
    "(different compound; only topic overlap).\n\n"
    "Reply JSON only: {\"verdict\":\"supported|partially|overstated|unsupported\","
    "\"rationale\":\"one short sentence\",\"confidence\":0.0-1.0}."
)


def _parse(raw: str) -> tuple[CitationVerdict, str, float]:
    """Crash-safe parse: a malformed verdict is treated as 'unsupported' (needs review)."""
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    try:
        data = json.loads(m.group(1) if m else raw)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return "unsupported", raw[:200], 0.0
    if not isinstance(data, dict):
        return "unsupported", "", 0.0
    rationale = str(data.get("rationale", ""))[:300]
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    v = data.get("verdict")
    verdict: CitationVerdict
    if v == "supported":
        verdict = "supported"
    elif v == "partially":
        verdict = "partially"
    elif v == "overstated":
        verdict = "overstated"
    else:
        verdict = "unsupported"  # default + anything unrecognized -> needs review
    return verdict, rationale, conf


class LLMCitationVerifier:
    """Independent graded judge. Sees only (claim, cited passages) — never the manuscript."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def judge(
        self, claim: str, passages: list[EvidenceContext]
    ) -> tuple[CitationVerdict, str, float]:
        # No passage cleared the relevance floor -> the cited source doesn't cover this claim
        # (wrong paper cited, or claim absent). Distinct, important signal; no LLM call needed.
        if not passages:
            return "not_found", "no relevant passage retrieved from the cited source", 0.0
        snippets = "\n".join(f"- {p.snippet}" for p in passages)
        raw = await self.llm.complete(
            system=_SYSTEM,
            user=f"Claim: {claim}\n\nCited source passage(s):\n{snippets}")
        return _parse(raw)
