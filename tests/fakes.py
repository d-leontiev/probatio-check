from probatio.models import EvidenceContext, SubQuestion  # noqa: F401


class FakeLLMClient:
    """Scripted LLMClient: returns responses in order, last one repeats."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, *, system: str, user: str, model: str | None = None,
                       think: bool = True) -> str:
        self.calls.append({"system": system, "user": user, "model": model, "think": think})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0] if self._responses else ""
