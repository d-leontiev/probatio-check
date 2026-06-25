import os
import httpx
import litellm


def _ollama_chat_payload(model: str, system: str, user: str, num_predict: int,
                         think: bool) -> dict:
    """Body for ollama's /api/chat. ``think`` is toggled per call: OFF for the huge
    reference-list parse — gemma-class reasoning models otherwise spend unbounded tokens
    'thinking' before the structured output, which hangs it (and litellm forwards neither the
    think flag nor an output cap to ollama) — and ON for bounded judgment calls (scope-tag,
    judge) where reasoning improves quality without risking a hang."""
    return {
        "model": model,
        "stream": False,
        "think": think,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }


class LiteLLMClient:
    """Real LLMClient. ollama/* models go straight to the ollama API (litellm forwards neither
    the think flag nor an output cap); everything else routes via LiteLLM."""

    def __init__(self, default_model: str, *, ollama_api_base: str = "",
                 num_predict: int = 16384, timeout: float = 600.0):
        self.default_model = default_model
        self.ollama_api_base = ollama_api_base
        self.num_predict = num_predict
        self.timeout = timeout

    async def complete(self, *, system: str, user: str, model: str | None = None,
                       think: bool = True) -> str:
        m = model or self.default_model
        if m.startswith("ollama/"):
            return await self._ollama_complete(m.split("/", 1)[1], system=system, user=user,
                                               think=think)
        resp = await litellm.acompletion(
            model=m,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    async def _ollama_complete(self, model: str, *, system: str, user: str, think: bool) -> str:
        # Prefer the app setting (PROBATIO_OLLAMA_API_BASE) so ONE knob routes the judge AND
        # embeddings; fall back to litellm's OLLAMA_API_BASE env, then the local default.
        base = (self.ollama_api_base or os.environ.get("OLLAMA_API_BASE")
                or "http://localhost:11434").rstrip("/")
        payload = _ollama_chat_payload(model, system, user, self.num_predict, think)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{base}/api/chat", json=payload)
            resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content", "") or ""
