from probatio.llm import _ollama_chat_payload


def test_ollama_payload_think_flag():
    # Reasoning toggled per call: OFF for the huge reference-list parse (thinking makes it
    # hang), ON for bounded judgment calls (scope-tag, judge) where it improves quality.
    on = _ollama_chat_payload("gemma4:31b", "SYS", "USR", 1234, think=True)
    off = _ollama_chat_payload("gemma4:31b", "SYS", "USR", 1234, think=False)
    assert on["think"] is True and off["think"] is False
    assert on["stream"] is False
    assert on["model"] == "gemma4:31b"
    assert on["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"}]
    assert on["options"]["num_predict"] == 1234
    assert on["options"]["temperature"] == 0.0


def test_ollama_payload_temp_zero():
    """Temperature must be 0.0 for deterministic judge output."""
    payload = _ollama_chat_payload("gemma4:31b", "SYS", "USR", 512, think=True)
    assert payload["options"]["temperature"] == 0.0
