from probatio.interfaces import (
    Gatherer, Highlighter, LLMClient, ManuscriptParser,
    CitationResolver, CitationVerifier, HighlightRect,
)


def test_check_protocols_are_runtime_checkable():
    # All kept Protocols must be importable and decorated runtime_checkable.
    for proto in (Gatherer, Highlighter, LLMClient, ManuscriptParser,
                  CitationResolver, CitationVerifier):
        assert hasattr(proto, "_is_runtime_protocol")


def test_highlight_rect_shape():
    r = HighlightRect(page=2, x0=1.0, y0=2.0, x1=3.0, y1=4.0)
    assert (r.page, r.x0, r.y1) == (2, 1.0, 4.0)
