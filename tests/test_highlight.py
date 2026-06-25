# tests/test_highlight.py
from pathlib import Path
from probatio.highlight import PyMuPDFHighlighter
from probatio.interfaces import Highlighter

def test_locate_exact_snippet(tiny_corpus):
    h = PyMuPDFHighlighter()
    assert isinstance(h, Highlighter)
    pdf = Path(tiny_corpus) / "smith2020.pdf"
    rects = h.locate(pdf, "ROCK inhibitors increased phagocytosis in cell culture.", page=2)
    assert len(rects) >= 1
    r = rects[0]
    assert r.page == 2 and r.x1 > r.x0 and r.y1 > r.y0

def test_locate_falls_back_to_first_sentence(tiny_corpus):
    h = PyMuPDFHighlighter()
    pdf = Path(tiny_corpus) / "jones2019.pdf"
    # snippet reflowed/extended beyond what is on the page -> fuzzy fallback
    rects = h.locate(pdf, "ABCA1 is a critical lipid efflux pump in RPE cells. EXTRA TEXT NOT PRESENT.")
    assert len(rects) >= 1  # found via first-sentence fallback

def test_locate_missing_returns_empty(tiny_corpus):
    h = PyMuPDFHighlighter()
    pdf = Path(tiny_corpus) / "smith2020.pdf"
    assert h.locate(pdf, "completely unrelated zzzzz string") == []


def test_locate_highlights_every_found_fragment(tiny_corpus):
    # A multi-sentence snippet: both sentences live in smith2020.pdf (pages 1 and 2).
    # We should highlight BOTH (the whole passage), not just the first sentence.
    h = PyMuPDFHighlighter()
    pdf = Path(tiny_corpus) / "smith2020.pdf"
    snippet = ("Retinal pigment epithelium phagocytosis declines with age. "
               "ROCK inhibitors increased phagocytosis in cell culture.")
    rects = h.locate(pdf, snippet)
    pages = {r.page for r in rects}
    assert pages == {1, 2}  # both fragments found and highlighted, on their own pages


def test_render_page_png_returns_image(tiny_corpus):
    h = PyMuPDFHighlighter()
    pdf = Path(tiny_corpus) / "smith2020.pdf"
    png, page = h.render_page_png(pdf, "ROCK inhibitors increased phagocytosis in cell culture.")
    assert page == 2 and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_page_png_applies_custom_highlight_color(tiny_corpus):
    # The verdict drives the highlight colour (green/yellow/red); two different
    # colours over the same snippet must produce different rendered output.
    h = PyMuPDFHighlighter()
    pdf = Path(tiny_corpus) / "smith2020.pdf"
    snippet = "ROCK inhibitors increased phagocytosis in cell culture."
    green = h.render_page_png(pdf, snippet, color=(0.0, 1.0, 0.0))
    red = h.render_page_png(pdf, snippet, color=(1.0, 0.0, 0.0))
    assert green is not None and red is not None
    assert green[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert green[0] != red[0]   # different highlight colour -> different pixels
