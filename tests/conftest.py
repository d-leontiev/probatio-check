import os
from pathlib import Path
import pytest
import fitz  # PyMuPDF


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    """Tests assert DEFAULT Settings behavior — never read a developer's local .env
    or stray PROBATIO_*/secret env vars (a real .env exists once the app is configured)."""
    from probatio.config import Settings
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for k in list(os.environ):
        if k.startswith("PROBATIO_"):
            monkeypatch.delenv(k, raising=False)


def _make_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()


@pytest.fixture
def tiny_corpus(tmp_path) -> Path:
    """A hermetic 2-PDF corpus with known sentences for evidence/highlight tests."""
    d = tmp_path / "corpus"
    d.mkdir()
    _make_pdf(d / "smith2020.pdf",
              ["Retinal pigment epithelium phagocytosis declines with age.",
               "ROCK inhibitors increased phagocytosis in cell culture."])
    _make_pdf(d / "jones2019.pdf",
              ["ABCA1 is a critical lipid efflux pump in RPE cells.",
               "Lipid handling is implicated in macular degeneration."])
    return d
