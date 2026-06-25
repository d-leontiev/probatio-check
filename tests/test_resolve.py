import pytest
from probatio.resolve import CitationResolver, _title_sim
from probatio.interfaces import CitationResolver as CitationResolverProto
from probatio.models import Citation, Reference


@pytest.fixture
def refs_dir(tmp_path):
    for n in ["a.pdf", "b.pdf", "c.pdf"]:
        (tmp_path / n).write_bytes(b"%PDF-1.4 stub")
    return tmp_path


def _make(meta_by_name, **kw):
    """Resolver whose per-PDF metadata is injected (no real PDF parsing)."""
    return CitationResolver(extract_meta=lambda p: meta_by_name.get(p.name, (None, None)), **kw)


def _cit(keys):
    return Citation(id="c1", claim="A claim.", ref_keys=keys)


@pytest.mark.asyncio
async def test_doi_exact_match(refs_dir):
    r = _make({"a.pdf": ("10.1/x", "Some Title"), "b.pdf": (None, "Other")})
    assert isinstance(r, CitationResolverProto)
    ref = Reference(key="12", raw="...", doi="https://doi.org/10.1/X")
    checks = await r.resolve([_cit(["12"])], [ref], refs_dir)
    assert len(checks) == 1
    assert checks[0].resolution == "resolved" and checks[0].source_pdf.name == "a.pdf"


@pytest.mark.asyncio
async def test_title_fuzzy_match(refs_dir):
    r = _make({"a.pdf": (None, "Neural networks for image recognition"),
               "b.pdf": (None, "Completely unrelated chemistry paper")})
    ref = Reference(key="3", raw="...", title="Neural Networks for Image Recognition")
    checks = await r.resolve([_cit(["3"])], [ref], refs_dir)
    assert checks[0].resolution == "resolved" and checks[0].source_pdf.name == "a.pdf"


@pytest.mark.asyncio
async def test_no_pdf_when_nothing_matches(refs_dir):
    r = _make({"a.pdf": (None, "Totally different topic about geology")})
    ref = Reference(key="3", raw="...", title="Semi-solid extrusion of pharmaceutical pastes")
    checks = await r.resolve([_cit(["3"])], [ref], refs_dir)
    assert checks[0].resolution == "no_pdf" and checks[0].source_pdf is None


@pytest.mark.asyncio
async def test_ambiguous_when_two_titles_close(refs_dir):
    r = _make({"a.pdf": (None, "3D printing of tablets part one"),
               "b.pdf": (None, "3D printing of tablets part two")})
    ref = Reference(key="3", raw="...", title="3D printing of tablets")
    checks = await r.resolve([_cit(["3"])], [ref], refs_dir)
    assert checks[0].resolution == "ambiguous" and checks[0].source_pdf is None


@pytest.mark.asyncio
async def test_unresolved_marker(refs_dir):
    r = _make({"a.pdf": ("10.1/x", "T")})
    # citation cites key "99" but no such reference exists
    checks = await r.resolve([_cit(["99"])], [Reference(key="1", raw="...")], refs_dir)
    assert checks[0].resolution == "unresolved_marker"


@pytest.mark.asyncio
async def test_grouped_citation_yields_one_check_per_ref(refs_dir):
    r = _make({"a.pdf": ("10.1/a", "A"), "b.pdf": ("10.1/b", "B")})
    refs = [Reference(key="1", raw="...", doi="10.1/a"),
            Reference(key="2", raw="...", doi="10.1/b")]
    checks = await r.resolve([_cit(["1", "2"])], refs, refs_dir)
    assert [c.ref_key for c in checks] == ["1", "2"]
    assert all(c.resolution == "resolved" for c in checks)


@pytest.mark.asyncio
async def test_author_year_fallback(refs_dir):
    r = _make({"a.pdf": (None, "A study of peptides")})
    ref = Reference(key="Smith2020", raw="...", title="A study of peptides",
                    authors=["Smith J"], year=2020)
    # marker key uses "Smith 2020" (space); reference key is "Smith2020" -> fallback matches by author+year
    checks = await r.resolve([_cit(["Smith 2020"])], [ref], refs_dir)
    assert checks[0].resolution == "resolved" and checks[0].reference.key == "Smith2020"


def test_title_sim_basic():
    assert _title_sim("Neural Nets", "neural nets") > 0.95
    assert _title_sim("Cats", "Quantum chromodynamics") < 0.4
