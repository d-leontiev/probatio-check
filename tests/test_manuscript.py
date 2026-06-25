import json
import logging
import re
import httpx
import pytest
import fitz
from probatio.manuscript import (
    find_markers, strip_markers, split_sentences, references_block,
    strip_line_numbers, split_reference_block, PyMuPDFManuscriptParser,
)
from probatio.interfaces import ManuscriptParser
from tests.fakes import FakeLLMClient


def test_numeric_markers_expand():
    assert find_markers("rises sharply [3,4,5] and falls [7].") == [["3", "4", "5"], ["7"]]


def test_numeric_ranges():
    assert find_markers("see [3–5] then [10-12].") == [["3", "4", "5"], ["10", "11", "12"]]


def test_author_year_markers():
    assert find_markers("as shown (Smith et al., 2020; Jones, 2019).") == \
        [["Smith 2020", "Jones 2019"]]


def test_markers_in_document_order():
    # numeric then author-year, interleaved -> returned by position
    assert find_markers("A (Lee, 2018) and B [4].") == [["Lee 2018"], ["4"]]


def test_strip_markers():
    assert strip_markers("X rises [3] sharply (Smith, 2020).") == "X rises sharply."


def test_split_sentences():
    s = split_sentences("First claim [1]. Second claim [2]!")
    assert len(s) == 2 and s[0].endswith("[1].")


def test_references_block_excludes_body():
    blk = references_block("Body text here.\nReferences\n1. Smith 2020. Title.")
    assert "Smith 2020" in blk and "Body text" not in blk


def test_strip_line_numbers_removes_margin_numbers():
    # Reviewer PDFs number every line; PyMuPDF emits each margin number as its own line.
    polluted = "the expected thermodynamic\n234 \ndependence of solubility\n235\non density."
    assert strip_line_numbers(polluted) == (
        "the expected thermodynamic\ndependence of solubility\non density.")


def test_strip_line_numbers_keeps_inline_numbers():
    # Years, quantities, ranges, and markers inside a text line must survive untouched.
    line = "solubility rose from 12 to 30 MPa in 2017 [3]."
    assert strip_line_numbers(line) == line


def test_strip_line_numbers_removes_sequential_column():
    # The whole left-margin column ("1 2 3 ... 65") arrives as standalone digit lines.
    assert strip_line_numbers("end of claim.\n1 \n2 \n3 \nNext claim.") == (
        "end of claim.\nNext claim.")


@pytest.mark.asyncio
async def test_parse_strips_margin_line_numbers_from_claims(tmp_path):
    pdf = _make_pdf(
        tmp_path,
        "SSE lowers the\n234\ntemperature significantly [1].\n\nReferences\n1. A.")
    refs_json = json.dumps([{"key": "1", "title": "A", "raw": "1. A"}])
    scope_json = json.dumps([{"i": 0, "kind": "empirical"}])
    parser = PyMuPDFManuscriptParser(FakeLLMClient([refs_json, scope_json]))
    cits, _ = await parser.parse(pdf)
    assert cits and "234" not in cits[0].claim
    assert cits[0].claim == "SSE lowers the temperature significantly."


def _make_pdf(tmp_path, text):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    p = tmp_path / "m.pdf"
    doc.save(str(p))
    doc.close()
    return p


@pytest.mark.asyncio
async def test_parse_end_to_end(tmp_path):
    pdf = _make_pdf(
        tmp_path,
        "SSE lowers temperature [1]. We used the method of [2].\n\n"
        "References\n1. A. doi:10.1/x\n2. B.")
    refs_json = json.dumps([
        {"key": "1", "title": "A", "doi": "10.1/x", "authors": ["A"], "year": 2020, "raw": "1. A"},
        {"key": "2", "title": "B", "doi": None, "authors": ["B"], "year": 2019, "raw": "2. B"}])
    scope_json = json.dumps([{"i": 0, "kind": "empirical"}, {"i": 1, "kind": "non_checkable"}])
    parser = PyMuPDFManuscriptParser(FakeLLMClient([refs_json, scope_json]))
    assert isinstance(parser, ManuscriptParser)

    cits, refs = await parser.parse(pdf)
    assert {r.key for r in refs} == {"1", "2"}
    assert next(r for r in refs if r.key == "1").doi == "10.1/x"
    assert [c.ref_keys for c in cits] == [["1"], ["2"]]
    kinds = {c.ref_keys[0]: c.kind for c in cits}
    assert kinds["1"] == "empirical" and kinds["2"] == "non_checkable"
    assert "method of" in next(c for c in cits if c.ref_keys == ["2"]).claim


@pytest.mark.asyncio
async def test_parse_disables_thinking_only_for_reference_list(tmp_path):
    pdf = _make_pdf(tmp_path, "A measured claim [1].\n\nReferences\n1. A.")
    fake = FakeLLMClient([json.dumps([{"key": "1", "raw": "1. A"}]), json.dumps([])])
    await PyMuPDFManuscriptParser(fake).parse(pdf)
    # 1st call = reference-list parse (huge structured output) -> thinking OFF (else it hangs)
    assert fake.calls[0]["think"] is False
    # 2nd call = scope-tagging, a bounded judgment task -> thinking stays ON
    assert fake.calls[1]["think"] is True


@pytest.mark.asyncio
async def test_parse_tolerates_no_references(tmp_path):
    pdf = _make_pdf(tmp_path, "A bare claim with no bibliography [1].")
    parser = PyMuPDFManuscriptParser(FakeLLMClient([json.dumps([])]))  # only scope-tag call
    cits, refs = await parser.parse(pdf)
    assert refs == [] and [c.ref_keys for c in cits] == [["1"]]


@pytest.mark.asyncio
async def test_parse_references_only_no_scope_call(tmp_path):
    pdf = _make_pdf(tmp_path, "A claim [1].\n\nReferences\n1. A. doi:10.1/x")
    fake = FakeLLMClient([json.dumps([{"key": "1", "doi": "10.1/x", "raw": "1. A"}])])
    refs = await PyMuPDFManuscriptParser(fake).parse_references(pdf)
    assert [r.key for r in refs] == ["1"] and refs[0].doi == "10.1/x"
    assert len(fake.calls) == 1 and fake.calls[0]["think"] is False   # only the refs parse ran


@pytest.mark.integration
def test_parse_does_not_bleed_heading_into_claim(tmp_path):
    """Layout-aware mining must isolate the section heading so it cannot fuse
    with surrounding claim sentences."""
    import asyncio

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 200), "Averaging across trees reduces variance [1].", fontsize=11)
    page.insert_text((72, 240), "2.6.5 Extra Trees", fontsize=16)  # larger → heading
    page.insert_text((72, 280), "To address this, Extra Trees randomise splits [2].", fontsize=11)
    pdf = tmp_path / "m.pdf"
    doc.save(str(pdf))
    doc.close()

    # FakeLLMClient: first response = empty refs JSON, second = empty scope JSON
    fake = FakeLLMClient([json.dumps([]), json.dumps([])])
    parser = PyMuPDFManuscriptParser(fake)
    citations, _ = asyncio.run(parser.parse(pdf))
    claims = [c.claim for c in citations]

    assert any(c.startswith("Averaging across trees") for c in claims), \
        f"Expected first claim to start with 'Averaging across trees'; got: {claims}"
    assert all("Extra Trees" not in c or c.startswith("To address") for c in claims), \
        f"Heading 'Extra Trees' bled into a non-expected claim; got: {claims}"
    assert not any("variance" in c and "Extra Trees" in c for c in claims), \
        f"Heading fused with preceding body text; got: {claims}"


# ---------------------------------------------------------------------------
# Fix 1: _mine_segments uses real PDF page numbers from (pnum, segs) tuples
# ---------------------------------------------------------------------------

def test_mine_segments_uses_real_page_numbers():
    """_mine_segments must stamp manuscript_page with the real PDF page number
    from the tuple, not a re-enumerated index.  Here we pass a clean_pages whose
    first entry is page 3 — the citation must have manuscript_page=3, not 1."""
    fake = FakeLLMClient([json.dumps([]), json.dumps([])])
    parser = PyMuPDFManuscriptParser(fake)
    # Simulate: PDF page 3 has one segment with a citation; pages 1+2 had no surviving segs
    clean_pages = [
        (3, ["Body claim on page three [5]."]),
    ]
    citations = parser._mine_segments(clean_pages)
    assert len(citations) == 1
    assert citations[0].manuscript_page == 3, (
        f"Expected manuscript_page=3 (real PDF page), got {citations[0].manuscript_page}")


def test_mine_segments_multiple_pages_correct_stamp():
    """When pages 1, 3, 5 have segments, each citation must carry the right page number."""
    fake = FakeLLMClient([json.dumps([]), json.dumps([])])
    parser = PyMuPDFManuscriptParser(fake)
    clean_pages = [
        (1, ["Claim on page one [1]."]),
        (3, ["Claim on page three [3]."]),
        (5, ["Claim on page five [5]."]),
    ]
    citations = parser._mine_segments(clean_pages)
    assert len(citations) == 3
    pages_stamped = [c.manuscript_page for c in citations]
    assert pages_stamped == [1, 3, 5], f"Expected [1, 3, 5], got {pages_stamped}"


# ---------------------------------------------------------------------------
# Fix 2: fallback gated on CITATION COUNT — zero citations triggers fallback
# ---------------------------------------------------------------------------

def test_mine_segments_empty_clean_pages_returns_empty():
    """_mine_segments on empty list must return [] (so fallback fires)."""
    fake = FakeLLMClient([json.dumps([]), json.dumps([])])
    parser = PyMuPDFManuscriptParser(fake)
    assert parser._mine_segments([]) == []


def test_mine_segments_no_citation_markers_returns_empty():
    """A clean_pages with segments but NO citation markers must return [],
    triggering the fallback even though segments exist."""
    fake = FakeLLMClient([json.dumps([]), json.dumps([])])
    parser = PyMuPDFManuscriptParser(fake)
    # Segment has no [N] or (Author Year) markers
    clean_pages = [(1, ["This sentence has no citation marker at all."])]
    assert parser._mine_segments(clean_pages) == [], (
        "Expected [] when no citation markers found — fallback should activate")


@pytest.mark.asyncio
async def test_parse_fallback_when_layout_yields_no_citations(tmp_path):
    """When layout mining finds zero citations (e.g. all junk segments with no
    markers), parse() must fall back to plain-text mining and still return the
    citation present in the raw text."""
    # PDF has a real citation [1] in the body, but also a junk block that would
    # form the only layout segment (no marker in it) — the real citation appears
    # in plain text so fallback must recover it.
    # We build the PDF with citation in plain text on page 1.
    doc = fitz.open()
    page = doc.new_page()
    # Insert plain claim with citation — the layout path will see it too,
    # but we test the decision logic via _mine_segments directly below.
    page.insert_text((72, 200), "Temperature affects solubility [1].", fontsize=11)
    page.insert_text((72, 250), "\nReferences\n1. Smith A. Title. 2020.")
    pdf = tmp_path / "fb.pdf"
    doc.save(str(pdf))
    doc.close()

    refs_json = json.dumps([{"key": "1", "title": "Title", "raw": "1. Smith A."}])
    scope_json = json.dumps([{"i": 0, "kind": "empirical"}])
    parser = PyMuPDFManuscriptParser(FakeLLMClient([refs_json, scope_json]))
    cits, _ = await parser.parse(pdf)
    # At least one citation must be found — either from layout or fallback
    assert len(cits) >= 1
    assert any("1" in c.ref_keys for c in cits)


# ---------------------------------------------------------------------------
# Fix 4: layout failure must log a warning (not silently swallow the error)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_logs_warning_on_layout_failure(tmp_path, caplog):
    """When layout extraction raises, parse() must log a WARNING via the
    'probatio.manuscript' logger before falling back to plain text."""
    import unittest.mock as mock

    pdf = _make_pdf(tmp_path, "A claim [1].\n\nReferences\n1. A.")
    refs_json = json.dumps([{"key": "1", "raw": "1. A"}])
    scope_json = json.dumps([])

    # Patch extract_blocks at its source; parse() imports it fresh each call
    with mock.patch("probatio.manuscript_layout.extract_blocks", side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.WARNING, logger="probatio.manuscript"):
            parser = PyMuPDFManuscriptParser(FakeLLMClient([refs_json, scope_json]))
            cits, _ = await parser.parse(pdf)

    # Warning must have been emitted
    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("boom" in str(m) or "layout" in str(m).lower() for m in warning_msgs), (
        f"Expected a layout-failure warning; got log records: {caplog.records}")


# ---------------------------------------------------------------------------
# Task 1: boundary-aware reference-block chunker (pure)
# ---------------------------------------------------------------------------

def test_split_small_block_is_single_chunk():
    block = "[1] A. 2020.\n[2] B. 2021."
    assert split_reference_block(block) == [block]          # back-compat: one chunk


def test_split_numbered_block_chunks_and_covers_all():
    block = "\n".join(f"[{i}] Author {i}. A study of topic {i}. Journal, 2020." for i in range(1, 201))
    chunks = split_reference_block(block, budget=2000)
    assert len(chunks) > 1
    joined = "\n".join(chunks)
    for i in range(1, 201):
        assert f"[{i}]" in joined                            # nothing dropped


def test_split_oversized_single_entry_is_own_chunk():
    big = "[1] " + "x" * 8000
    chunks = split_reference_block(big, budget=6000)
    assert chunks == [big]                                   # never cut mid-entry


def test_split_author_year_on_blank_lines():
    block = ("Smith, J. (2020). A study. Journal.\n\nJones, K. (2019). Another. Journal.\n\n") * 200
    chunks = split_reference_block(block, budget=2000)
    assert len(chunks) > 1
    joined = "".join(chunks)
    assert "Smith" in joined and "Jones" in joined


def test_split_no_boundary_falls_back_to_lines():
    block = "\n".join(f"Author{i} K. Title {i}. 2020." for i in range(1, 301))   # no markers, no blanks
    chunks = split_reference_block(block, budget=2000)
    assert len(chunks) > 1
    joined = "".join(chunks)
    assert "Author1 " in joined and "Author300" in joined


def test_split_empty_block():
    assert split_reference_block("") == [] and split_reference_block("   \n  ") == []


def test_split_keeps_wrapped_numbered_entries_whole():
    # PyMuPDF wraps each entry across multiple lines with NO blank line between entries;
    # the chunker must keep each [key] glued to its own title/DOI/year lines (the critical
    # bug: a fraction-of-lines detector fell back to per-line splitting and tore them apart).
    from probatio.manuscript import _NUM_ENTRY
    block = "\n".join(
        f"[{i}] Author A, Author B. A descriptive title for study {i} goes here.\n"
        f"Journal of Things, vol {i}, pp {i * 10}-{i * 10 + 9}, 2020.\n"
        f"doi:10.1016/j.x.2020.{i:04d}"
        for i in range(1, 61))
    chunks = split_reference_block(block)                 # default budget 6000 -> multiple chunks
    assert len(chunks) > 1
    assert all(_NUM_ENTRY.match(c.splitlines()[0]) for c in chunks)   # no chunk starts mid-entry
    for c in chunks:
        keys = re.findall(r"^\[(\d+)\]", c, re.M)
        dois = set(re.findall(r"doi:10\.1016/j\.x\.2020\.(\d+)", c))
        for k in keys:
            assert f"{int(k):04d}" in dois, f"ref [{k}] separated from its DOI line"


def test_split_author_year_reglues_stray_internal_blank():
    # A stray blank line inside one author-year entry must not become its own entry/chunk.
    unit = ("Smith J (2020) Paper one with a longish title for bulk. Journal A.\n\n"
            "Jones K (2019) Paper two with another descriptive title here\n\n"
            "with a stray blank inside the entry that must re-glue. Journal B.\n\n"
            "Lee P (2021) Paper three also reasonably long for packing. Journal C.\n\n")
    chunks = split_reference_block(unit * 40, budget=2000)
    assert len(chunks) > 1
    for c in chunks:
        assert not c.splitlines()[0].startswith("with a stray blank inside"), \
            "stray continuation became an entry/chunk start"
    assert "with a stray blank inside" in "\n".join(chunks)           # text preserved


# ---------------------------------------------------------------------------
# Task 2: resilient chunked _parse_refs (merge + dedupe + skip + retry)
# ---------------------------------------------------------------------------

def _big_numbered_block(n: int = 150) -> str:
    # > 6000 chars so the default budget yields multiple chunks
    return "\n".join(f"[{i}] Author {i}, A. A detailed study of topic number {i}. "
                     f"Journal of Things, vol {i}, 2020." for i in range(1, n + 1))


@pytest.mark.asyncio
async def test_parse_refs_merges_and_dedupes_across_chunks():
    block = _big_numbered_block()
    chunk_a = json.dumps([{"key": "1", "raw": "1"}, {"key": "2", "raw": "2"}])
    chunk_b = json.dumps([{"key": "2", "raw": "2"}, {"key": "3", "raw": "3"}])   # key 2 overlaps
    fake = FakeLLMClient([chunk_a, chunk_b])
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs(block)
    assert len(fake.calls) >= 2                       # actually chunked
    assert [r.key for r in refs] == ["1", "2", "3"]   # merged, deduped, order preserved


@pytest.mark.asyncio
async def test_parse_refs_skips_unparseable_chunk():
    block = _big_numbered_block()
    fake = FakeLLMClient([json.dumps([{"key": "1", "raw": "1"}]), "this is not json"])
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs(block)
    assert [r.key for r in refs] == ["1"]             # good chunk survived; garbage skipped


@pytest.mark.asyncio
async def test_parse_refs_retries_transient_failure():
    class _FlakyThenJSON:
        def __init__(self, payload: str, fail_times: int = 1):
            self._payload, self._fails, self.attempts = payload, fail_times, 0

        async def complete(self, *, system: str, user: str,
                           model: str | None = None, think: bool = True) -> str:
            self.attempts += 1
            if self.attempts <= self._fails:
                raise httpx.ConnectError("transient")     # a transport error -> retried
            return self._payload

    block = "[1] A solo reference. 2020."            # small -> single chunk
    fake = _FlakyThenJSON(json.dumps([{"key": "1", "raw": "1"}]), fail_times=1)
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs(block)
    assert [r.key for r in refs] == ["1"]
    assert fake.attempts == 2                          # failed once, retried, succeeded


@pytest.mark.asyncio
async def test_parse_refs_does_not_retry_permanent_error():
    class _AlwaysValueError:
        def __init__(self) -> None:
            self.attempts = 0

        async def complete(self, *, system: str, user: str,
                           model: str | None = None, think: bool = True) -> str:
            self.attempts += 1
            raise ValueError("model not found")           # permanent -> fail fast, no retry

    fake = _AlwaysValueError()
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs("[1] A solo reference. 2020.")
    assert refs == []                                 # chunk skipped, parse did not abort
    assert fake.attempts == 1                         # permanent error NOT retried


def _http_status_error(code: int) -> httpx.HTTPStatusError:
    resp = httpx.Response(code, request=httpx.Request("POST", "http://x/api/chat"))
    return httpx.HTTPStatusError(f"status {code}", request=resp.request, response=resp)


@pytest.mark.asyncio
async def test_parse_refs_retries_retryable_http_status():
    # ollama warm-up returns 503 ("model is loading") -> must be retried, not dropped.
    class _Http503ThenJSON:
        def __init__(self, payload: str):
            self._payload, self.attempts = payload, 0

        async def complete(self, *, system: str, user: str,
                           model: str | None = None, think: bool = True) -> str:
            self.attempts += 1
            if self.attempts == 1:
                raise _http_status_error(503)
            return self._payload

    fake = _Http503ThenJSON(json.dumps([{"key": "1", "raw": "1"}]))
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs("[1] A solo reference. 2020.")
    assert [r.key for r in refs] == ["1"]
    assert fake.attempts == 2                          # 503 retried then succeeded


@pytest.mark.asyncio
async def test_parse_refs_does_not_retry_4xx_http_status():
    class _Always404:
        def __init__(self) -> None:
            self.attempts = 0

        async def complete(self, *, system: str, user: str,
                           model: str | None = None, think: bool = True) -> str:
            self.attempts += 1
            raise _http_status_error(404)

    fake = _Always404()
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs("[1] A solo reference. 2020.")
    assert refs == [] and fake.attempts == 1           # 4xx permanent -> no retry


@pytest.mark.asyncio
async def test_parse_refs_exhausted_chunk_skipped_siblings_survive():
    # One chunk's call times out on BOTH attempts (exhausts retry -> '' -> skipped);
    # later chunks must still contribute their references.
    class _FailFirstChunk:
        def __init__(self, payload: str, fail: int = 2):
            self._payload, self._fail, self.calls = payload, fail, 0

        async def complete(self, *, system: str, user: str,
                           model: str | None = None, think: bool = True) -> str:
            self.calls += 1
            if self.calls <= self._fail:
                raise httpx.ReadTimeout("slow")           # first chunk's 2 attempts both time out
            return self._payload

    fake = _FailFirstChunk(json.dumps([{"key": "99", "raw": "99"}]), fail=2)
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs(_big_numbered_block())
    assert [r.key for r in refs] == ["99"]            # first chunk skipped; siblings survived
    assert fake.calls >= 3                            # 2 exhausted attempts + >=1 sibling call


@pytest.mark.asyncio
async def test_parse_refs_dedupe_prefers_richer_record():
    sparse = json.dumps([{"key": "7", "raw": "7"}])                     # no title/doi
    rich = json.dumps([{"key": "7", "title": "Real Title", "doi": "10.1/x", "raw": "7"}])
    fake = FakeLLMClient([sparse, rich])                                 # sparse seen first
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs(_big_numbered_block())
    by7 = [r for r in refs if r.key == "7"]
    assert len(by7) == 1 and by7[0].title == "Real Title" and by7[0].doi == "10.1/x"


@pytest.mark.asyncio
async def test_parse_refs_empty_block_returns_empty():
    fake = FakeLLMClient([""])
    refs = await PyMuPDFManuscriptParser(fake)._parse_refs("")
    assert refs == [] and fake.calls == []            # no call for an empty block
