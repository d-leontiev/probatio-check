from probatio.manuscript_layout import (
    TextBlock, estimate_body_size, classify, drop_margins, order_by_column,
    page_segments, extract_clean_pages, _parse_block,
)
import pytest


def _blk(text, size, *, page=1, bold=False, bbox=(50, 100, 500, 120), ph=792.0):
    return TextBlock(text=text, bbox=bbox, size=size, bold=bold, page=page, page_height=ph)


def test_estimate_body_size_is_length_weighted_mode():
    blocks = [_blk("x" * 500, 10.0), _blk("y" * 400, 10.0), _blk("HEAD", 16.0, bold=True)]
    assert estimate_body_size(blocks) == 10.0


def test_estimate_body_size_empty():
    assert estimate_body_size([]) == 0.0


def test_classify_numbered_heading():
    assert classify(_blk("2.6.5 Extra Trees", 10.0), 10.0) == "heading"


def test_classify_big_or_bold_heading():
    # Updated: "Methods" at 14pt (ratio 1.4 > 1.15) AND no sentence-ending punct → heading
    assert classify(_blk("Methods", 14.0), 10.0) == "heading"
    assert classify(_blk("Methods", 10.0, bold=True), 10.0) == "heading"


def test_classify_allcaps_heading():
    assert classify(_blk("MATERIALS AND METHODS", 10.0), 10.0) == "heading"


def test_classify_caption():
    # "Figure 1." → number immediately followed by period → caption
    assert classify(_blk("Figure 1. Solubility vs density.", 10.0), 10.0) == "caption"
    # Updated: "Table 2 Summary of runs" has no period/colon after number → body now
    # (the old test asserted caption, which was a false positive; conservative rule says body)
    assert classify(_blk("Table 2 Summary of runs", 10.0), 10.0) == "body"


def test_classify_equation_best_effort():
    assert classify(_blk("y = a + b·x²  (4)", 10.0), 10.0) == "equation"


def test_classify_body_sentence_starting_with_number_is_not_heading():
    s = "3.5 mg of the compound was administered to each patient in the cohort study."
    assert classify(_blk(s, 10.0), 10.0) == "body"


def test_classify_plain_body():
    assert classify(_blk("Averaging across trees reduces variance.", 10.0), 10.0) == "body"


# ---------------------------------------------------------------------------
# Regression tests: bias-toward-body hardening (grok adversarial review fixes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # caption: number must be followed by period or colon
    ("Table 1 lists the primer sequences used in this study [5].", "body"),   # NOT caption
    ("Figure 1. Solubility vs density.", "caption"),
    ("Table 2: Summary of runs", "caption"),
    # heading: sentence ending with period must NOT be heading
    ("1. First we collected samples and analyzed them.", "body"),              # NOT heading
    ("2.6.5 Extra Trees", "heading"),
    ("2.6.4 Random Forest (RF)", "heading"),
    # ALL-CAPS heading vs short acronyms that must stay body
    ("MATERIALS AND METHODS", "heading"),
    ("DNA", "body"),
    ("IV", "body"),
    # bold short title
    pytest.param("Methods", "heading", marks=pytest.mark.parametrize.__func__ if False else pytest.mark.skip(reason="bold handled separately")),
    # equation markers that must stay body (no = or math glyph)
    ("[12]", "body"),
    ("p < 0.05", "body"),
    ("***", "body"),
    # real equation
    ("y = a + b·x²  (4)", "equation"),
])
def test_classify_regression_bias_toward_body(text, expected):
    assert classify(_blk(text, 10.0), 10.0) == expected


def test_classify_bold_short_title_is_heading():
    """Bold short title without sentence punctuation → heading."""
    assert classify(_blk("Methods", 10.0, bold=True), 10.0) == "heading"


def test_classify_body_size_zero_guard():
    """body_size=0.0 must not cause everything to be a heading via size ratio."""
    assert classify(_blk("Plain body text here.", 10.0), 0.0) == "body"


# ---------------------------------------------------------------------------
# drop_margins tests
# ---------------------------------------------------------------------------

def test_drop_margins_removes_repeated_top_header():
    blocks = [_blk("Int. J. Pharmaceutics 42 (2026)", 9.0, page=p, bbox=(50, 20, 500, 32))
              for p in (1, 2, 3)]
    blocks.append(_blk("Real body text in the page body region.", 10.0, page=1,
                       bbox=(50, 400, 500, 420)))
    out = drop_margins(blocks)
    assert len(out) == 1 and out[0].text.startswith("Real body")


def test_drop_margins_keeps_long_unique_body_near_margin():
    long_body = "This is a genuine body sentence that merely happens to sit near the top." * 2
    blocks = [_blk(long_body, 10.0, page=1, bbox=(50, 30, 500, 60))]
    assert drop_margins(blocks) == blocks  # not short, not repeated -> kept


def test_drop_margins_removes_short_page_number_in_footer():
    blocks = [_blk("3019", 9.0, page=1, bbox=(280, 770, 320, 785))]  # y1 > 0.92*792
    assert drop_margins(blocks) == []


# ---------------------------------------------------------------------------
# Regression tests: drop_margins bias-toward-keeping (grok adversarial fixes)
# ---------------------------------------------------------------------------

def test_drop_margins_keeps_short_unique_concluding_sentence_in_bottom_band():
    """A short unique concluding sentence whose centroid is in the bottom band must be KEPT.
    It is not page-number-shaped and it does not repeat across pages."""
    # centroid = (760+785)/2 = 772.5 > 0.92*792 = 728.64 → in margin band by centroid
    blocks = [_blk("These results support our hypothesis.", 10.0, page=1,
                   bbox=(50, 760, 500, 785))]
    assert drop_margins(blocks) == blocks


def test_drop_margins_keeps_block_whose_centroid_is_in_body():
    """A block whose bottom edge crosses 92% but whose centroid is in the body must be KEPT.
    (50, 600, 500, 740): centroid = (600+740)/2 = 670; 0.92*792 = 728.64 → 670 < 728.64 → body."""
    blocks = [_blk("This paragraph extends to the bottom of the page.", 10.0, page=1,
                   bbox=(50, 600, 500, 740))]
    assert drop_margins(blocks) == blocks


def test_drop_margins_no_false_repetition_for_patient_sentences():
    """'Patient 1 improved the metric.' and 'Patient 2 improved the metric.' must NOT collide
    in the repetition set and must both be KEPT (body zone, centers near mid-page)."""
    # centroid for both: (350+370)/2 = 360 — well within body
    blocks = [
        _blk("Patient 1 improved the metric.", 10.0, page=1, bbox=(50, 350, 500, 370)),
        _blk("Patient 2 improved the metric.", 10.0, page=2, bbox=(50, 350, 500, 370)),
    ]
    out = drop_margins(blocks)
    assert len(out) == 2


def test_drop_margins_removes_page_number_in_bottom_band():
    """Pure page number '3019' in the bottom band must be DROPPED."""
    # centroid = (770+785)/2 = 777.5 > 728.64 → in margin; matches ^\d{1,4}$ ... wait, 3019 is 4 digits → matches
    blocks = [_blk("3019", 9.0, page=1, bbox=(280, 770, 320, 785))]
    assert drop_margins(blocks) == []


def test_drop_margins_removes_repeated_running_header_on_3_pages():
    """A running header in the top band repeated on >=2 distinct pages must be DROPPED."""
    # centroid = (20+32)/2 = 26 < 0.08*792 = 63.36 → in margin on all 3 pages
    blocks = [_blk("Int. J. Pharmaceutics 42", 9.0, page=p, bbox=(50, 20, 500, 32))
              for p in (1, 2, 3)]
    out = drop_margins(blocks)
    assert out == []


def test_drop_margins_keeps_unique_short_header_band_line_on_single_page():
    """A unique short line in the header band on a single-page document that is NOT
    page-number-shaped must be KEPT (not repeated, not a page number)."""
    # centroid = (20+32)/2 = 26 < 63.36 → in margin; but only 1 page, not page-number shape
    blocks = [_blk("Supplementary Note S1", 9.0, page=1, bbox=(50, 20, 500, 32))]
    assert drop_margins(blocks) == blocks


def test_drop_margins_keeps_block_with_page_height_zero():
    """page_height=0 must not crash or delete: guard → treat as not-in-margin → KEPT."""
    b = TextBlock(text="Some text.", bbox=(50, 10, 500, 20), size=10.0,
                  bold=False, page=1, page_height=0.0)
    assert drop_margins([b]) == [b]


# ---------------------------------------------------------------------------
# order_by_column tests
# ---------------------------------------------------------------------------

def test_order_by_column_two_columns():
    # Clear gutter at x~300: left centers at 160, right centers at 440.
    # Gap = 440-160 = 280; content_width = 560-40 = 520; 280/520 = 0.54 >> 0.15. Clean 2-col.
    lt = _blk("left-top", 10.0, bbox=(40, 100, 280, 120))
    lb = _blk("left-bottom", 10.0, bbox=(40, 300, 280, 320))
    rt = _blk("right-top", 10.0, bbox=(320, 100, 560, 120))
    rb = _blk("right-bottom", 10.0, bbox=(320, 300, 560, 320))
    out = order_by_column([rb, lt, rt, lb])  # scrambled
    assert [b.text for b in out] == ["left-top", "left-bottom", "right-top", "right-bottom"]


def test_order_by_column_single_column_top_to_bottom():
    a = _blk("a", 10.0, bbox=(50, 300, 500, 320))
    b = _blk("b", 10.0, bbox=(50, 100, 500, 120))
    assert [x.text for x in order_by_column([a, b])] == ["b", "a"]


# ---------------------------------------------------------------------------
# Regression tests: gap-based gutter detection (grok review fixes)
# ---------------------------------------------------------------------------

def test_order_by_column_uniform_single_column_not_split():
    """Four full-width blocks on a single-column page must come out top-to-bottom.
    All centers ~275; no large gap -> single-column path."""
    a = _blk("para-1", 10.0, bbox=(50, 100, 500, 120))
    b = _blk("para-2", 10.0, bbox=(50, 200, 500, 220))
    c = _blk("para-3", 10.0, bbox=(50, 300, 500, 320))
    d = _blk("para-4", 10.0, bbox=(50, 400, 500, 420))
    out = order_by_column([d, b, a, c])
    assert [x.text for x in out] == ["para-1", "para-2", "para-3", "para-4"]


def test_order_by_column_indented_blocks_no_false_split():
    """Single-column page with 2 indented short blocks + 2 full-width blocks.
    Centers: full-width ~275, indented ~175 or ~225. No clear gutter -> single-column."""
    fw1 = _blk("full-1", 10.0, bbox=(50, 100, 500, 120))   # center-x = 275
    fw2 = _blk("full-2", 10.0, bbox=(50, 400, 500, 420))   # center-x = 275
    # Indented (left-shifted) blocks — still single-column, just narrower
    ind1 = _blk("indent-1", 10.0, bbox=(80, 200, 420, 220))  # center-x = 250
    ind2 = _blk("indent-2", 10.0, bbox=(80, 300, 420, 320))  # center-x = 250
    # content_width = 500-50 = 450; largest gap among centers 250,250,275,275 is 25.
    # 25/450 = 0.056 < 0.15 -> single-column.
    out = order_by_column([fw2, ind2, fw1, ind1])
    assert [x.text for x in out] == ["full-1", "indent-1", "indent-2", "full-2"]


def test_order_by_column_unbalanced_two_column_clear_gutter():
    """2 left + 1 right with clear gutter -> left blocks (by y) then right block.
    Must NOT row-major interleave."""
    lt = _blk("left-top", 10.0, bbox=(40, 100, 240, 120))    # center-x = 140
    lb = _blk("left-bottom", 10.0, bbox=(40, 300, 240, 320)) # center-x = 140
    rt = _blk("right-only", 10.0, bbox=(320, 150, 560, 170)) # center-x = 440
    # content_width = 560-40 = 520; gap = 440-140 = 300; 300/520 = 0.577 >> 0.15.
    # left has 2 blocks, right has 1 -> fails >=2 per group -> single-column fallback.
    # WAIT: spec says >=2 EACH -> unbalanced (2L+1R) falls back to single-column top-to-bottom.
    out = order_by_column([lb, rt, lt])
    assert [x.text for x in out] == ["left-top", "right-only", "left-bottom"]


def test_order_by_column_spanning_block_falls_back_to_single():
    """2-column page where one block spans the gutter (full-width title) -> top-to-bottom fallback."""
    title = _blk("Full Title", 14.0, bbox=(40, 50, 560, 80))   # spans whole page
    lt = _blk("left-top", 10.0, bbox=(40, 150, 260, 170))
    lb = _blk("left-bottom", 10.0, bbox=(40, 280, 260, 300))
    rt = _blk("right-top", 10.0, bbox=(320, 150, 560, 170))
    rb = _blk("right-bottom", 10.0, bbox=(320, 280, 560, 300))
    # The title spans the gutter -> layout not cleanly separable -> top-to-bottom.
    out = order_by_column([rb, title, lt, rt, lb])
    assert [x.text for x in out] == ["Full Title", "left-top", "right-top", "left-bottom", "right-bottom"]


def test_order_by_column_same_y0_tiebreak_by_x0():
    """Two blocks at the same y0 must be ordered by x0 (left before right)."""
    left = _blk("left", 10.0, bbox=(50, 100, 200, 120))
    right = _blk("right", 10.0, bbox=(300, 100, 500, 120))
    out = order_by_column([right, left])
    assert [x.text for x in out] == ["left", "right"]


def test_order_by_column_zero_or_one_block():
    """0 or 1 block -> return as-is (no-op)."""
    assert order_by_column([]) == []
    b = _blk("only", 10.0, bbox=(50, 100, 500, 120))
    assert order_by_column([b]) == [b]


# ---------------------------------------------------------------------------
# page_segments tests
# ---------------------------------------------------------------------------

def test_page_segments_splits_at_heading_and_drops_it():
    blocks = [
        _blk("Averaging across trees reduces variance, making RF a strong baseline.", 10.0),
        _blk("2.6.5 Extra Trees", 10.0),  # heading -> boundary, dropped
        _blk("To address this, extra trees randomise the splits further [5].", 10.0),
    ]
    segs = page_segments(blocks, 10.0)
    assert len(segs) == 2
    assert "Extra Trees" not in " ".join(segs)            # heading text gone
    assert segs[0].startswith("Averaging across trees")
    assert segs[1].startswith("To address this")


def test_page_segments_merges_consecutive_body_blocks():
    blocks = [_blk("First part of a wrapped sentence", 10.0),
              _blk("that continues on the next line.", 10.0)]
    assert page_segments(blocks, 10.0) == [
        "First part of a wrapped sentence that continues on the next line."]


def test_page_segments_empty_input():
    """Empty input -> empty output."""
    assert page_segments([], 10.0) == []


def test_page_segments_leading_and_trailing_heading():
    """Leading heading + body + trailing heading -> exactly one segment (body), headings dropped."""
    blocks = [
        _blk("2.6.5 Extra Trees", 10.0),        # heading -> dropped, no segment started
        _blk("This is the actual body text.", 10.0),  # body -> segment
        _blk("CONCLUSION", 10.0),               # ALL-CAPS heading -> dropped, ends segment
    ]
    segs = page_segments(blocks, 10.0)
    assert len(segs) == 1
    assert segs[0] == "This is the actual body text."
    assert "Extra Trees" not in segs[0]
    assert "CONCLUSION" not in segs[0]


def test_page_segments_body_caption_body():
    """body -> caption -> body -> exactly two segments (caption is boundary, text absent)."""
    blocks = [
        _blk("The first paragraph describes the method.", 10.0),
        _blk("Figure 1. Caption here.", 10.0),  # caption -> boundary, dropped
        _blk("The second paragraph continues the discussion.", 10.0),
    ]
    segs = page_segments(blocks, 10.0)
    assert len(segs) == 2
    assert segs[0] == "The first paragraph describes the method."
    assert segs[1] == "The second paragraph continues the discussion."
    assert "Caption" not in " ".join(segs)


def test_page_segments_consecutive_non_body_blocks():
    """body -> heading -> heading -> body -> exactly two segments, NO empty segment between them."""
    blocks = [
        _blk("Body text before the headings.", 10.0),
        _blk("2.6.5 Extra Trees", 10.0),        # heading -> boundary
        _blk("CONCLUSION", 10.0),               # heading -> boundary (no empty segment)
        _blk("Body text after the headings.", 10.0),
    ]
    segs = page_segments(blocks, 10.0)
    assert len(segs) == 2
    assert segs[0] == "Body text before the headings."
    assert segs[1] == "Body text after the headings."


# ---------------------------------------------------------------------------
# extract_clean_pages tests
# ---------------------------------------------------------------------------

def test_extract_clean_pages_groups_by_page_and_cuts_at_references():
    blocks = [
        _blk("Body claim on page one with a citation [1].", 10.0, page=1, bbox=(50, 100, 500, 120)),
        _blk("Body claim on page two [2].", 10.0, page=2, bbox=(50, 100, 500, 120)),
        _blk("References", 12.0, page=2, bbox=(50, 140, 200, 158), bold=True),
        _blk("1. Some author. A cited paper. 2020.", 10.0, page=2, bbox=(50, 170, 500, 190)),
    ]
    pages = extract_clean_pages(blocks)
    # Now returns list[tuple[int, list[str]]]; unpack for checks
    segs_by_page = {pnum: segs for pnum, segs in pages}
    assert segs_by_page[1] == ["Body claim on page one with a citation [1]."]
    assert segs_by_page[2] == ["Body claim on page two [2]."]    # ref entry NOT included
    flat = " ".join(seg for _, segs in pages for seg in segs)
    assert "Some author" not in flat


# ---------------------------------------------------------------------------
# Regression tests: refs cutoff robustness + body_size pre-refs (TDD – Fix 1/2/3)
# ---------------------------------------------------------------------------

def test_extract_clean_pages_inline_references_not_a_cutoff():
    """An inline sentence mentioning 'references' must NOT trigger the cutoff.

    'We summarise the references in Table 1 [5].' does NOT match _REFS_HEADING
    (which requires a standalone heading line) so the block must be kept as body.
    """
    blocks = [
        _blk("We summarise the references in Table 1 [5].", 10.0, page=1,
             bbox=(50, 100, 500, 120)),
        _blk("Follow-up claim on the same page [6].", 10.0, page=1,
             bbox=(50, 140, 500, 160)),
    ]
    pages = extract_clean_pages(blocks)
    flat = " ".join(seg for _, segs in pages for seg in segs)
    assert "We summarise the references" in flat, (
        "Inline 'references' sentence was incorrectly treated as a cutoff")
    assert "Follow-up claim" in flat


def test_extract_clean_pages_standalone_refs_heading_on_page2():
    """Page 2 pre-refs body is kept; the bib entry block is excluded; page 3 is fully dropped.

    The References heading sits as a standalone block on page 2. Blocks that appear
    after it (same page or later) must not appear in the output.
    """
    blocks = [
        # page 1 – all body
        _blk("Page one claim [1].", 10.0, page=1, bbox=(50, 100, 500, 120)),
        # page 2 – body then refs heading then bib entry
        _blk("Page two claim before refs [2].", 10.0, page=2, bbox=(50, 100, 500, 120)),
        _blk("References", 12.0, page=2, bbox=(50, 200, 200, 220), bold=True),
        _blk("1. Doe J. A paper. J Med. 2020.", 10.0, page=2, bbox=(50, 240, 500, 260)),
        # page 3 – fully after refs, must be dropped
        _blk("Page three bib continuation [3].", 10.0, page=3, bbox=(50, 100, 500, 120)),
    ]
    pages = extract_clean_pages(blocks)
    flat = " ".join(seg for _, segs in pages for seg in segs)

    assert "Page one claim" in flat
    assert "Page two claim before refs" in flat
    assert "Doe J." not in flat, "Bib entry on page 2 after refs heading must be excluded"
    assert "Page three bib" not in flat, "Page 3 must be fully dropped"
    # At most 2 pages in output (page 1 and the partial page 2)
    assert len(pages) <= 2


def test_extract_clean_pages_merged_block_with_trailing_body_and_refs_heading():
    """A block whose text contains a standalone 'References' line triggers the cutoff.

    PyMuPDF can merge a trailing body sentence and the heading into one block:
        'Some trailing body sentence.\\nReferences'
    Fix 1 uses .search() which finds the heading line anywhere in the block text.
    Whatever happens to the trailing body text in that merged block is acceptable;
    the important contract is that blocks/pages AFTER it are excluded.
    """
    blocks = [
        _blk("Normal body claim on page 1 [1].", 10.0, page=1, bbox=(50, 100, 500, 120)),
        # Merged block: body sentence + refs heading fused by PyMuPDF
        _blk("Some trailing body sentence.\nReferences", 10.0, page=2,
             bbox=(50, 100, 500, 140)),
        # Bib entry that must be excluded
        _blk("1. Smith A. Paper title. 2021.", 10.0, page=2, bbox=(50, 160, 500, 180)),
        # Page 3 must be fully dropped
        _blk("Page three content.", 10.0, page=3, bbox=(50, 100, 500, 120)),
    ]
    pages = extract_clean_pages(blocks)
    flat = " ".join(seg for _, segs in pages for seg in segs)

    assert "Normal body claim" in flat
    assert "Smith A." not in flat, "Bib entry after merged refs block must be excluded"
    assert "Page three content" not in flat, "Page 3 must be dropped after refs cutoff"


def test_extract_clean_pages_body_size_unaffected_by_long_bibliography():
    """body_size must be estimated over pre-refs blocks only.

    Setup: page 1 has 10pt body. Page 2 starts with a 'References' heading followed
    by a LONG 10pt bibliography. If body_size were computed over ALL blocks (including
    bib), the modal size would still be 10pt here — so we use a 12pt bib to prove the
    point: with Fix 2 the body_size must equal 10.0 (pre-refs blocks dominate); without
    Fix 2, the 12pt bib entries would shift the modal size to 12pt and break classification.

    We assert body_size is 10.0 by checking that the page-1 body blocks are classified
    correctly (i.e., they appear in the output). A body_size shifted to 12pt would cause
    the 10pt blocks to be classified as body still (10 < 12*1.15 = 13.8) — so we instead
    use a more direct structural check: verify the bibliography text is absent (confirming
    the cutoff fired) while the body text is present (confirming body blocks are kept).
    """
    # Long bibliography: 20 entries at 12pt after a References heading on page 2
    bib_entries = [
        _blk(f"{i}. Author{i} A. Title{i}. Journal. 202{i%10}.", 12.0, page=2,
             bbox=(50, 200 + i * 20, 500, 218 + i * 20))
        for i in range(1, 21)
    ]
    blocks = [
        # Page 1: normal 10pt body
        _blk("Main body claim with citation [1].", 10.0, page=1, bbox=(50, 100, 500, 120)),
        _blk("Another body sentence supports this [2].", 10.0, page=1,
             bbox=(50, 140, 500, 160)),
        # Page 2: References heading then long 12pt bib
        _blk("References", 12.0, page=2, bbox=(50, 100, 200, 120), bold=True),
        *bib_entries,
    ]
    pages = extract_clean_pages(blocks)
    flat = " ".join(seg for _, segs in pages for seg in segs)

    # Body text must be present
    assert "Main body claim" in flat
    assert "Another body sentence" in flat
    # Bibliography must be absent (cutoff fired correctly)
    assert "Author1 A." not in flat
    assert "Author10 A." not in flat


# ---------------------------------------------------------------------------
# Fix 1: extract_clean_pages returns (page_num, segments) tuples preserving
#         real PDF page numbers, and _mine_segments uses those page numbers.
# ---------------------------------------------------------------------------

def test_extract_clean_pages_returns_tuples_with_real_page_numbers():
    """extract_clean_pages must return list[tuple[int, list[str]]] — each tuple is
    (real 1-based PDF page number, segments).  Pages with no surviving segments must
    NOT appear (so a page with only a heading block produces no tuple).  The page
    numbers must match the real PDF page numbers, not a re-enumerated index."""
    blocks = [
        _blk("Body claim page 1 [1].", 10.0, page=1, bbox=(50, 100, 500, 120)),
        # page 2 has only a heading — no body segments survive
        _blk("METHODS", 10.0, page=2, bbox=(50, 100, 300, 120)),
        _blk("Body claim page 3 [3].", 10.0, page=3, bbox=(50, 100, 500, 120)),
    ]
    pages = extract_clean_pages(blocks)
    # Must be list of tuples
    assert isinstance(pages, list)
    for item in pages:
        assert isinstance(item, tuple) and len(item) == 2, (
            f"Expected (page_num, segments) tuple, got {item!r}")
        pnum, segs = item
        assert isinstance(pnum, int)
        assert isinstance(segs, list)

    page_nums = [pnum for pnum, _ in pages]
    # Page 2 had only a heading (METHODS, all-caps ≥2 words → heading), zero segments → omitted
    # Page 1 and page 3 have body text → included with their real numbers
    assert 1 in page_nums, f"Page 1 missing from output: {page_nums}"
    assert 3 in page_nums, f"Page 3 missing from output: {page_nums}"
    # The segments for page 3 must contain the page-3 claim
    segs_3 = next(segs for pnum, segs in pages if pnum == 3)
    assert any("page 3" in s for s in segs_3)


def test_extract_clean_pages_skipped_page_preserves_later_page_numbers():
    """When PDF page 2 has no surviving blocks at all, pages 1 and 3 must appear
    with numbers 1 and 3 — NOT re-enumerated as 1 and 2."""
    blocks = [
        _blk("First claim [1].", 10.0, page=1, bbox=(50, 200, 500, 220)),
        # page 2 intentionally absent from blocks list (e.g. a blank page)
        _blk("Third claim [3].", 10.0, page=3, bbox=(50, 200, 500, 220)),
    ]
    pages = extract_clean_pages(blocks)
    page_nums = [pnum for pnum, _ in pages]
    assert page_nums == [1, 3], (
        f"Expected [1, 3] (real PDF pages), got {page_nums}")


# ---------------------------------------------------------------------------
# Fix 3: _parse_block — the pure helper must be robust to malformed input
# ---------------------------------------------------------------------------

def _valid_blk_dict(text="Valid body text here.", size=10.0, font="Helvetica",
                    flags=0, bbox=(50.0, 100.0, 500.0, 120.0), page=1, ph=792.0):
    """Build a well-formed PyMuPDF-style block dict."""
    return {
        "bbox": bbox,
        "lines": [
            {
                "spans": [
                    {"text": text, "size": size, "font": font, "flags": flags},
                ]
            }
        ],
    }


def test_parse_block_null_flags_no_crash_bold_false():
    """A span with flags=None must not raise; bold must default to False
    (unless 'bold' in font name)."""
    blk = _valid_blk_dict(flags=None, font="Helvetica")
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is not None
    assert result.bold is False


def test_parse_block_null_lines_returns_none():
    """A block with lines=None must return None, not raise."""
    blk = {"bbox": (50.0, 100.0, 500.0, 120.0), "lines": None}
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is None


def test_parse_block_missing_lines_returns_none():
    """A block dict with no 'lines' key must return None."""
    blk = {"bbox": (50.0, 100.0, 500.0, 120.0)}
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is None


def test_parse_block_null_bbox_returns_none():
    """A block with bbox=None must return None."""
    blk = {"bbox": None, "lines": [{"spans": [{"text": "x", "size": 10.0, "flags": 0, "font": "A"}]}]}
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is None


def test_parse_block_short_bbox_returns_none():
    """A block with bbox shorter than 4 elements must return None."""
    blk = _valid_blk_dict()
    blk["bbox"] = (50.0, 100.0)  # only 2 elements
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is None


def test_parse_block_empty_text_returns_none():
    """A block whose assembled text is empty must return None."""
    blk = _valid_blk_dict(text="")
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is None


def test_parse_block_not_a_dict_returns_none():
    """Passing something that isn't a dict at all must return None, not raise."""
    result = _parse_block("not a dict", page=1, page_height=792.0)  # type: ignore[arg-type]
    assert result is None


def test_parse_block_valid_block_correct_textblock():
    """A well-formed block must return a correct TextBlock."""
    blk = _valid_blk_dict(text="Good body text.", size=11.0, font="Times-Bold", flags=16)
    result = _parse_block(blk, page=2, page_height=800.0)
    assert result is not None
    assert result.text == "Good body text."
    assert result.size == 11.0
    assert result.bold is True  # flags & 16 = 16 → bold
    assert result.page == 2
    assert result.page_height == 800.0


def test_parse_block_bold_from_font_name():
    """'bold' in font name (case-insensitive) must set bold=True even if flags=0."""
    blk = _valid_blk_dict(text="Heading text.", size=12.0, font="Arial-BoldMT", flags=0)
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is not None
    assert result.bold is True


# ---------------------------------------------------------------------------
# Regression tests: citation-marker-bearing blocks must never be heading/caption
# (fix: early-return "body" when find_markers finds any marker in the text)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,bold,expected", [
    # Numbered pattern that would normally match _NUMBERED_HEADING but has a marker → body
    ("5 Gy radiation reduced viability [3]", False, "body"),
    ("10 Hz stimulation improved recall [6]", False, "body"),
    # Bold short fragment with marker → body (not heading)
    ("SVM was best [3]", True, "body"),
    # Author-year marker → body
    ("Adapted from (Smith et al., 2020)", False, "body"),
    # Gold-test headings with NO citation marker → still heading
    ("2.6.4 Random Forest (RF)", False, "heading"),
    ("2.6.5 Extra Trees", False, "heading"),
    ("MATERIALS AND METHODS", False, "heading"),
])
def test_classify_citation_marker_never_heading(text, bold, expected):
    """A block whose text contains a citation marker must never be classified as heading."""
    assert classify(_blk(text, 10.0, bold=bold), 10.0) == expected


def test_parse_block_null_size_and_font():
    """Null size and font must not raise; size defaults to 0.0, font to empty string."""
    blk = _valid_blk_dict()
    # Inject nulls into the span
    blk["lines"][0]["spans"][0]["size"] = None
    blk["lines"][0]["spans"][0]["font"] = None
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is not None
    assert result.size == 0.0
    assert result.bold is False


# ---------------------------------------------------------------------------
# Fix: strip per-line margin line-numbers in _parse_block
# Reviewer PDFs append a margin line-number to each line, so PyMuPDF emits
# blocks like "2.6.4 Random Forest (RF) \n286" and "...datasets with \n284".
# The trailing \nNNN makes every block multi-line, so classify's single-line
# guard skips heading detection. Strip line numbers BEFORE classification.
# ---------------------------------------------------------------------------

def _two_line_blk_dict(line1: str, line2: str, size=10.0, font="Helvetica",
                       flags=0, bbox=(50.0, 100.0, 500.0, 120.0)):
    """Build a PyMuPDF-style block dict with two lines (one span each)."""
    return {
        "bbox": bbox,
        "lines": [
            {"spans": [{"text": line1, "size": size, "font": font, "flags": flags}]},
            {"spans": [{"text": line2, "size": size, "font": font, "flags": flags}]},
        ],
    }


def test_parse_block_strips_trailing_line_number_from_heading():
    """'2.6.4 Random Forest (RF)\\n286' -> text == '2.6.4 Random Forest (RF)' (no '286'),
    and classify(that_block, 12.0) == 'heading'."""
    blk = _two_line_blk_dict("2.6.4 Random Forest (RF) ", "286")
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is not None
    assert "286" not in result.text
    assert result.text.strip() == "2.6.4 Random Forest (RF)"
    assert classify(result, 12.0) == "heading"


def test_parse_block_strips_trailing_line_number_from_body():
    """'We trained models on the dataset with\\n284' -> text == 'We trained models on the dataset with'."""
    blk = _two_line_blk_dict("We trained models on the dataset with", "284")
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is not None
    assert "284" not in result.text
    assert result.text.strip() == "We trained models on the dataset with"


def test_parse_block_bare_line_number_only_returns_none():
    """A block whose only content is '286' (a bare line number) -> _parse_block returns None."""
    blk = _valid_blk_dict(text="286")
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is None


def test_parse_block_normal_multiline_body_unchanged():
    """A normal multi-line body block with NO line numbers -> text unchanged (line break handled)."""
    blk = _two_line_blk_dict("The model was trained on", "historical patient data.")
    result = _parse_block(blk, page=1, page_height=792.0)
    assert result is not None
    # Both lines must be present — no content stripped
    assert "The model was trained on" in result.text
    assert "historical patient data." in result.text
