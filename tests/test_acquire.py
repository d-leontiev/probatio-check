import json
from pathlib import Path

import httpx
import pytest

from probatio.models import OALocation, OALookup, AcquisitionResult, AcquisitionReport, Reference
from probatio.interfaces import OAClient
from probatio.acquire import (
    UnpaywallOpenAlexClient,
    acquire_open_access,
    download_pdf,
    _doi_filename,
)
from probatio.report import write_acquisition_manifest


def test_oalookup_and_result_models():
    lk = OALookup(status="oa", location=OALocation(pdf_url="http://x/a.pdf", source="unpaywall"))
    assert lk.status == "oa" and lk.location.pdf_url.endswith("a.pdf")
    assert OALookup(status="unresolved").location is None
    r = AcquisitionResult(ref_key="1", status="fetched", pdf_path="/tmp/a.pdf")
    rep = AcquisitionReport(manuscript="m.pdf", results=[r], summary={"fetched": 1})
    assert rep.results[0].status == "fetched" and rep.summary["fetched"] == 1


def test_fake_oaclient_satisfies_protocol():
    class _Fake:
        async def locate(self, *, doi, title):
            return OALookup(status="unresolved")
    assert isinstance(_Fake(), OAClient)


def test_safe_pdf_path_rejects_traversal(tmp_path):
    from probatio.acquire import safe_pdf_path
    assert safe_pdf_path(tmp_path, "smith2020").parent == tmp_path.resolve()
    for bad in ("../evil", "a/b", "..", ".", "/etc/passwd"):
        with pytest.raises(ValueError):
            safe_pdf_path(tmp_path, bad)


async def _aret(b: bytes) -> bytes:
    return b


@pytest.mark.asyncio
async def test_download_pdf_accepts_pdf(tmp_path):
    dest = tmp_path / "a.pdf"
    assert await download_pdf("http://x/a.pdf", dest, fetch=lambda u: _aret(b"%PDF-1.7 body"))
    assert dest.read_bytes().startswith(b"%PDF")


@pytest.mark.asyncio
async def test_download_pdf_rejects_html(tmp_path):
    dest = tmp_path / "a.pdf"
    ok = await download_pdf("http://x/a", dest, fetch=lambda u: _aret(b"<html>no</html>"))
    assert ok is False and not dest.exists()


@pytest.mark.asyncio
async def test_download_pdf_rejects_oversized(tmp_path):
    dest = tmp_path / "a.pdf"
    ok = await download_pdf("http://x/a.pdf", dest, max_bytes=8, fetch=lambda u: _aret(b"%PDFxxxxxxxx"))
    assert ok is False and not dest.exists()


@pytest.mark.asyncio
async def test_download_pdf_handles_fetch_error(tmp_path):
    async def boom(u):
        raise RuntimeError("net down")
    assert await download_pdf("http://x", tmp_path / "a.pdf", fetch=boom) is False


class _FakeOAClient:
    def __init__(self, by_id: dict):
        self.by_id, self.calls = by_id, []

    async def locate(self, *, doi, title):
        self.calls.append((doi, title))
        return self.by_id.get(doi or title, OALookup(status="unresolved"))


async def _ok_dl(url, dest):
    Path(dest).write_bytes(b"%PDF-1.7")
    return True


async def _fail_dl(url, dest):
    return False


def _ref(key, doi=None, title=None):
    return Reference(key=key, raw=key, doi=doi, title=title)


@pytest.mark.asyncio
async def test_acquire_classifies_each_status(tmp_path):
    client = _FakeOAClient({
        "10.1/oa": OALookup(status="oa", location=OALocation(pdf_url="http://x/oa.pdf")),
        "10.1/closed": OALookup(status="closed"),
    })
    refs = [_ref("1", "10.1/oa"), _ref("2", "10.1/closed"), _ref("3", "10.1/missing"),
            _ref("4")]  # no doi/title -> unresolved
    report = await acquire_open_access(refs, tmp_path, client=client, download=_ok_dl)
    by_key = {r.ref_key: r.status for r in report.results}
    assert by_key == {"1": "fetched", "2": "paywalled", "3": "not_found", "4": "not_found"}
    assert (tmp_path / _doi_filename("10.1/oa", "1")).exists()
    assert report.summary["fetched"] == 1


@pytest.mark.asyncio
async def test_acquire_skips_already_present(tmp_path):
    (tmp_path / _doi_filename("10.1/oa", "1")).write_bytes(b"%PDF old")
    client = _FakeOAClient({"10.1/oa": OALookup(status="oa",
                                                location=OALocation(pdf_url="http://x/oa.pdf"))})
    report = await acquire_open_access([_ref("1", "10.1/oa")], tmp_path,
                                       client=client, download=_ok_dl)
    assert report.results[0].status == "already_present"
    assert client.calls == []                      # no network for an already-present ref


@pytest.mark.asyncio
async def test_acquire_download_failure_is_error_not_abort(tmp_path):
    client = _FakeOAClient({
        "10.1/a": OALookup(status="oa", location=OALocation(pdf_url="http://x/a.pdf")),
        "10.1/b": OALookup(status="oa", location=OALocation(pdf_url="http://x/b.pdf")),
    })
    report = await acquire_open_access([_ref("1", "10.1/a"), _ref("2", "10.1/b")],
                                       tmp_path, client=client, download=_fail_dl)
    assert [r.status for r in report.results] == ["error", "error"]


@pytest.mark.asyncio
async def test_acquire_isolates_locate_exception(tmp_path):
    class _Boom:
        async def locate(self, *, doi, title):
            if doi == "10.1/boom":
                raise RuntimeError("api down")
            return OALookup(status="oa", location=OALocation(pdf_url="http://x/a.pdf"))
    report = await acquire_open_access([_ref("1", "10.1/boom"), _ref("2", "10.1/ok")],
                                       tmp_path, client=_Boom(), download=_ok_dl)
    by_key = {r.ref_key: r.status for r in report.results}
    assert by_key == {"1": "error", "2": "fetched"}


def _transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_unpaywall_oa_hit():
    def handler(req):
        assert "api.unpaywall.org" in str(req.url) and "email=" in str(req.url)
        return httpx.Response(200, json={"is_oa": True, "oa_status": "gold",
                                         "best_oa_location": {"url_for_pdf": "http://x/a.pdf"}})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    lk = await client.locate(doi="10.1/x", title=None)
    assert lk.status == "oa" and lk.location.pdf_url == "http://x/a.pdf" and lk.location.source == "unpaywall"


@pytest.mark.asyncio
async def test_unpaywall_closed():
    def handler(req):
        return httpx.Response(200, json={"is_oa": False, "best_oa_location": None})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    assert (await client.locate(doi="10.1/x", title=None)).status == "closed"


@pytest.mark.asyncio
async def test_unpaywall_404_then_no_title_unresolved():
    def handler(req):
        return httpx.Response(404, text="<!doctype html><title>404 Not Found</title>")  # live shape
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    assert (await client.locate(doi="10.1/missing", title=None)).status == "unresolved"


@pytest.mark.asyncio
async def test_openalex_title_fallback_oa():
    def handler(req):
        assert "api.openalex.org" in str(req.url)
        return httpx.Response(200, json={"results": [
            {"doi": "https://doi.org/10.1/y",
             "open_access": {"is_oa": True, "oa_url": "http://x/y.pdf"}}]})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    lk = await client.locate(doi=None, title="Some paper title")
    assert lk.status == "oa" and lk.location.pdf_url == "http://x/y.pdf"
    assert lk.location.source == "openalex" and lk.location.doi == "10.1/y"


@pytest.mark.asyncio
async def test_openalex_no_results_unresolved():
    def handler(req):
        return httpx.Response(200, json={"results": []})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    assert (await client.locate(doi=None, title="nope")).status == "unresolved"


def test_write_acquisition_manifest(tmp_path):
    report = AcquisitionReport(manuscript="m.pdf", summary={"fetched": 1, "paywalled": 1},
        results=[AcquisitionResult(ref_key="1", status="fetched", pdf_path="/r/1.pdf"),
                 AcquisitionResult(ref_key="2", status="paywalled", title="Closed Paper")])
    json_path, md_path = write_acquisition_manifest(report, tmp_path)
    data = json.loads(json_path.read_text())
    assert data["summary"]["fetched"] == 1 and len(data["results"]) == 2
    md = md_path.read_text()
    # problems-first: the paywalled (to-supply) entry appears before the fetched one
    assert md.index("Closed Paper") < md.index("/r/1.pdf")


def test_acquire_cli_writes_manifest(tmp_path, monkeypatch):
    import probatio.acquire_cli as cli
    import probatio.manuscript as manuscript
    import probatio.acquire as acquire

    # Stub the heavy/online bits so the CLI test is hermetic.
    async def fake_parse_references(self, pdf):
        return [Reference(key="1", raw="1", doi="10.1/x", title="T")]
    monkeypatch.setattr(manuscript.PyMuPDFManuscriptParser, "parse_references",
                        fake_parse_references)

    async def fake_acquire(refs, refs_dir, *, client, max_concurrency=4):
        return AcquisitionReport(results=[AcquisitionResult(ref_key="1", status="paywalled")],
                                 summary={"paywalled": 1})
    monkeypatch.setattr(acquire, "acquire_open_access", fake_acquire)
    monkeypatch.setattr(acquire, "UnpaywallOpenAlexClient", lambda **k: object())

    pdf = tmp_path / "m.pdf"
    pdf.write_bytes(b"%PDF-1.7")          # parse_references is stubbed, content is irrelevant
    refs_dir = tmp_path / "refs"
    cli.main(["--manuscript", str(pdf), "--refs", str(refs_dir)])
    assert (refs_dir / "acquisition.json").exists()


# --- regressions from the Phase-2 adversarial review --------------------------

async def _content_dl(url, dest):
    Path(dest).write_bytes(b"%PDF-" + url.encode())   # distinct content per URL
    return True


@pytest.mark.asyncio
async def test_acquired_pdf_is_resolved_by_resolver(tmp_path):
    """The keystone contract: a PDF NAMED by acquire must be matched by the resolver via the
    filename (when the PDF carries no extractable embedded DOI). Catches the slash-slug break."""
    from probatio.resolve import CitationResolver
    from probatio.models import Citation
    doi = "10.1038/s41586-020-2649-2"
    (tmp_path / _doi_filename(doi, "1")).write_bytes(b"%PDF-1.7 acquired")
    resolver = CitationResolver(extract_meta=lambda p: (None, None))  # no DOI inside the PDF
    cit = Citation(id="c0001", claim="A claim.", ref_keys=["1"])
    checks = await resolver.resolve([cit], [Reference(key="1", raw="1", doi=doi)], tmp_path)
    assert checks[0].resolution == "resolved" and checks[0].source_pdf is not None


@pytest.mark.asyncio
async def test_acquire_distinct_dois_same_slug_no_clobber(tmp_path):
    # '10.1/a:b' and '10.1/a/b' slugify identically -> must NOT overwrite each other.
    client = _FakeOAClient({
        "10.1/a:b": OALookup(status="oa", location=OALocation(pdf_url="http://x/colon.pdf")),
        "10.1/a/b": OALookup(status="oa", location=OALocation(pdf_url="http://x/slash.pdf")),
    })
    report = await acquire_open_access([_ref("1", "10.1/a:b"), _ref("2", "10.1/a/b")],
                                      tmp_path, client=client, download=_content_dl)
    assert [r.status for r in report.results] == ["fetched", "fetched"]
    paths = {r.ref_key: Path(r.pdf_path) for r in report.results}
    assert paths["1"] != paths["2"]                              # distinct files
    assert paths["1"].read_bytes() == b"%PDF-http://x/colon.pdf"  # correct, un-clobbered content
    assert paths["2"].read_bytes() == b"%PDF-http://x/slash.pdf"


@pytest.mark.asyncio
async def test_resolver_anchored_slug_not_substring(tmp_path):
    """A citation to 10.1234/abc must NOT resolve to a sibling file for 10.1234/abc.123 whose
    slug merely CONTAINS the shorter DOI's slug (anchored '<slug>__' match, no bare substring)."""
    from probatio.resolve import CitationResolver
    from probatio.models import Citation
    base, sibling = "10.1234/abc", "10.1234/abc.123"
    (tmp_path / _doi_filename(sibling, "5")).write_bytes(b"%PDF sibling")   # visited first (sorts first)
    (tmp_path / _doi_filename(base, "1")).write_bytes(b"%PDF base")
    resolver = CitationResolver(extract_meta=lambda p: (None, None))        # match must come from filename
    cit = Citation(id="c0001", claim="A claim.", ref_keys=["1"])
    checks = await resolver.resolve([cit], [Reference(key="1", raw="1", doi=base)], tmp_path)
    assert checks[0].resolution == "resolved"
    assert checks[0].source_pdf.name == _doi_filename(base, "1")            # the CORRECT file


def test_doi_filename_normalizes_prefixed_doi():
    # Real parsed DOIs carry a 'https://doi.org/' prefix; the filename must use the BARE DOI
    # slug so it matches the resolver (which normalizes ref.doi before slugging).
    n = _doi_filename("https://doi.org/10.1146/Annurev-X", "Smith 2020")
    assert n.startswith("10.1146_annurev-x__")
    assert "https" not in n and "doi.org" not in n


@pytest.mark.asyncio
async def test_acquired_prefixed_doi_resolves(tmp_path):
    from probatio.resolve import CitationResolver
    from probatio.models import Citation
    pdoi = "https://doi.org/10.1016/j.energy.2014.07.044"
    (tmp_path / _doi_filename(pdoi, "1")).write_bytes(b"%PDF-1.7 x")
    resolver = CitationResolver(extract_meta=lambda p: (None, None))
    cit = Citation(id="c0001", claim="A.", ref_keys=["1"])
    checks = await resolver.resolve([cit], [Reference(key="1", raw="1", doi=pdoi)], tmp_path)
    assert checks[0].resolution == "resolved"


@pytest.mark.asyncio
async def test_unpaywall_strips_doi_prefix():
    seen = {}
    def handler(req):
        seen["path"] = req.url.path
        return httpx.Response(200, json={"is_oa": False})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    await client.locate(doi="https://doi.org/10.1/abc", title=None)
    assert "doi.org" not in seen["path"] and seen["path"].endswith("/v2/10.1/abc")


@pytest.mark.asyncio
async def test_acquire_distinct_keys_same_slug_no_clobber(tmp_path):
    # Same DOI-slug AND same key-slug but DISTINCT raw keys must still not clobber.
    client = _FakeOAClient({
        "10.1/a:b": OALookup(status="oa", location=OALocation(pdf_url="http://x/colon.pdf")),
        "10.1/a/b": OALookup(status="oa", location=OALocation(pdf_url="http://x/slash.pdf")),
    })
    refs = [_ref("O'Brien 2020", "10.1/a:b"), _ref("O Brien 2020", "10.1/a/b")]
    report = await acquire_open_access(refs, tmp_path, client=client, download=_content_dl)
    assert [r.status for r in report.results] == ["fetched", "fetched"]
    paths = [Path(r.pdf_path) for r in report.results]
    assert paths[0] != paths[1]                                  # distinct despite identical slugs
    assert paths[0].read_bytes() == b"%PDF-http://x/colon.pdf"   # un-clobbered content
    assert paths[1].read_bytes() == b"%PDF-http://x/slash.pdf"


@pytest.mark.asyncio
async def test_acquire_oa_without_location_is_error(tmp_path):
    client = _FakeOAClient({"10.1/x": OALookup(status="oa", location=None)})
    report = await acquire_open_access([_ref("1", "10.1/x")], tmp_path, client=client, download=_ok_dl)
    assert report.results[0].status == "error"   # oa-but-no-URL is a fault, not 'paywalled'


@pytest.mark.asyncio
async def test_acquire_already_present_must_be_valid_pdf(tmp_path):
    # A present-but-non-PDF file must NOT count as already_present (re-fetch instead).
    (tmp_path / _doi_filename("10.1/oa", "1")).write_bytes(b"<html>not a pdf</html>")
    client = _FakeOAClient({"10.1/oa": OALookup(status="oa",
                                                location=OALocation(pdf_url="http://x/oa.pdf"))})
    report = await acquire_open_access([_ref("1", "10.1/oa")], tmp_path,
                                       client=client, download=_ok_dl)
    assert report.results[0].status == "fetched"   # not skipped as already_present


@pytest.mark.asyncio
async def test_unpaywall_oa_landing_only_is_closed():
    def handler(req):
        return httpx.Response(200, json={"is_oa": True, "oa_status": "bronze",
            "best_oa_location": {"url_for_pdf": None, "url": "http://x/landing"}})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    assert (await client.locate(doi="10.1/x", title=None)).status == "closed"


@pytest.mark.asyncio
async def test_openalex_prefers_direct_pdf_over_landing():
    def handler(req):
        return httpx.Response(200, json={"results": [{
            "doi": "https://doi.org/10.1/y",
            "open_access": {"is_oa": True, "oa_url": "http://x/landing"},
            "best_oa_location": {"pdf_url": "http://x/direct.pdf"}}]})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    lk = await client.locate(doi=None, title="t")
    assert lk.status == "oa" and lk.location.pdf_url == "http://x/direct.pdf"


@pytest.mark.asyncio
async def test_openalex_landing_only_is_closed():
    def handler(req):
        return httpx.Response(200, json={"results": [{
            "doi": "https://doi.org/10.1/y",
            "open_access": {"is_oa": True, "oa_url": "http://x/landing"},  # not a .pdf
            "best_oa_location": {"pdf_url": None}}]})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    assert (await client.locate(doi=None, title="t")).status == "closed"


@pytest.mark.asyncio
async def test_unpaywall_non_json_200_unresolved():
    def handler(req):
        return httpx.Response(200, text="<html>maintenance</html>")
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    assert (await client.locate(doi="10.1/x", title=None)).status == "unresolved"


@pytest.mark.asyncio
async def test_openalex_non_json_200_unresolved():
    def handler(req):
        return httpx.Response(200, text="not json at all")
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    assert (await client.locate(doi=None, title="t")).status == "unresolved"


@pytest.mark.asyncio
async def test_openalex_caps_long_title():
    seen = {}
    def handler(req):
        seen["search"] = req.url.params.get("search")
        return httpx.Response(200, json={"results": []})
    client = UnpaywallOpenAlexClient(email="e@x.org", transport=_transport(handler))
    await client.locate(doi=None, title="x" * 5000)
    assert seen["search"] is not None and len(seen["search"]) <= 300


@pytest.mark.asyncio
async def test_download_pdf_logs_url(tmp_path, caplog):
    import logging as _logging
    with caplog.at_level(_logging.INFO, logger="probatio.acquire"):
        await download_pdf("http://cdn.example/a.pdf", tmp_path / "a.pdf",
                           fetch=lambda u: _aret(b"%PDF-1.7"))
    assert any("cdn.example" in r.message for r in caplog.records)   # third-host egress audited
