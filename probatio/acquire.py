import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import quote

from probatio.interfaces import OAClient
from probatio.models import (
    AcquisitionReport,
    AcquisitionResult,
    OALocation,
    OALookup,
    Reference,
)
from probatio.resolve import doi_slug, norm_doi

_SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")


def safe_pdf_path(pdf_dir: Path, paper_id: str) -> Path:
    """Map a paper id to <pdf_dir>/<id>.pdf, rejecting ids that could escape pdf_dir."""
    if paper_id in (".", "..") or not _SAFE_ID.fullmatch(paper_id):
        raise ValueError(f"unsafe paper id for filename: {paper_id!r}")
    target = (pdf_dir / f"{paper_id}.pdf").resolve()
    if target.parent != pdf_dir.resolve():
        raise ValueError(f"paper id escapes pdf_dir: {paper_id!r}")
    return target


_PDF_MAGIC = b"%PDF"
Fetch = Callable[[str], Awaitable[bytes]]


async def _http_get_bytes(url: str, *, timeout: float = 30.0) -> bytes:
    import httpx
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


async def download_pdf(url: str, dest: Path, *, max_bytes: int = 50_000_000,
                       fetch: Fetch | None = None) -> bool:
    """Fetch a PDF to dest. True only on a valid %PDF body within max_bytes; never raises
    (a bad/oversized/non-PDF download -> False + warning) so one failure cannot abort a batch."""
    log = logging.getLogger(__name__)
    log.info("downloading PDF from %s", url)   # audit: this GET reaches a third (publisher) host
    getter = fetch or _http_get_bytes
    try:
        data = await getter(url)
    except Exception as e:  # noqa: BLE001 - a failed download is a non-fatal per-ref outcome
        log.warning("download failed for %s: %s", url, e)
        return False
    if len(data) > max_bytes:
        log.warning("download exceeds %d bytes, rejecting: %s", max_bytes, url)
        return False
    if not data.startswith(_PDF_MAGIC):
        log.warning("download is not a PDF (no %%PDF magic): %s", url)
        return False
    dest.write_bytes(data)
    return True


def _is_pdf_file(path: Path) -> bool:
    """True if the file exists and begins with the %PDF magic (so an already-present file is
    only skipped when it is a real PDF, matching the rigor of a fresh download)."""
    try:
        with path.open("rb") as f:
            return f.read(4) == _PDF_MAGIC
    except OSError:
        return False


DownloadFn = Callable[[str, Path], Awaitable[bool]]


def _doi_filename(doi: str | None, key: str) -> str:
    """Unique, resolver-matchable PDF filename for a reference.

    With a DOI the name is `<doi_slug>__<key_slug>-<keyhash>.pdf`: the `doi_slug` prefix is the
    EXACT transform the resolver applies for its anchored DOI-in-filename match (so the file is
    found), and the `__…` suffix makes the name unique per reference. The suffix carries a short
    stable hash of the RAW key so two distinct keys that slugify identically (e.g. "O'Brien 2020"
    and "O Brien 2020") get different files instead of one clobbering the other — slug-equality is
    not key-equality, and `_dedupe` only collapses exact-equal keys. Without a DOI the key names
    the file."""
    kh = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    if doi:
        return f"{doi_slug(norm_doi(doi))}__{doi_slug(key)}-{kh}.pdf"   # norm_doi: strip any URL/doi: prefix
    return f"{doi_slug(key)}-{kh}.pdf"


async def acquire_open_access(
    references: list[Reference], refs_dir: Path, *, client: OAClient,
    download: DownloadFn | None = None, max_concurrency: int = 4,
) -> AcquisitionReport:
    """Locate + fetch the open-access PDF for each reference. Per-reference errors are
    isolated (one failure never aborts the batch). Returns a classified report."""
    refs_dir = Path(refs_dir)
    refs_dir.mkdir(parents=True, exist_ok=True)
    dl = download or download_pdf
    existing = {p.name.lower() for p in refs_dir.glob("*.pdf")}
    sem = asyncio.Semaphore(max_concurrency)
    log = logging.getLogger(__name__)

    async def one(ref: Reference) -> AcquisitionResult:
        base = AcquisitionResult(ref_key=ref.key, doi=ref.doi, title=ref.title, status="error")
        fname = _doi_filename(ref.doi, ref.key)
        dest = refs_dir / fname
        if fname.lower() in existing and _is_pdf_file(dest):
            return base.model_copy(update={"status": "already_present", "pdf_path": str(dest)})
        async with sem:
            try:
                lookup = await client.locate(doi=ref.doi, title=ref.title)
            except Exception as e:  # noqa: BLE001 - one lookup failure must not abort the batch
                log.warning("OA lookup failed for ref %s: %s", ref.key, e)
                return base.model_copy(update={"status": "error", "detail": f"locate: {e}"})
        if lookup.status == "unresolved":
            return base.model_copy(update={"status": "not_found"})
        if lookup.status == "closed":
            return base.model_copy(update={"status": "paywalled"})
        if lookup.location is None:   # oa but no usable URL: a client fault, not a paywall
            return base.model_copy(update={"status": "error",
                                           "detail": "oa lookup returned no location"})
        if await dl(lookup.location.pdf_url, dest):
            return base.model_copy(update={"status": "fetched", "pdf_path": str(dest),
                                           "source_url": lookup.location.pdf_url})
        return base.model_copy(update={"status": "error", "detail": "download failed",
                                       "source_url": lookup.location.pdf_url})

    results = list(await asyncio.gather(*(one(r) for r in references)))
    summary: dict[str, int] = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1
    return AcquisitionReport(results=results, summary=summary)


class UnpaywallOpenAlexClient:
    """OAClient: Unpaywall by DOI, OpenAlex by title fallback. Logs every outbound call.
    `transport` (httpx.MockTransport) is injectable for hermetic tests."""

    def __init__(self, *, email: str, timeout: float = 30.0,
                 transport: object | None = None):
        self.email = email
        self.timeout = timeout
        self._transport = transport

    def _client(self):  # type: ignore[no-untyped-def]
        import httpx
        kwargs: dict[str, object] = {"timeout": self.timeout, "follow_redirects": True}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    async def locate(self, *, doi: str | None, title: str | None) -> OALookup:
        async with self._client() as c:
            if doi:
                lk = await self._unpaywall(c, doi)
                if lk.status != "unresolved":
                    return lk
            if title:
                return await self._openalex(c, title)
            return OALookup(status="unresolved")

    async def _unpaywall(self, c, doi: str) -> OALookup:  # type: ignore[no-untyped-def]
        doi = norm_doi(doi)   # Unpaywall wants the bare DOI; a 'https://doi.org/…' prefix 404s/mis-resolves
        logging.getLogger(__name__).info("unpaywall lookup doi=%s", doi)
        r = await c.get(f"https://api.unpaywall.org/v2/{quote(doi, safe='/')}",
                        params={"email": self.email})
        if r.status_code == 404:
            return OALookup(status="unresolved")
        r.raise_for_status()
        try:
            j = r.json()
        except (json.JSONDecodeError, ValueError):   # a 200 with a non-JSON body (CDN/maintenance page)
            return OALookup(status="unresolved")
        if not j.get("is_oa"):
            return OALookup(status="closed")
        # Only a DIRECT PDF link is auto-fetchable; a bare landing-page `url` (common for
        # bronze/green OA) is not, so report it as closed (-> manual drop-in) rather than
        # handing the orchestrator an HTML page that download_pdf would reject as an "error".
        url = (j.get("best_oa_location") or {}).get("url_for_pdf")
        if not url:
            return OALookup(status="closed")
        return OALookup(status="oa", location=OALocation(
            pdf_url=url, source="unpaywall", doi=doi, oa_status=str(j.get("oa_status", ""))))

    async def _openalex(self, c, title: str) -> OALookup:  # type: ignore[no-untyped-def]
        logging.getLogger(__name__).info("openalex lookup title=%r", title[:120])
        r = await c.get("https://api.openalex.org/works",
                        params={"search": title[:300], "per_page": 1, "mailto": self.email})
        r.raise_for_status()
        try:
            payload = r.json()
        except (json.JSONDecodeError, ValueError):
            return OALookup(status="unresolved")
        results = payload.get("results") or []
        if not results:
            return OALookup(status="unresolved")
        w = results[0]
        oa = w.get("open_access") or {}
        if not oa.get("is_oa"):
            return OALookup(status="closed")
        # Prefer the direct-PDF field; accept oa_url only when it is itself a direct PDF link.
        url = (w.get("best_oa_location") or {}).get("pdf_url")
        if not url:
            cand = str(oa.get("oa_url") or "")
            url = cand if cand.lower().endswith(".pdf") else None
        if not url:
            return OALookup(status="closed")
        doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
        return OALookup(status="oa",
                        location=OALocation(pdf_url=url, source="openalex", doi=doi))
