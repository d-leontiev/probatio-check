from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel


class EvidenceContext(BaseModel):
    id: str                       # stable, e.g. "smith2020neural::p3::0"
    paper_id: str
    snippet: str                  # verbatim text from the PDF
    page: Optional[int] = None
    score: float = 0.0
    rcs_summary: str = ""          # PaperQA2 contextual summary


class SubQuestion(BaseModel):
    id: str
    text: str
    order: int
    rationale: str = ""


# --- "check" mode: citation verification against the cited PDFs ---------------
# Graded faithfulness verdict for one in-text citation vs its source passage.
CitationVerdict = Literal[
    "supported",      # source directly states the claim (every substantive part)
    "partially",      # central assertion substantively supported; secondary parts absent or only in part
    "overstated",     # claim generalizes or strengthens beyond the source's scope
    "unsupported",    # no substantive part supported, contradicts the claim, or off-topic/different entity
    "not_found",      # no relevant passage retrieved from the source
    "not_a_claim",    # non-checkable citation (method-of / see-review / credit)
    "unchecked",      # default, pre-verification
]
# Outcome of matching an in-text marker to a source PDF in the folder.
ResolutionStatus = Literal[
    "resolved", "no_pdf", "unresolved_marker", "ambiguous", "unreadable_source"
]


class Reference(BaseModel):
    """One parsed bibliography entry from the manuscript under review."""
    key: str                       # "12" (numeric) or "Smith2020" (author-year)
    raw: str
    title: Optional[str] = None
    doi: Optional[str] = None
    authors: list[str] = []
    year: Optional[int] = None


OAStatus = Literal["oa", "closed", "unresolved"]
AcquisitionStatus = Literal[
    "fetched", "paywalled", "not_found", "already_present", "error"]


class OALocation(BaseModel):
    pdf_url: str
    source: str = ""              # "unpaywall" | "openalex"
    doi: str | None = None
    oa_status: str = ""           # informational (e.g. gold/green/hybrid)


class OALookup(BaseModel):
    """Three outcomes so the caller distinguishes 'found but closed' from 'not found'."""
    status: OAStatus
    location: OALocation | None = None


class AcquisitionResult(BaseModel):
    ref_key: str
    doi: str | None = None
    title: str | None = None
    status: AcquisitionStatus
    pdf_path: str | None = None
    source_url: str | None = None
    detail: str = ""


class AcquisitionReport(BaseModel):
    manuscript: str = ""
    results: list[AcquisitionResult] = []
    summary: dict[str, int] = {}


class Citation(BaseModel):
    """An in-text citation: the claim sentence + the reference key(s) it cites."""
    id: str                        # "c0001"
    claim: str                     # the full sentence the marker backs
    ref_keys: list[str]            # markers expanded -> reference keys
    section: str = ""
    manuscript_page: Optional[int] = None
    kind: Literal["empirical", "non_checkable"] = "empirical"


class CitationCheck(BaseModel):
    """One (claim, cited-reference) pair and its verification outcome."""
    citation: Citation
    ref_key: str                   # the specific cited reference this check is for
    reference: Optional[Reference] = None
    source_pdf: Optional[Path] = None
    resolution: ResolutionStatus
    passages: list[EvidenceContext] = []   # verbatim, from the cited PDF
    verdict: CitationVerdict = "unchecked"
    rationale: str = ""
    confidence: float = 0.0
    human_override: Optional[CitationVerdict] = None
    reviewed: bool = False
    note: str = ""


class CitationReport(BaseModel):
    manuscript: str
    checks: list[CitationCheck]
    coverage: dict[str, int] = {}  # bucket -> count
