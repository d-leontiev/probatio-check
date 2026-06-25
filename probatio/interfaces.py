from pathlib import Path
from typing import Protocol, runtime_checkable, Any
from pydantic import BaseModel
from probatio.models import (
    EvidenceContext, SubQuestion,
    Citation, Reference, CitationCheck, CitationVerdict, OALookup,
)


class HighlightRect(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class IndexHandle(BaseModel):
    """Opaque handle returned by Indexer, consumed by Gatherer.

    Holds whatever the evidence track needs (paper-qa Docs + Settings).
    model_config allows arbitrary (non-pydantic) attribute types.
    """
    model_config = {"arbitrary_types_allowed": True}
    docs: Any            # paper_qa.Docs
    settings: Any        # paper_qa.Settings
    pdf_dir: Path


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, *, system: str, user: str, model: str | None = None,
                       think: bool = True) -> str: ...


@runtime_checkable
class Gatherer(Protocol):
    async def gather(self, index: IndexHandle, subq: SubQuestion) -> list[EvidenceContext]: ...


@runtime_checkable
class Highlighter(Protocol):
    def locate(
        self, pdf_path: Path, snippet: str, page: int | None = None
    ) -> list[HighlightRect]: ...


# --- "check" mode protocols (citation verification) --------------------------
@runtime_checkable
class ManuscriptParser(Protocol):
    """Parse a manuscript PDF into its in-text citations + bibliography."""
    async def parse(
        self, manuscript_pdf: Path
    ) -> tuple[list[Citation], list[Reference]]: ...


@runtime_checkable
class CitationResolver(Protocol):
    """Match each (citation, ref_key) to its source PDF in the folder."""
    async def resolve(
        self, citations: list[Citation], references: list[Reference], refs_dir: Path
    ) -> list[CitationCheck]: ...


@runtime_checkable
class CitationVerifier(Protocol):
    """Independent judge: sees only (claim, passages), returns a graded verdict."""
    async def judge(
        self, claim: str, passages: list[EvidenceContext]
    ) -> tuple[CitationVerdict, str, float]: ...


@runtime_checkable
class OAClient(Protocol):
    """Locate an open-access PDF for a reference by DOI (preferred) or title."""
    async def locate(self, *, doi: str | None, title: str | None) -> OALookup: ...
