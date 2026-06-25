from pathlib import Path
from typing import Awaitable, Callable, Optional
from probatio.config import Settings, paperqa_settings
from probatio.gather import PaperQAGatherer
from probatio.interfaces import Gatherer, IndexHandle
from probatio.models import EvidenceContext, SubQuestion

BuildIndex = Callable[[Path], Awaitable[IndexHandle]]


class RefRetriever:
    """Verbatim retrieval scoped to a single cited PDF.

    Each source PDF is embedded once into its OWN single-document index (cached by path, so a
    PDF cited many times is embedded once) and retrieval runs only within it — making this a
    true per-citation check rather than "supported by some reference". RCS is off
    (`evidence_skip_summary=True`) so the returned snippets are the verbatim source text the
    judge will see and the reviewer will audit.
    """

    def __init__(self, settings: Settings, *,
                 build_index: Optional[BuildIndex] = None,
                 gatherer: Optional[Gatherer] = None):
        self.settings = settings
        self._pq = paperqa_settings(settings)
        self._pq.answer.evidence_skip_summary = True   # verbatim snippets, no RCS pass
        self._build: BuildIndex = build_index or self._default_build
        self._gatherer: Gatherer = gatherer or PaperQAGatherer()
        self._cache: dict[Path, IndexHandle] = {}

    async def _default_build(self, pdf_path: Path) -> IndexHandle:
        from paperqa import Docs
        docs = Docs()
        # citation/docname/dockey = the pdf stem -> no LLM citation inference, and
        # EvidenceContext.paper_id == the source PDF stem (used for highlighting).
        await docs.aadd(pdf_path, citation=pdf_path.stem, docname=pdf_path.stem,
                        dockey=pdf_path.stem, settings=self._pq)
        return IndexHandle(docs=docs, settings=self._pq, pdf_dir=pdf_path.parent)

    async def index(self, pdf_path: Path) -> IndexHandle:
        if pdf_path not in self._cache:
            self._cache[pdf_path] = await self._build(pdf_path)
        return self._cache[pdf_path]

    async def passages_for(self, pdf_path: Path, claim: str, k: int = 3) -> list[EvidenceContext]:
        idx = await self.index(pdf_path)
        ctxs = await self._gatherer.gather(idx, SubQuestion(id="check", text=claim, order=0))
        return ctxs[:k]
