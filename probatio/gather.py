import re
from probatio.interfaces import IndexHandle
from probatio.models import EvidenceContext, SubQuestion

_PAGE = re.compile(r"pages?\s+(\d+)")


def _first_page(name: str) -> int | None:
    m = _PAGE.search(name or "")
    return int(m.group(1)) if m else None


class PaperQAGatherer:
    """Gathers evidence per sub-question, maps contexts -> EvidenceContext.

    Uses aget_evidence (retrieval + ranking), NOT aquery — we only want the
    evidence contexts, never paper-qa's generated answer prose, so we skip the
    answer-LLM call entirely. With Settings.skip_paperqa_summary this becomes a
    fully LLM-free, local-embedding-only retrieval pass.
    """
    async def gather(self, index: IndexHandle, subq: SubQuestion) -> list[EvidenceContext]:
        session = await index.docs.aget_evidence(subq.text, settings=index.settings)
        out: list[EvidenceContext] = []
        for i, c in enumerate(getattr(session, "contexts", [])):
            text_obj = c.text
            paper_id = getattr(getattr(text_obj, "doc", None), "dockey", None) \
                or (text_obj.name or "").split(" ")[0]
            page = _first_page(getattr(text_obj, "name", ""))
            out.append(EvidenceContext(
                # include subq.id so ids stay unique across sub-questions and don't
                # clobber each other when the pipeline flattens them into one dict.
                id=f"{paper_id}::p{page if page is not None else 'NA'}::{subq.id}::{i}",
                paper_id=paper_id,
                snippet=text_obj.text,
                page=page,
                score=float(getattr(c, "score", 0) or 0),
                rcs_summary=getattr(c, "context", "") or "",
            ))
        return out
