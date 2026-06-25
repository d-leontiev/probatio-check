from collections import Counter
from pathlib import Path
from probatio.check_retrieval import RefRetriever
from probatio.interfaces import ManuscriptParser, CitationResolver, CitationVerifier
from probatio.models import CitationCheck, CitationReport


def _bucket(c: CitationCheck) -> str:
    """A check's final status: its verdict once judged, else its resolution outcome."""
    return c.verdict if c.verdict != "unchecked" else c.resolution


async def check_pipeline(
    *,
    manuscript: Path,
    refs_dir: Path,
    parser: ManuscriptParser,
    resolver: CitationResolver,
    retriever: RefRetriever,
    verifier: CitationVerifier,
    k: int = 3,
) -> CitationReport:
    """Parse -> resolve -> (retrieve verbatim -> judge) each resolved empirical citation.

    Non-checkable citations are listed but never judged; resolution failures keep their bucket;
    nothing is silently dropped — every check ends in exactly one coverage bucket.
    """
    citations, references = await parser.parse(manuscript)
    checks = await resolver.resolve(citations, references, refs_dir)
    for chk in checks:
        if chk.citation.kind == "non_checkable":
            chk.verdict = "not_a_claim"            # transparency, not judged
            continue
        if chk.resolution != "resolved" or chk.source_pdf is None:
            continue                               # bucket stays its resolution status
        try:
            passages = await retriever.passages_for(chk.source_pdf, chk.citation.claim, k=k)
        except Exception:  # noqa: BLE001 - an unreadable/corrupt source must not abort the run
            chk.resolution = "unreadable_source"
            continue
        if not passages:
            chk.verdict = "not_found"
            continue
        verdict, rationale, confidence = await verifier.judge(chk.citation.claim, passages)
        chk.passages = passages
        chk.verdict, chk.rationale, chk.confidence = verdict, rationale, confidence
    coverage = dict(Counter(_bucket(c) for c in checks))
    return CitationReport(manuscript=str(manuscript), checks=checks, coverage=coverage)
