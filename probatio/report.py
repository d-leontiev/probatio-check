import json
from pathlib import Path
from probatio.models import CitationReport, CitationCheck, AcquisitionReport

_STATUS_ORDER = {  # problems first
    "unsupported": 0, "overstated": 1, "not_found": 2, "partially": 3, "supported": 4,
    "ambiguous": 5, "no_pdf": 6, "unresolved_marker": 7, "unreadable_source": 8,
    "not_a_claim": 9, "unchecked": 10,
}


def _status(c: CitationCheck) -> str:
    """Displayed status: a human override wins, else the verdict, else the resolution bucket."""
    if c.human_override:
        return c.human_override
    return c.verdict if c.verdict != "unchecked" else c.resolution



def write_citation_sidecar(report: CitationReport, out_dir: Path) -> tuple[Path, Path]:
    """Write the citation-check receipts next to the run.

    citations.json — the full structured report (round-trips back into the audit UI).
    citations.md   — human-readable, problems first, ready to paste into a review.
    Returns (json_path, md_path).
    """
    json_path = out_dir / "citations.json"
    json_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))

    checks = sorted(report.checks, key=lambda c: _STATUS_ORDER.get(_status(c), 99))
    md = [f"# Citation check — {report.manuscript}", "",
          "Coverage: " + ", ".join(f"{k}={v}" for k, v in sorted(report.coverage.items())), ""]
    for c in checks:
        src = c.source_pdf.name if c.source_pdf else c.ref_key
        md.append(f"## [{_status(c)}] {c.citation.claim}")
        md.append(f"- cites **[{c.ref_key}]** → {src}")
        if c.rationale:
            md.append(f"- {c.rationale}")
        for p in c.passages[:1]:
            md.append(f"- source: “{p.snippet.strip()[:300]}”")
        if c.note:
            md.append(f"- note: {c.note}")
        md.append("")
    md_path = out_dir / "citations.md"
    md_path.write_text("\n".join(md))
    return json_path, md_path


_ACQ_ORDER = {"error": 0, "not_found": 1, "paywalled": 2, "fetched": 3, "already_present": 4}


def write_acquisition_manifest(report: AcquisitionReport, out_dir: Path) -> tuple[Path, Path]:
    """acquisition.json (machine) + acquisition.md (problems-first: what to drop in)."""
    json_path = out_dir / "acquisition.json"
    json_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    results = sorted(report.results, key=lambda r: _ACQ_ORDER.get(r.status, 99))
    md = [f"# Acquisition — {report.manuscript}", "",
          "Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(report.summary.items())), "",
          "## To supply manually (drop into the refs folder)", ""]
    for r in results:
        if r.status in ("error", "not_found", "paywalled"):
            label = r.title or r.doi or "(no metadata)"
            md.append(f"- [{r.status}] **{r.ref_key}** — {label}"
                      + (f" — {r.detail}" if r.detail else ""))
    md += ["", "## Fetched", ""]
    for r in results:
        if r.status in ("fetched", "already_present"):
            md.append(f"- [{r.status}] **{r.ref_key}** → {r.pdf_path}")
    md_path = out_dir / "acquisition.md"
    md_path.write_text("\n".join(md))
    return json_path, md_path
