import argparse
from pathlib import Path
from probatio.models import CitationReport


def load_check_report(run_dir: Path) -> CitationReport:
    """Reconstruct a CitationReport from a saved citations.json — no re-run."""
    return CitationReport.model_validate_json((Path(run_dir) / "citations.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="probatio-check-ui",
        description="Serve the citation-audit UI for a finished check run (no re-run, no LLM).")
    ap.add_argument("--run", required=True, type=Path,
                    help="run dir containing citations.json")
    ap.add_argument("--refs", type=Path, default=None,
                    help="folder of the cited reference PDFs (default: <run>/pdfs)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn
    from probatio.web.app import create_check_app
    report = load_check_report(args.run)
    refs_dir = args.refs or (args.run / "pdfs")
    app = create_check_app(report=report, refs_dir=refs_dir, out_dir=args.run)
    print(f"probatio citation-check UI -> http://127.0.0.1:{args.port}  "
          f"({len(report.checks)} citations; refs: {refs_dir})")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
