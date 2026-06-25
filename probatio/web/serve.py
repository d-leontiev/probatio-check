import argparse
from pathlib import Path
from probatio.models import CitationReport
from probatio.web.app import create_app


def load_check_report(run_dir: Path) -> CitationReport:
    """Reconstruct a CitationReport from a saved citations.json — no re-run."""
    return CitationReport.model_validate_json((Path(run_dir) / "citations.json").read_text())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="probatio-check-ui",
        description="Citation-check UI. With --run, serve a finished run; without, a launcher "
                    "to acquire references and run a check from the browser.")
    ap.add_argument("--run", type=Path, default=None, help="run dir with citations.json (audit-only)")
    ap.add_argument("--refs", type=Path, default=None, help="refs PDFs (default: <run>/pdfs)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    import uvicorn
    from probatio.web.app import create_check_app
    from probatio.config import Settings
    if args.run is not None:
        report = load_check_report(args.run)
        refs_dir = args.refs or (args.run / "pdfs")
        app = create_check_app(report=report, refs_dir=refs_dir, out_dir=args.run)
        print(f"probatio audit UI -> http://127.0.0.1:{args.port}  ({len(report.checks)} citations)")
    else:
        app = create_app(Settings())
        print(f"probatio launcher -> http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
