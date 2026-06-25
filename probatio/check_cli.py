import argparse
import asyncio
import os
from pathlib import Path
from probatio.config import Settings, assert_local_only, make_verify_client
from probatio.check import check_pipeline
from probatio.report import write_citation_sidecar


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="probatio-check",
        description="Confidential, local citation checker: verify a manuscript's in-text "
                    "citations against the cited PDFs you supply. Nothing leaves the machine.")
    ap.add_argument("--manuscript", required=True, type=Path, help="the manuscript PDF under review")
    ap.add_argument("--refs", required=True, type=Path, help="folder of the cited reference PDFs")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for citations.json/.md (default: alongside the manuscript)")
    ap.add_argument("--serve", action="store_true", help="launch the citation audit UI when done")
    ap.add_argument("--port", type=int, default=8000, help="port for the audit UI (default: 8000)")
    args = ap.parse_args(argv)

    s = Settings()
    assert_local_only(s)                       # fail-closed: refuse any cloud/non-local component
    if s.ollama_api_base:
        os.environ["OLLAMA_API_BASE"] = s.ollama_api_base

    from probatio.manuscript import PyMuPDFManuscriptParser
    from probatio.resolve import CitationResolver
    from probatio.check_retrieval import RefRetriever
    from probatio.verify_citations import LLMCitationVerifier

    llm = make_verify_client(s)                # local gemma-4: parses the bibliography + judges
    report = asyncio.run(check_pipeline(
        manuscript=args.manuscript, refs_dir=args.refs,
        parser=PyMuPDFManuscriptParser(llm), resolver=CitationResolver(),
        retriever=RefRetriever(s), verifier=LLMCitationVerifier(llm),
        k=s.check_passages))

    out = args.out or args.manuscript.parent
    out.mkdir(parents=True, exist_ok=True)
    json_path, _md = write_citation_sidecar(report, out)
    print(f"Checked {len(report.checks)} citations -> {json_path}")
    print("Coverage: " + ", ".join(f"{k}={v}" for k, v in sorted(report.coverage.items())))

    if args.serve:
        import uvicorn
        from probatio.web.app import create_check_app
        print(f"Citation audit UI -> http://127.0.0.1:{args.port}")
        uvicorn.run(create_check_app(report=report, refs_dir=args.refs, out_dir=out),
                    host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
