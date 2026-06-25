import argparse
import asyncio
import os
from pathlib import Path
from probatio.config import Settings, assert_local_only, make_verify_client
from probatio.report import write_acquisition_manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="probatio-acquire",
        description="Fetch the open-access PDFs a manuscript cites. Sends only reference "
                    "DOIs/titles to Unpaywall/OpenAlex; the manuscript body never leaves the machine.")
    ap.add_argument("--manuscript", required=True, type=Path, help="the manuscript PDF under review")
    ap.add_argument("--refs", required=True, type=Path, help="folder to populate with reference PDFs")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for acquisition.json/.md (default: the refs folder)")
    ap.add_argument("--email", default=None, help="contact email for Unpaywall/OpenAlex")
    ap.add_argument("--max-concurrency", type=int, default=4)
    args = ap.parse_args(argv)

    s = Settings()
    assert_local_only(s)                       # the bibliography is parsed only by the local LLM
    if s.ollama_api_base:
        os.environ["OLLAMA_API_BASE"] = s.ollama_api_base

    from probatio.manuscript import PyMuPDFManuscriptParser
    from probatio.acquire import UnpaywallOpenAlexClient, acquire_open_access

    async def _run():
        parser = PyMuPDFManuscriptParser(make_verify_client(s))
        refs = await parser.parse_references(args.manuscript)
        client = UnpaywallOpenAlexClient(email=args.email or s.unpaywall_email)
        report = await acquire_open_access(
            refs, args.refs, client=client, max_concurrency=args.max_concurrency)
        report.manuscript = str(args.manuscript)
        return refs, report

    refs, report = asyncio.run(_run())
    out = args.out or args.refs
    out.mkdir(parents=True, exist_ok=True)
    json_path, _md = write_acquisition_manifest(report, out)
    print(f"Acquired {len(refs)} references -> {json_path}")
    print("Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(report.summary.items())))
    missing = sum(report.summary.get(k, 0) for k in ("paywalled", "not_found", "error"))
    if missing:
        print(f"Drop the {missing} missing PDF(s) into {args.refs} (see acquisition.md)")


if __name__ == "__main__":
    main()
