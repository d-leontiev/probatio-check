import re
from pathlib import Path
import probatio.web as web

_STATIC = Path(web.__file__).parent / "static"
# An off-machine *fetchable* URL: http(s) not pointing at loopback. XML-namespace
# identifiers (www.w3.org) are never fetched, so they are allowlisted.
_OFFSITE = re.compile(r"https?://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|www\.w3\.org)", re.I)


def test_served_assets_have_no_offsite_urls():
    for name in ("index.html", "app.css", "app.js"):
        text = (_STATIC / name).read_text()
        assert not _OFFSITE.findall(text), f"{name} references an off-machine URL"
