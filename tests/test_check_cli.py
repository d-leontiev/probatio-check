import pytest
from probatio.models import CitationReport
from probatio.config import ConfidentialityError


def test_main_runs_pipeline_and_writes_sidecar(tmp_path, monkeypatch):
    calls = {}

    async def fake_pipeline(**kw):
        calls.update(kw)
        return CitationReport(manuscript=str(kw["manuscript"]), checks=[], coverage={})

    monkeypatch.setattr("probatio.check_cli.assert_local_only", lambda s: None)
    monkeypatch.setattr("probatio.check_cli.check_pipeline", fake_pipeline)
    (tmp_path / "m.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "refs").mkdir()

    from probatio.check_cli import main
    main(["--manuscript", str(tmp_path / "m.pdf"), "--refs", str(tmp_path / "refs"),
          "--out", str(tmp_path)])

    assert calls["manuscript"] == tmp_path / "m.pdf"
    assert calls["refs_dir"] == tmp_path / "refs"
    assert calls["k"] == 10                      # judge passage count comes from settings
    assert (tmp_path / "citations.json").exists()


def test_missing_manuscript_is_argparse_error():
    from probatio.check_cli import main
    with pytest.raises(SystemExit):
        main(["--refs", "/tmp"])


def test_confidentiality_guard_blocks_cloud_models(tmp_path, monkeypatch):
    # Defaults are local, but pointing check mode at a cloud model must still abort before any
    # work (fail-closed). Here verify_model is overridden to a cloud model via env.
    monkeypatch.setenv("PROBATIO_VERIFY_MODEL", "claude-haiku-4-5-20251001")
    (tmp_path / "m.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "refs").mkdir()
    from probatio.check_cli import main
    with pytest.raises(ConfidentialityError):
        main(["--manuscript", str(tmp_path / "m.pdf"), "--refs", str(tmp_path / "refs")])


def test_serve_builds_launcher_app_without_run(monkeypatch):
    import probatio.web.serve as serve
    built = {}
    monkeypatch.setattr(serve, "create_app", lambda s: built.setdefault("launcher", True) or object())
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    serve.main(["--port", "9999"])           # no --run
    assert built.get("launcher") is True
