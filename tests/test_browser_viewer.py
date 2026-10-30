from __future__ import annotations

from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

from SpatialBiologyToolkit import browser_viewer
from SpatialBiologyToolkit.cli.main import app


def test_cli_check_and_launch_are_lightweight(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    monkeypatch.setattr(browser_viewer, "static_directory", lambda: tmp_path)
    result = runner.invoke(app, ["gui", "web", "--check"])
    assert result.exit_code == 0
    assert "Browser viewer ready" in result.stdout
    launched = []
    monkeypatch.setattr(browser_viewer, "serve", lambda **kwargs: launched.append(kwargs))
    assert runner.invoke(app, ["gui", "web", "--no-browser", "--port", "8877"]).exit_code == 0
    assert launched == [{"port": 8877, "open_browser": False}]
    assert runner.invoke(app, ["gui", "web", "--port", "0"]).exit_code != 0


def test_missing_build_has_actionable_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(browser_viewer, "__file__", str(tmp_path / "__init__.py"))
    result = CliRunner().invoke(app, ["gui", "web", "--check"])
    assert result.exit_code != 0
    assert "pnpm build" in result.output


def test_static_server_never_lists_directories_or_accepts_uploads(tmp_path: Path):
    root = tmp_path / "static"
    root.mkdir()
    (root / "index.html").write_text("<h1>Viewer</h1>", encoding="utf-8")
    (root / "assets").mkdir()
    (tmp_path / "private.h5ad").write_bytes(b"private")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(browser_viewer.StaticViewerHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            assert response.read() == b"<h1>Viewer</h1>"
            assert "connect-src 'self'" in response.headers["Content-Security-Policy"]
        for path in ["/assets/", "/../private.h5ad", "/%2e%2e/private.h5ad"]:
            with pytest.raises(HTTPError) as exc:
                urlopen(base + path)
            assert exc.value.code == 404
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base + "/", data=b"dataset", method="POST"))
        assert exc.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
