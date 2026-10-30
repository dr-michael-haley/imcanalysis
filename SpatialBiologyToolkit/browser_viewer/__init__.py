"""Static application launcher; scientific files are read only by the browser."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
import webbrowser


def static_directory() -> Path:
    """Locate the browser build without importing scientific dependencies."""
    directory = Path(__file__).resolve().parent / "static"
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(
            "Browser viewer assets are not built. In the repository's web_viewer "
            "directory, run 'pnpm install --frozen-lockfile' and 'pnpm build', "
            "then retry 'sbt gui web'. A hosted copy needs no SBT installation."
        )
    return directory


class StaticViewerHandler(SimpleHTTPRequestHandler):
    """Serve only bundled regular files; never expose datasets or accept uploads."""

    def send_head(self) -> BinaryIO | None:
        root = Path(self.directory).resolve()
        target = Path(self.translate_path(self.path)).resolve()
        if target == root:
            target = root / "index.html"
        if not target.is_relative_to(root) or not target.is_file():
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
            "worker-src 'self' blob:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Avoid logging browser paths; datasets never reach this handler.
        return


def serve(*, port: int = 8765, open_browser: bool = True) -> None:
    """Serve packaged UI assets on loopback until interrupted."""
    root = static_directory()
    handler = partial(StaticViewerHandler, directory=str(root))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"SBT IMC Explorer: {url}\nSelect local files in the browser. Ctrl+C to stop.", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
