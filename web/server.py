"""Minimal HTTP server for the puzzle game web UI.

Usage:
    conda activate jigsaw
    # From project root:
    python3 -m web.server [--port 8080]
    # From web/ directory:
    python3 server.py [--port 8080]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import signal
import socket
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np

# Add project root to path (handles both "python3 -m web.server" and "python3 server.py")
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATIC_DIR = _THIS_DIR / "static"


class PuzzleHTTPHandler(SimpleHTTPRequestHandler):
    """Handles static file serving and puzzle API endpoints."""

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self._handle_api()
        elif self.path == "/" or self.path == "/index.html":
            self._serve_file(STATIC_DIR / "index.html", "text/html")
        elif self.path.startswith("/static/"):
            rel = self.path[len("/static/"):]
            file_path = STATIC_DIR / rel
            if file_path.is_file():
                content_type = self._guess_type(str(file_path))
                self._serve_file(file_path, content_type)
            else:
                self.send_error(404, f"File not found: {rel}")
        else:
            self.send_error(404, "Not found")

    def _handle_api(self) -> None:
        if self.path.startswith("/api/image"):
            self._api_generate_image()
        else:
            self.send_error(404, "Unknown API endpoint")

    def _api_generate_image(self) -> None:
        """Generate a natural-like image and return as base64 PNG."""
        try:
            from jigsaw.utils import generate_natural_like_image

            # Parse query params
            params = self._parse_query()
            size = int(params.get("size", "600"))
            seed = int(params.get("seed", "42"))
            size = max(100, min(size, 2000))  # clamp

            image = generate_natural_like_image(size=size, seed=seed)

            # Convert to PNG base64
            try:
                import cv2

                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".png", bgr)
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            except ImportError:
                # Fallback: use PIL
                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(image)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            self._json_response({"image": b64, "size": size, "seed": seed})

        except Exception as e:
            self._json_response({"error": str(e)}, status=500)

    def _parse_query(self) -> dict:
        """Parse query string into dict."""
        if "?" not in self.path:
            return {}
        qs = self.path.split("?", 1)[1]
        params = {}
        for pair in qs.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k] = v
        return params

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _guess_type(path: str) -> str:
        if path.endswith(".html"):
            return "text/html"
        if path.endswith(".css"):
            return "text/css"
        if path.endswith(".js"):
            return "application/javascript"
        if path.endswith(".png"):
            return "image/png"
        if path.endswith(".jpg") or path.endswith(".jpeg"):
            return "image/jpeg"
        return "application/octet-stream"

    def log_message(self, format: str, *args) -> None:
        # Suppress default logging for static files
        if args and "/static/" in str(args[0]):
            return
        super().log_message(format, *args)


class ReusableHTTPServer(HTTPServer):
    """HTTPServer that sets SO_REUSEADDR before bind to avoid 'Address already in use'."""

    allow_reuse_address = True
    allow_reuse_port = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Puzzle game web server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    server = ReusableHTTPServer(("0.0.0.0", args.port), PuzzleHTTPHandler)

    def _shutdown(signum, frame):
        print("\nShutting down.")
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"Puzzle game server running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
