#!/usr/bin/env python3
"""
Static HTTP server for Robot/ with dev-friendly cache headers.

Also exposes POST /api/plan — JSON in, JSON out — using path_planner.plan_path
(weighted A* on the occupancy grid from the browser + detection keepout).

python3 -m http.server does not send Cache-Control; browsers often reuse a
stale index.html. We disable caching for HTML/JS/CSS so UI changes show up
immediately after restart.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import traceback

try:
    from path_planner import plan_path
except ImportError:  # pragma: no cover
    plan_path = None


class _ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class RobotDevRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self) -> None:
        path = self.path.partition("?")[0].rstrip("/")
        if path != "/api/plan":
            self.send_error(404, "Not Found")
            return
        if plan_path is None:
            self.send_error(500, "path_planner not available")
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
            result = plan_path(payload)
            data = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except json.JSONDecodeError:
            err = json.dumps({"ok": False, "reason": "bad-json"}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
        except Exception:
            tb = traceback.format_exc()
            err = json.dumps({"ok": False, "reason": "server-error", "detail": tb}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def end_headers(self) -> None:
        path = self.path.partition("?")[0].rstrip("/") or "/"
        if path == "/" or path.lower().endswith(
            (".html", ".htm", ".js", ".css", ".mjs")
        ):
            self.send_header(
                "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
            )
            self.send_header("Pragma", "no-cache")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=8765,
        help="TCP port (default: 8765)",
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; use 0.0.0.0 for all interfaces)",
    )
    args = parser.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    with _ReusableThreadingTCPServer(
        (args.bind, args.port), RobotDevRequestHandler
    ) as httpd:
        print(
            f"Robot dev server: http://{args.bind}:{args.port}/ "
            "(HTML/JS/CSS: no-store cache; POST /api/plan for Python pathfinding)"
        )
        httpd.serve_forever()


if __name__ == "__main__":
    main()
