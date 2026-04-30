#!/usr/bin/env python3
"""
Static HTTP server for Robot/ with dev-friendly cache headers.

python3 -m http.server does not send Cache-Control; browsers often reuse a
stale index.html. We disable caching for HTML/JS/CSS so UI changes show up
immediately after restart.
"""
from __future__ import annotations

import argparse
import http.server
import os
import socketserver


class _ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class RobotDevRequestHandler(http.server.SimpleHTTPRequestHandler):
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
            "(HTML/JS/CSS: no-store cache)"
        )
        httpd.serve_forever()


if __name__ == "__main__":
    main()
