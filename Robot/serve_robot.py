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
import atexit
import http.server
import json
import os
import socketserver
import signal
import sys
import time
import traceback

try:
    from path_planner import plan_path
except ImportError:  # pragma: no cover
    plan_path = None


class _ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


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
        default=8799,
        help="TCP port (default: 8799)",
    )
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0; use 127.0.0.1 for local-only)",
    )
    parser.add_argument(
        "--kill-existing",
        action="store_true",
        help="If a previous serve_robot instance is still running on this port (tracked via pidfile), terminate it before starting.",
    )
    args = parser.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    pidfile = os.path.join(root, f".serve_robot.{args.bind.replace(':','_')}.{args.port}.pid")

    def _pid_alive(pid: int) -> bool:
        if pid <= 1:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _cleanup_pidfile() -> None:
        try:
            if os.path.exists(pidfile):
                os.remove(pidfile)
        except Exception:
            pass

    def _term_existing_from_pidfile() -> bool:
        if not os.path.exists(pidfile):
            return False
        try:
            with open(pidfile, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            pid = int(raw)
        except Exception:
            _cleanup_pidfile()
            return False
        if not _pid_alive(pid):
            _cleanup_pidfile()
            return False
        if not args.kill_existing:
            print(
                f"Port {args.port} already has a tracked server process (pid={pid}).\n"
                f"Stop it first, or rerun with: python3 serve_robot.py {args.port} --bind {args.bind} --kill-existing",
                file=sys.stderr,
            )
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to terminate pid {pid}: {e}", file=sys.stderr)
            return False
        # Wait briefly for it to exit.
        for _ in range(30):
            if not _pid_alive(pid):
                _cleanup_pidfile()
                return True
            time.sleep(0.05)
        # Last resort
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        _cleanup_pidfile()
        return True

    _term_existing_from_pidfile()

    def _handle_exit(_signum: int, _frame) -> None:  # type: ignore[override]
        _cleanup_pidfile()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_exit)
    signal.signal(signal.SIGINT, _handle_exit)
    atexit.register(_cleanup_pidfile)

    try:
        httpd = _ReusableThreadingTCPServer((args.bind, args.port), RobotDevRequestHandler)
    except OSError as e:
        if getattr(e, "errno", None) == 98:
            msg = (
                f"Address already in use: {args.bind}:{args.port}.\n"
                f"If this is a previous run, stop the old process or rerun with --kill-existing.\n"
                f"Tip: ss -ltnp | grep :{args.port}"
            )
            print(msg, file=sys.stderr)
        raise

    with httpd:
        with open(pidfile, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        print(
            f"Robot dev server: http://{args.bind}:{args.port}/ "
            "(HTML/JS/CSS: no-store cache; POST /api/plan for Python pathfinding)"
        )
        # Note: for remote access, use the machine's public IP (not 127.0.0.1).
        # Example: http://<PUBLIC_IP>:<PORT>/
        httpd.serve_forever()


if __name__ == "__main__":
    main()
