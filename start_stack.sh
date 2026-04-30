#!/usr/bin/env bash
# Robot static HTTP + VLA live_frame_server (FastAPI). From repo root: ./start_stack.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_DIR="${ROBOT_DIR:-$HERE/Robot}"
VLA_DIR="${VLA_DIR:-$HERE/VLA}"
ROBOT_PORT="${ROBOT_PORT:-8765}"
VLA_PORT="${VLA_PORT:-8787}"
# Max out CPU for BLAS/OpenMP/OpenCV/torch (override VLA_NUM_THREADS to cap).
export VLA_NUM_THREADS="${VLA_NUM_THREADS:-$(nproc)}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-false}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PY_VLA="${VLA_DIR}/.venv/bin/python3"
if [[ ! -x "$PY_VLA" ]]; then
  PY_VLA="python3"
fi

cd "$ROBOT_DIR"
python3 -m http.server "$ROBOT_PORT" --bind 0.0.0.0 &
PID_ROBOT=$!

cd "$VLA_DIR"
"$PY_VLA" scripts/live_frame_server.py --host 0.0.0.0 --port "$VLA_PORT" &
PID_VLA=$!

cleanup() {
  kill "$PID_ROBOT" "$PID_VLA" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Robot:  http://127.0.0.1:${ROBOT_PORT}/"
echo "VLA API: http://127.0.0.1:${VLA_PORT}/health"
wait "$PID_ROBOT" "$PID_VLA"
