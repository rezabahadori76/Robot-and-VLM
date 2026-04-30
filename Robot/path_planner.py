#!/usr/bin/env python3
"""
Map-based multi-algorithm route planning for wheelchair navigation.

Algorithms used:
- Dijkstra (A* with zero heuristic)
- A* (Manhattan heuristic on 4-neighbor grid)
- Weighted A* (faster directional bias, still practical on occupancy maps)

Route quality model:
- Hard collision constraints from blocked cells + detections keepout disks
- Corridor risk from BFS distance-to-obstacle field
- Optional novelty penalty to avoid reusing prior routes
- Optional line-of-sight compression to reduce zig-zag waypoints
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

INF = 10**9
RISK_DEFAULT = 0.22
EPS_DEFAULT = 0.35
NOVELTY_DEFAULT = 0.42


@dataclass(frozen=True)
class Bounds:
    x0: float
    x1: float
    z0: float
    z1: float


@dataclass(frozen=True)
class Cell:
    i: int
    j: int


def _world_to_cell(x: float, z: float, b: Bounds, cell_size: float, nx: int, nz: int) -> Cell:
    i = int(round((x - b.x0) / cell_size))
    j = int(round((z - b.z0) / cell_size))
    i = max(0, min(nx - 1, i))
    j = max(0, min(nz - 1, j))
    return Cell(i=i, j=j)


def _cell_to_world(c: Cell, b: Bounds, cell_size: float) -> Tuple[float, float]:
    return (b.x0 + c.i * cell_size, b.z0 + c.j * cell_size)


def _grid_dims(b: Bounds, cell_size: float) -> Tuple[int, int]:
    nx = int(math.ceil((b.x1 - b.x0) / cell_size)) + 1
    nz = int(math.ceil((b.z1 - b.z0) / cell_size)) + 1
    return nx, nz


def _mark_disk(
    blocked: Set[Cell],
    cx: float,
    cz: float,
    radius: float,
    b: Bounds,
    cell_size: float,
    nx: int,
    nz: int,
) -> None:
    c0 = _world_to_cell(cx, cz, b, cell_size, nx, nz)
    r_idx = int(math.ceil(radius / cell_size)) + 1
    for di in range(-r_idx, r_idx + 1):
        for dj in range(-r_idx, r_idx + 1):
            i = c0.i + di
            j = c0.j + dj
            if i < 0 or j < 0 or i >= nx or j >= nz:
                continue
            wx = b.x0 + i * cell_size
            wz = b.z0 + j * cell_size
            if math.hypot(wx - cx, wz - cz) <= radius:
                blocked.add(Cell(i=i, j=j))


def _detection_radius(
    distance: Optional[float],
    base: float = 0.46,
    min_keepout_m: float = 0.0,
) -> float:
    d = distance if isinstance(distance, (int, float)) and math.isfinite(distance) else 2.0
    scale = max(0.82, min(1.28, 1.12 - d * 0.018))
    min_r = max(0.30, float(min_keepout_m or 0.0))
    return min(0.75, max(min_r, base * scale))


def _parse_blocked_keys(raw: object, nx: int, nz: int) -> Set[Cell]:
    out: Set[Cell] = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, str):
            continue
        parts = item.split(",")
        if len(parts) != 2:
            continue
        try:
            i = int(parts[0])
            j = int(parts[1])
        except ValueError:
            continue
        if i < 0 or j < 0 or i >= nx or j >= nz:
            continue
        out.add(Cell(i=i, j=j))
    return out


def _extract_avoid_signatures(avoid_paths: object) -> Set[Tuple[Tuple[float, float], ...]]:
    sigs: Set[Tuple[Tuple[float, float], ...]] = set()
    if not isinstance(avoid_paths, list):
        return sigs
    for route in avoid_paths:
        if not isinstance(route, list) or len(route) < 2:
            continue
        pts: List[Tuple[float, float]] = []
        for p in route:
            if not isinstance(p, dict):
                continue
            if "x" not in p or "z" not in p:
                continue
            pts.append((round(float(p["x"]), 4), round(float(p["z"]), 4)))
        if len(pts) >= 2:
            sigs.add(tuple(pts))
    return sigs


def _neighbors(c: Cell) -> Iterable[Cell]:
    yield Cell(c.i + 1, c.j)
    yield Cell(c.i - 1, c.j)
    yield Cell(c.i, c.j + 1)
    yield Cell(c.i, c.j - 1)


def _heur(a: Cell, g: Cell) -> float:
    return float(abs(a.i - g.i) + abs(a.j - g.j))


def _bfs_clearance_from_blocked(nx: int, nz: int, blocked: Set[Cell]) -> List[List[float]]:
    """For every cell, graph distance (4-neigh) to nearest blocked cell; blocked => 0."""
    dist: List[List[float]] = [[INF] * nz for _ in range(nx)]
    q: deque[Cell] = deque()
    for c in blocked:
        dist[c.i][c.j] = 0.0
        q.append(c)
    while q:
        cur = q.popleft()
        d0 = dist[cur.i][cur.j]
        for nb in _neighbors(cur):
            if nb.i < 0 or nb.j < 0 or nb.i >= nx or nb.j >= nz:
                continue
            if nb in blocked:
                continue
            nd = d0 + 1.0
            if nd < dist[nb.i][nb.j]:
                dist[nb.i][nb.j] = nd
                q.append(nb)
    return dist


def _edge_cost(
    cur: Cell,
    nb: Cell,
    dist: List[List[float]],
    risk_w: float,
    eps: float,
) -> float:
    dmn = min(dist[cur.i][cur.j], dist[nb.i][nb.j])
    if dmn >= INF * 0.5:
        return 1.0
    return 1.0 + risk_w / (eps + dmn)


def _snap_to_nearby_free(c: Cell, blocked: Set[Cell], nx: int, nz: int, max_r: int = 10) -> Optional[Cell]:
    if c not in blocked:
        return c
    best: Optional[Cell] = None
    best_d = INF
    for di in range(-max_r, max_r + 1):
        for dj in range(-max_r, max_r + 1):
            ni, nj = c.i + di, c.j + dj
            if ni < 0 or nj < 0 or ni >= nx or nj >= nz:
                continue
            cc = Cell(ni, nj)
            if cc in blocked:
                continue
            md = abs(di) + abs(dj)
            if md < best_d:
                best_d = md
                best = cc
    return best


def _build_avoid_mask(
    avoid_paths: object,
    b: Bounds,
    cell_size: float,
    nx: int,
    nz: int,
) -> Set[Cell]:
    """Cells close to previously selected routes; used for novelty penalty."""
    mask: Set[Cell] = set()
    if not isinstance(avoid_paths, list):
        return mask
    for route in avoid_paths:
        if not isinstance(route, list):
            continue
        for p in route:
            if not isinstance(p, dict):
                continue
            if "x" not in p or "z" not in p:
                continue
            c = _world_to_cell(float(p["x"]), float(p["z"]), b, cell_size, nx, nz)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = c.i + di
                    nj = c.j + dj
                    if ni < 0 or nj < 0 or ni >= nx or nj >= nz:
                        continue
                    mask.add(Cell(ni, nj))
    return mask


def _astar_weighted(
    start: Cell,
    goal: Cell,
    nx: int,
    nz: int,
    blocked: Set[Cell],
    dist: List[List[float]],
    risk_w: float,
    eps: float,
    h_weight: float = 1.0,
    novelty_mask: Optional[Set[Cell]] = None,
    novelty_weight: float = NOVELTY_DEFAULT,
    min_clearance_cells: float = 0.0,
) -> Optional[List[Cell]]:
    if start in blocked or goal in blocked:
        return None
    if dist[start.i][start.j] < min_clearance_cells or dist[goal.i][goal.j] < min_clearance_cells:
        return None
    open_heap: List[Tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start))
    g_score: Dict[Cell, float] = {start: 0.0}
    parent: Dict[Cell, Cell] = {}
    closed: Set[Cell] = set()

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            out = [cur]
            while cur in parent:
                cur = parent[cur]
                out.append(cur)
            out.reverse()
            return out
        closed.add(cur)
        gcur = g_score[cur]

        for nb in _neighbors(cur):
            if nb.i < 0 or nb.j < 0 or nb.i >= nx or nb.j >= nz:
                continue
            if nb in blocked:
                continue
            if dist[nb.i][nb.j] < min_clearance_cells:
                continue
            w = _edge_cost(cur, nb, dist, risk_w, eps)
            tg = gcur + w
            if tg < g_score.get(nb, float("inf")):
                parent[nb] = cur
                g_score[nb] = tg
                counter += 1
                novelty = novelty_weight if novelty_mask and nb in novelty_mask else 0.0
                f = tg + h_weight * _heur(nb, goal) + novelty
                heapq.heappush(open_heap, (f, counter, nb))
    return None


def _line_hits_blocked(i0: int, j0: int, i1: int, j1: int, blocked: Set[Cell], nx: int, nz: int) -> bool:
    steps = max(abs(i1 - i0), abs(j1 - j0), 1)
    for s in range(steps + 1):
        t = s / steps
        i = int(round(i0 + (i1 - i0) * t))
        j = int(round(j0 + (j1 - j0) * t))
        if i < 0 or j < 0 or i >= nx or j >= nz:
            return True
        if Cell(i, j) in blocked:
            return True
    return False


def _compress_cell_path(path: List[Cell], blocked: Set[Cell], nx: int, nz: int) -> List[Cell]:
    if len(path) <= 2:
        return path
    out: List[Cell] = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i and _line_hits_blocked(path[i].i, path[i].j, path[j].i, path[j].j, blocked, nx, nz):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _path_length_world(path: Sequence[dict]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += math.hypot(path[i + 1]["x"] - path[i]["x"], path[i + 1]["z"] - path[i]["z"])
    return total


def _avg_clearance_cells(cells: Sequence[Cell], dist_field: List[List[float]]) -> float:
    if not cells:
        return 0.0
    s = 0.0
    for c in cells:
        d = dist_field[c.i][c.j]
        if d >= INF * 0.5:
            d = 8.0
        s += d
    return s / len(cells)


def plan_path(payload: dict) -> dict:
    b_raw = payload["bounds"]
    b = Bounds(
        x0=float(b_raw["x0"]),
        x1=float(b_raw["x1"]),
        z0=float(b_raw["z0"]),
        z1=float(b_raw["z1"]),
    )
    cell_size = float(payload.get("cell_size", 0.14))
    nx, nz = _grid_dims(b, cell_size)

    start = payload["start"]
    goal = payload["goal"]
    s_cell = _world_to_cell(float(start["x"]), float(start["z"]), b, cell_size, nx, nz)
    g_cell = _world_to_cell(float(goal["x"]), float(goal["z"]), b, cell_size, nx, nz)

    blocked: Set[Cell] = set()
    blocked |= _parse_blocked_keys(payload.get("blocked_cells"), nx, nz)
    blocked |= _parse_blocked_keys(payload.get("dynamic_blocked_cells"), nx, nz)

    for obs in payload.get("static_obstacles", []):
        _mark_disk(
            blocked,
            float(obs["x"]),
            float(obs["z"]),
            float(obs.get("radius", 0.45)),
            b,
            cell_size,
            nx,
            nz,
        )

    marked_detections = 0
    min_keepout_m = float(payload.get("min_keepout_m", 0.0))
    for det in payload.get("detections", []):
        p = det.get("point") or {}
        if "x" not in p or "z" not in p:
            continue
        r = _detection_radius(
            det.get("distance"),
            base=float(payload.get("detection_base_radius", 0.46)),
            min_keepout_m=min_keepout_m,
        )
        _mark_disk(blocked, float(p["x"]), float(p["z"]), r, b, cell_size, nx, nz)
        marked_detections += 1

    s_cell = _snap_to_nearby_free(s_cell, blocked, nx, nz) or s_cell
    g_cell = _snap_to_nearby_free(g_cell, blocked, nx, nz) or g_cell
    if s_cell in blocked or g_cell in blocked:
        return {
            "ok": False,
            "reason": "start-or-goal-blocked",
            "markedDetections": marked_detections,
            "grid": {"nx": nx, "nz": nz, "cellSize": cell_size},
        }

    risk_w = float(payload.get("corridor_risk_weight", RISK_DEFAULT))
    eps = float(payload.get("corridor_risk_eps", EPS_DEFAULT))
    min_clearance_m = float(payload.get("min_clearance_m", 0.32))
    min_clearance_cells = max(0.0, min_clearance_m / max(1e-6, cell_size))
    novelty_weight = float(payload.get("novelty_weight", NOVELTY_DEFAULT))
    avoid_paths_raw = payload.get("avoid_paths")
    avoid_mask = _build_avoid_mask(avoid_paths_raw, b, cell_size, nx, nz)
    avoid_signatures = _extract_avoid_signatures(avoid_paths_raw)
    dist_field = _bfs_clearance_from_blocked(nx, nz, blocked)
    compress = bool(payload.get("compress_path", True))

    algorithms = [
        {"id": "dijkstra", "h_weight": 0.0, "risk_scale": 1.0, "novelty_on": True},
        {"id": "astar", "h_weight": 1.0, "risk_scale": 1.0, "novelty_on": True},
        {"id": "weighted_astar", "h_weight": 1.25, "risk_scale": 0.9, "novelty_on": True},
        {"id": "astar_low_risk_bias", "h_weight": 1.0, "risk_scale": 1.3, "novelty_on": False},
    ]
    if isinstance(payload.get("algorithms"), list):
        requested = set(str(x) for x in payload["algorithms"])
        algorithms = [a for a in algorithms if a["id"] in requested] or algorithms

    candidates = []
    for cfg in algorithms:
        cells = _astar_weighted(
            s_cell,
            g_cell,
            nx,
            nz,
            blocked,
            dist_field,
            risk_w=risk_w * cfg["risk_scale"],
            eps=eps,
            h_weight=cfg["h_weight"],
            novelty_mask=avoid_mask,
            novelty_weight=novelty_weight if cfg["novelty_on"] else 0.0,
            min_clearance_cells=min_clearance_cells,
        )
        if not cells:
            continue
        if compress:
            cells = _compress_cell_path(cells, blocked, nx, nz)
        path = []
        for c in cells:
            x, z = _cell_to_world(c, b, cell_size)
            path.append({"x": round(x, 4), "z": round(z, 4)})
        avg_clear = _avg_clearance_cells(cells, dist_field)
        length_m = _path_length_world(path)
        # lower is better: shorter length and wider avg clearance
        score = length_m + 1.8 / (0.4 + avg_clear)
        candidates.append(
            {
                "algorithm": cfg["id"],
                "path": path,
                "pathLengthCells": len(cells),
                "lengthM": round(length_m, 4),
                "avgClearanceCells": round(avg_clear, 4),
                "score": round(score, 6),
            }
        )

    if not candidates:
        return {
            "ok": False,
            "reason": "no-path",
            "markedDetections": marked_detections,
            "grid": {"nx": nx, "nz": nz, "cellSize": cell_size},
        }

    # Deduplicate exact repeated paths (same coordinates sequence)
    unique = {}
    for c in candidates:
        sig = tuple((p["x"], p["z"]) for p in c["path"])
        prev = unique.get(sig)
        if prev is None or c["score"] < prev["score"]:
            unique[sig] = c
    candidates = list(unique.values())
    candidates.sort(key=lambda c: c["score"])

    # Hard rule: new path must not be exactly equal to any prior path signature.
    if avoid_signatures:
        novel = []
        for c in candidates:
            sig = tuple((p["x"], p["z"]) for p in c["path"])
            if sig not in avoid_signatures:
                novel.append(c)
        candidates = novel

    if not candidates:
        return {
            "ok": False,
            "reason": "no-novel-path",
            "detail": "all candidate routes exactly matched previous path signatures",
            "markedDetections": marked_detections,
            "grid": {"nx": nx, "nz": nz, "cellSize": cell_size},
        }

    max_alts = int(payload.get("max_alternatives", 3))
    alts = candidates[: max(1, min(6, max_alts))]
    best = alts[0]

    return {
        "ok": True,
        "algorithm": best["algorithm"],
        "how": "Multi-algorithm map planning (Dijkstra/A*/Weighted A*) with collision constraints, corridor-risk scoring, and anti-repeat novelty penalty.",
        "markedDetections": marked_detections,
        "path": best["path"],
        "pathLengthCells": best["pathLengthCells"],
        "alternatives": alts,
        "grid": {"nx": nx, "nz": nz, "cellSize": cell_size},
    }


def _read_payload(path: Optional[str]) -> dict:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map-based wheelchair path planner (weighted A* + detection keepout)."
    )
    parser.add_argument("--input", help="Input JSON file path. If omitted, reads stdin.")
    args = parser.parse_args()

    try:
        payload = _read_payload(args.input)
        result = plan_path(payload)
    except Exception as exc:  # pragma: no cover
        result = {"ok": False, "reason": "invalid-input", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
