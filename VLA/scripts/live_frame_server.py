#!/usr/bin/env python3
"""
FastAPI server:
- POST /process_frame        (multipart JPEG) -> overlay JPEG (viz-style)
- POST /process_frame_bundle (multipart JPEG) -> JSON { overlay_jpeg_b64, detections, segment_boundaries, ... }

Policy (as requested by Robot UI):
- Display: object detection (DINO boxes + labels)
- Planning input: segmentation boundaries (SAM contours)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.chdir(_ROOT)


def _effective_cpu_count() -> int:
    """Logical CPUs for this process (respects cgroup/affinity). Override with VLA_NUM_THREADS."""
    raw = os.environ.get("VLA_NUM_THREADS", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return max(1, v)
        except ValueError:
            pass
    try:
        aff = len(os.sched_getaffinity(0))
        if aff > 0:
            return aff
    except (AttributeError, NotImplementedError, OSError):
        pass
    return max(1, os.cpu_count() or 1)


def _configure_threads() -> None:
    """BLAS/OpenMP/NumExpr before NumPy/OpenCV — same idea as pipeline_runner."""
    n = _effective_cpu_count()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "GOTO_NUM_THREADS",
    ):
        if name not in os.environ:
            os.environ[name] = str(n)
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


_configure_threads()

import io  # noqa: E402
import base64  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response  # noqa: E402

from common.types import FramePacket, FrameWorldState, Pose, SemanticFrame  # noqa: E402
from detection.grounding_dino_detector import GroundingDinoDetector  # noqa: E402
from segmentation.sam_segmenter import SAMSegmenter  # noqa: E402
from visualization.overlay import render_overlay_on_bgr  # noqa: E402


def _resize_max_side(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr, 1.0
    s = max_side / m
    nw, nh = int(round(w * s)), int(round(h * s))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), s


def _rle_decode(rle: dict) -> np.ndarray:
    """Decode Fortran-order RLE into HxW uint8 mask."""
    counts = rle.get("counts", [])
    size = rle.get("size", None)
    if not size or len(size) != 2:
        raise ValueError("invalid rle size")
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        run = int(c)
        if run <= 0:
            continue
        end = min(idx + run, flat.size)
        if val == 1:
            flat[idx:end] = 1
        idx = end
        val = 1 - val
        if idx >= flat.size:
            break
    return flat.reshape((h, w), order="F")


def _mask_to_contours(mask: np.ndarray) -> list[list[list[int]]]:
    """Return simplified contours as list of point lists [[x,y], ...]."""
    m = (mask.astype(np.uint8) * 255)
    cnts, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[list[list[int]]] = []
    for c in cnts:
        if c is None or len(c) < 6:
            continue
        peri = float(cv2.arcLength(c, True))
        eps = max(1.2, peri * 0.012)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 4:
            continue
        out.append([[int(x), int(y)] for x, y in pts])
    return out


def build_app(cfg: dict) -> FastAPI:
    viz = cfg.get("visualization", {})
    lb = dict(cfg.get("live_bridge", {}))
    if os.environ.get("VLA_MAX_INFER_SIDE"):
        lb["max_infer_side"] = int(os.environ["VLA_MAX_INFER_SIDE"])
    if os.environ.get("VLA_JPEG_QUALITY"):
        lb["jpeg_quality"] = int(os.environ["VLA_JPEG_QUALITY"])
    max_side = int(lb.get("max_infer_side", 640))
    jpeg_q = int(lb.get("jpeg_quality", 88))

    detector = GroundingDinoDetector(cfg["detection"])
    segmenter = SAMSegmenter(cfg["segmentation"])
    detector.initialize()
    segmenter.initialize()

    frame_counter = {"i": 0}

    app = FastAPI(title="VLA live frame bridge", version="1.0")

    robot_public_url = os.environ.get("ROBOT_PUBLIC_URL", "http://127.0.0.1:8765").rstrip("/")

    @app.get("/", response_model=None)
    def root(request: Request) -> HTMLResponse | JSONResponse:
        """Browsers: HTML hint (8787 is API only). Scripts: add ?format=json or Accept: application/json."""
        q = request.query_params.get("format", "")
        no_redirect = request.query_params.get("no_redirect", "").lower() in {"1", "true", "yes"}
        accept = request.headers.get("accept", "")
        want_json = q == "json" or (
            "application/json" in accept and "text/html" not in accept.split(",")[0]
        )
        payload = {
            "service": "vla-live-frame",
            "ok": True,
            "endpoints": ["/health", "POST /process_frame"],
            "ui_app_url": robot_public_url,
            "hint": "Open ui_app_url in the browser; this port is the inference API only.",
        }
        if want_json:
            return JSONResponse(payload)
        if "text/html" in accept and not no_redirect:
            # User-friendly behavior: opening API port in browser should jump to UI app.
            return RedirectResponse(url=f"{robot_public_url}/", status_code=307)
        if "text/html" in accept:
            return HTMLResponse(
                f"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>VLA API — UI is on another port</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem;
      background: #0d1117; color: #e6edf3; line-height: 1.6; }}
    a {{ color: #58a6ff; }}
    code {{ background: #21262d; padding: 0.15em 0.4em; border-radius: 4px; }}
    .box {{ border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1rem; }}
  </style>
</head>
<body>
  <h1>Port 8787 = API only</h1>
  <p>This endpoint serves <strong>frame inference</strong> (Grounding DINO + SAM), not the wheelchair 3D UI.</p>
  <p class="box">Open the demo UI in your browser at:<br/>
    <a href="{robot_public_url}/">{robot_public_url}/</a>
  </p>
  <p>Inside the demo HUD, set <code>VLA API</code> to: <code>http://127.0.0.1:8787</code></p>
  <p><small>JSON: <a href="/?format=json">/?format=json</a> · <a href="/health">/health</a></small></p>
</body>
</html>"""
            )
        return JSONResponse(payload)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "pipeline": {
                "mode": "grounding_dino_then_sam",
                "segmentation_requires_detections": True,
            },
            "detection": detector.status(),
            "segmentation": segmenter.status(),
        }

    @app.post("/process_frame")
    async def process_frame_upload(frame: UploadFile = File(...)) -> Response:
        raw = await frame.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty body")
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="not a valid image")

        orig_h, orig_w = bgr.shape[:2]
        small, _scale = _resize_max_side(bgr, max_side)

        frame_counter["i"] += 1
        fid = frame_counter["i"]
        packet = FramePacket(
            frame_id=fid,
            timestamp=0.0,
            rgb_path=None,
            rgb=small,
        )

        try:
            detections = detector.detect(packet)
            # Strict pipeline: detect objects first (Grounding DINO), then segment only detected objects (SAM).
            if detections:
                segments = segmenter.segment(packet, detections)
            else:
                segments = []
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # FrameWorldState requires semantics; live bridge does not run VLM — empty stub, no overlay text.
        state = FrameWorldState(
            frame_id=fid,
            timestamp=0.0,
            pose=Pose(0.0, 0.0, 0.0),
            detections=detections,
            segments=segments,
            semantics=SemanticFrame(room_label="", caption="", attributes={}),
        )

        out_small = render_overlay_on_bgr(
            small,
            state,
            # Legacy endpoint: keep default behavior as configured (typically masks).
            draw_masks=bool(viz.get("draw_masks", True)),
            draw_boxes=bool(viz.get("draw_boxes", False)),
            box_line_thickness=int(viz.get("overlay_box_thickness", 2)),
            label_font_thickness=int(viz.get("overlay_label_thickness", 1)),
            draw_semantic_label=bool(viz.get("show_semantic_label", False)),
        )
        if out_small.shape[1] != orig_w or out_small.shape[0] != orig_h:
            out = cv2.resize(out_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            out = out_small

        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.post("/process_frame_bundle")
    async def process_frame_bundle(frame: UploadFile = File(...)) -> JSONResponse:
        """Return boxes overlay for display + segmentation boundaries for planning."""
        raw = await frame.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty body")
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="not a valid image")

        orig_h, orig_w = bgr.shape[:2]
        small, _scale = _resize_max_side(bgr, max_side)

        frame_counter["i"] += 1
        fid = frame_counter["i"]
        packet = FramePacket(
            frame_id=fid,
            timestamp=0.0,
            rgb_path=None,
            rgb=small,
        )

        try:
            detections = detector.detect(packet)
            if detections:
                segments = segmenter.segment(packet, detections)
            else:
                segments = []
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Build contours in the ORIGINAL capture resolution for easy reprojection on the client.
        seg_boundaries = []
        for s in segments:
            try:
                m_small = _rle_decode(s.mask_rle)
                if m_small.shape[0] != orig_h or m_small.shape[1] != orig_w:
                    m = cv2.resize(m_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                else:
                    m = m_small
                contours = _mask_to_contours(m)
            except Exception:
                contours = []
            seg_boundaries.append(
                {
                    "label": s.label,
                    "score": float(s.score),
                    "bbox_xyxy": [float(x) for x in s.bbox_xyxy],
                    "contours": contours,
                }
            )

        # Render overlay for display: boxes only (DINO output).
        state = FrameWorldState(
            frame_id=fid,
            timestamp=0.0,
            pose=Pose(0.0, 0.0, 0.0),
            detections=detections,
            segments=segments,
            semantics=SemanticFrame(room_label="", caption="", attributes={}),
        )
        out_small = render_overlay_on_bgr(
            small,
            state,
            # User-requested mixed overlay: show SAM segmentation + DINO boxes together.
            draw_masks=True,
            draw_boxes=True,
            box_line_thickness=int(viz.get("overlay_box_thickness", 2)),
            label_font_thickness=int(viz.get("overlay_label_thickness", 1)),
            draw_semantic_label=False,
        )
        if out_small.shape[1] != orig_w or out_small.shape[0] != orig_h:
            out = cv2.resize(out_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            out = out_small
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")

        det_payload = [
            {
                "label": d.label,
                "score": float(d.score),
                "bbox_xyxy": [float(x) for x in d.bbox_xyxy],
            }
            for d in detections
        ]
        return JSONResponse(
            {
                "ok": True,
                "frame_id": fid,
                "orig_w": orig_w,
                "orig_h": orig_h,
                "infer_w": int(small.shape[1]),
                "infer_h": int(small.shape[0]),
                "detections": det_payload,
                "segment_boundaries": seg_boundaries,
                "overlay_jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_ROOT / "config" / "live_robot_bridge.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--log-level",
        default=os.environ.get("VLA_LOG_LEVEL", "warning"),
        help="uvicorn log level (default: warning — less I/O than info)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch

    n = _effective_cpu_count()
    try:
        torch.set_num_threads(n)
    except RuntimeError:
        pass
    try:
        interop = max(1, min(8, max(1, n // 8)))
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    app = build_app(cfg)
    # Throughput: fast HTTP stack + no per-request access log; loop=uvloop when installed (uvicorn[standard]).
    run_kw: dict = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": False,
        "timeout_keep_alive": 120,
    }
    try:
        import httptools  # noqa: F401

        run_kw["http"] = "httptools"
    except ImportError:
        pass
    try:
        import uvloop  # noqa: F401

        run_kw["loop"] = "uvloop"
    except ImportError:
        pass
    uvicorn.run(**run_kw)


if __name__ == "__main__":
    main()
