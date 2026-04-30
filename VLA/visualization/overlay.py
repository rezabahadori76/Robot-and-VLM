from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from common.types import FrameWorldState

# BGR palette: avoid pure green at index 0 (was dominating whole frame for large “first” masks).
# Colors are chosen via _color_bgr_for_label(label) so each class keeps a stable hue.
_PALETTE_BGR: list[tuple[int, int, int]] = [
    (40, 140, 255),  # orange
    (255, 180, 60),  # cyan-ish
    (200, 80, 255),  # magenta / pink
    (60, 220, 220),  # yellow
    (255, 100, 100), # light red
    (100, 255, 150), # mint (not neon green)
    (180, 120, 255), # lavender
    (50, 200, 255),  # sky
    (255, 200, 80),  # gold
    (90, 70, 220),   # coral
    (220, 180, 60),  # spring green
    (130, 255, 180), # seafoam
    (255, 60, 160),  # rose
    (160, 160, 40),  # olive
    (70, 130, 200),  # steel orange
    (200, 255, 100), # lime (muted)
]


def _stable_label_hash(label: str) -> int:
    s = (label or "object").lower().strip()
    h = 5381
    for i, c in enumerate(s):
        h = ((h << 5) + h) + ord(c) + (i * 17)
        h &= 0xFFFFFFFF
    return int(h)


def _color_bgr_for_label(label: str) -> tuple[int, int, int]:
    idx = _stable_label_hash(label) % len(_PALETTE_BGR)
    return _PALETTE_BGR[idx]


def _rle_to_mask(rle: dict) -> np.ndarray:
    size = rle['size']
    counts = rle['counts']
    flat = []
    val = 0
    for c in counts:
        flat.extend([val] * int(c))
        val = 1 - val
    return np.array(flat, dtype=np.uint8).reshape((size[0], size[1]), order='F')


def render_overlay_on_bgr(
    img_bgr: np.ndarray,
    state: FrameWorldState,
    draw_masks: bool,
    draw_boxes: bool,
    box_line_thickness: int = 1,
    label_font_thickness: int = 1,
    draw_semantic_label: bool = True,
) -> np.ndarray:
    """Draw detection boxes + SAM-style masks on a BGR image (same style as viz/overlays).

    When ``draw_semantic_label`` is False, no VLM/room caption is drawn (detection+SAM only).
    """
    img = img_bgr.copy()
    if draw_masks:
        # Color by semantic class name so e.g. "floor" ≠ "wall" even when order changes.
        for seg in state.segments:
            mask = _rle_to_mask(seg.mask_rle)
            b, g, r = _color_bgr_for_label(seg.label)
            color = np.array([b, g, r], dtype=np.uint8)
            img[mask == 1] = (0.52 * img[mask == 1] + 0.48 * color).astype(np.uint8)

    if draw_boxes:
        t = max(1, int(box_line_thickness))
        lt = max(1, int(label_font_thickness))
        for det in state.detections:
            x1, y1, x2, y2 = map(int, det.bbox_xyxy)
            col = _color_bgr_for_label(det.label)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, t)
            cv2.putText(
                img,
                f'{det.label}:{det.score:.2f}',
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                lt,
            )

    if draw_semantic_label and (state.semantics.room_label or "").strip():
        cv2.putText(
            img,
            f'room={state.semantics.room_label}',
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 220, 255),
            2,
        )
    return img


def render_perception_overlay(
    states: List[FrameWorldState],
    frames_dir: Path,
    output_dir: Path,
    draw_masks: bool,
    draw_boxes: bool,
    box_line_thickness: int = 1,
    label_font_thickness: int = 1,
    draw_semantic_label: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for st in states:
        frame_path = frames_dir / f'frame_{st.frame_id:06d}.jpg'
        if not frame_path.exists():
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        out = render_overlay_on_bgr(
            img,
            st,
            draw_masks,
            draw_boxes,
            box_line_thickness,
            label_font_thickness,
            draw_semantic_label,
        )
        cv2.imwrite(str(output_dir / f'overlay_{st.frame_id:06d}.jpg'), out)
