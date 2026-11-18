"""Drawing helpers: boxes, labels and the on-screen statistics panel."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from detector import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)

#: Distinct colours used for the panel / help text.
PANEL_COLOR = (32, 32, 32)
TRACK_COLOR = (0, 200, 255)
FPS_COLOR = (80, 255, 120)


def _hsv_color(index: int) -> Tuple[int, int, int]:
    """Return a stable, well separated BGR colour for a given index."""
    hue = int((index * 47) % 180)
    return cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()


def color_for(track_id: int | None, class_id: int, color_by: str = "id") -> Tuple[int, int, int]:
    """Pick the box colour: by tracking id (default) or by class id."""
    index = class_id if color_by == "class" else (track_id if track_id is not None else class_id)
    return tuple(int(c) for c in _hsv_color(abs(index) * 3 + 7))


def _put_text(img, text: str, org, color=TEXT_COLOR, scale: float = 0.5, thickness: int = 1, outline: bool = True) -> None:
    """Draw text, optionally with a dark outline for readability on any frame."""
    if outline:
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _readable_on(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return black or white, whichever contrasts better with ``color``."""
    b, g, r = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_detections(img: np.ndarray, detections: Sequence[Detection], color_by: str = "id") -> None:
    """Draw one box, label and confidence per detection, in place."""
    for det in detections:
        x1, y1, x2, y2 = det.box
        color = color_for(det.track_id, det.class_id, color_by)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"ID {det.track_id} | {label}"

        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(y1, th + 6)
        cv2.rectangle(img, (x1, top - th - 6), (x1 + tw + 6, top + baseline), color, -1)
        _put_text(img, label, (x1 + 3, top - 4), color=_readable_on(color), outline=False)


def draw_panel(
    img: np.ndarray,
    fps: float,
    tracked: int,
    detected: int,
    extra: Iterable[str] = (),
) -> None:
    """Draw the FPS / object counter panel in the top-left corner."""
    lines: List[str] = [f"FPS: {fps:5.1f}", f"Tracked objects: {tracked}", f"Detections: {detected}"]
    lines.extend(extra)

    metrics = [cv2.getTextSize(line, FONT, 0.5, 1) for line in lines]
    panel_w = max(m[0][0] for m in metrics) + 16
    panel_h = sum(m[0][1] + m[1] + 6 for m in metrics) + 10
    cv2.rectangle(img, (8, 8), (8 + panel_w, 8 + panel_h), PANEL_COLOR, -1)

    y = 8 + metrics[0][0][1] + 2
    for line, ((width, height), baseline) in zip(lines, metrics):
        color = FPS_COLOR if line.startswith("FPS") else TEXT_COLOR
        _put_text(img, line, (16, y), color=color, scale=0.5)
        y += height + baseline + 6


def draw_help(img: np.ndarray) -> None:
    """Draw the key hints at the bottom of the frame."""
    text = "q / ESC: quit"
    (w, h), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    cv2.rectangle(img, (8, img.shape[0] - h - 16), (8 + w + 14, img.shape[0] - 6), PANEL_COLOR, -1)
    _put_text(img, text, (15, img.shape[0] - 12), color=(200, 200, 200), scale=0.45)
