"""Video source handling: parsing, safe opening and writing.

Every helper in this module raises :class:`SourceError` with a human readable
message instead of letting a raw OpenCV error escape, so the CLI can report
problems such as "camera not found" or "file does not exist".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2

Source = Union[int, str]

#: Backends tried (in order) when ``backend="auto"`` on Windows.
_WINDOWS_BACKENDS = ("auto", "dshow", "msmf")


class SourceError(RuntimeError):
    """Raised when a video source cannot be opened or yields no frames."""


def parse_source(source: str) -> Source:
    """Convert a CLI source string into a camera index or a path/URL.

    Numeric values are treated as webcam indexes, everything else as a file
    path or stream URL.
    """
    text = str(source).strip()
    if not text:
        raise SourceError("No video source given. Use a camera index (e.g. 0) or a file path.")
    if text.lstrip("+").isdigit():
        index = int(text)
        if index < 0:
            raise SourceError(f"Invalid camera index '{source}'. It must be 0 or greater.")
        return index
    if text.lstrip("-").isdigit():
        raise SourceError(f"Invalid camera index '{source}'. It must be 0 or greater.")
    return text


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://", "rtsp://", "rtmp://", "tcp://"))


def _backend_id(name: str) -> Optional[int]:
    return {
        "auto": cv2.CAP_ANY,
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }.get(name.lower())


def _open_capture(source: Source, backend: str) -> cv2.VideoCapture:
    backend_id = _backend_id(backend)
    if backend_id is None:
        raise SourceError(
            f"Unknown capture backend '{backend}'. Choose one of: auto, any, dshow, msf."
        )
    return cv2.VideoCapture(source, backend_id)


def open_capture(
    source: Source,
    backend: str = "auto",
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
) -> cv2.VideoCapture:
    """Open a webcam or video file, raising :class:`SourceError` on failure.

    When ``backend="auto"`` on Windows the default backend is tried first and
    DirectShow is used as a fallback, which is more reliable for webcams.
    """
    if isinstance(source, str) and not _is_url(source) and not Path(source).exists():
        raise SourceError(f"Video file not found: '{source}'")

    attempts: list[str] = ["auto"]
    if backend.lower() != "auto":
        attempts = [backend]
    elif sys.platform.startswith("win"):
        attempts = list(_WINDOWS_BACKENDS)

    capture: Optional[cv2.VideoCapture] = None
    for name in attempts:
        capture = _open_capture(source, name)
        if capture.isOpened():
            break
        capture.release()
        capture = None

    if capture is None:
        kind = f"camera index {source}" if isinstance(source, int) else f"'{source}'"
        raise SourceError(f"Could not open {kind}. Check that it exists and is not in use.")

    if width and height:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        capture.set(cv2.CAP_PROP_FPS, fps)

    return capture


def read_first_frame(capture: cv2.VideoCapture, source: Source) -> Tuple[bool, "object"]:
    """Grab a frame to verify the source really produces images."""
    ok, frame = capture.read()
    if not ok or frame is None:
        kind = f"camera index {source}" if isinstance(source, int) else f"'{source}'"
        raise SourceError(f"Opened {kind} but could not read any frame from it.")
    return ok, frame


def open_writer(path: str, fps: float, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    """Create a :class:`cv2.VideoWriter`, trying a few codecs for compatibility."""
    output = Path(path)
    if output.parent and str(output.parent) not in ("", "."):
        output.parent.mkdir(parents=True, exist_ok=True)

    codecs = ["mp4v", "MJPG"] if output.suffix.lower() in (".mp4", ".m4v", ".mov") else ["MJPG", "mp4v"]
    for codec in codecs:
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), max(fps, 1.0), size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def describe_source(capture: cv2.VideoCapture) -> dict:
    """Return a small dict of useful stream properties."""
    return {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
