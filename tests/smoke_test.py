"""Smoke tests for the detection/tracking pipeline.

Run standalone:      python tests/smoke_test.py
Run with pytest:     python -m pytest tests/smoke_test.py -q
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from detector import Detector, ModelError  # noqa: E402
from video import SourceError, open_capture, open_writer, parse_source  # noqa: E402

SAMPLE = ROOT / "sample.mp4"
MODEL = "yolo26n.pt"


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source(" 2 ") == 2
    assert parse_source("clip.mp4") == "clip.mp4"
    assert parse_source("https://host/stream.mp4").startswith("https://")
    for bad in ("", "-1"):
        try:
            parse_source(bad)
        except SourceError:
            pass
        else:
            raise AssertionError(f"expected SourceError for {bad!r}")


def test_invalid_source_raises():
    try:
        open_capture("definitely_missing_file.mp4")
    except SourceError:
        return
    raise AssertionError("expected SourceError for a missing file")


def test_missing_model_raises():
    detector = Detector(model="missing_dir/missing_model_file.pt")
    try:
        detector.load()
    except ModelError:
        return
    raise AssertionError("expected ModelError for a missing .pt file")


def test_unknown_tracker_raises():
    try:
        Detector(model=MODEL, tracker="nope.yaml").load()
    except ModelError:
        return
    raise AssertionError("expected ModelError for an unknown tracker")


def test_model_inference_runs():
    detector = Detector(model=MODEL, device="cpu", conf=0.25).load()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.track(frame)  # must not raise even with zero detections
    assert detector.class_names, "model should expose class names"


def test_synthetic_video_written():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.mp4"
        writer = open_writer(str(path), 30.0, (320, 240))
        assert writer is not None
        for i in range(30):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[100:200, 20 + i * 5 : 120 + i * 5] = (0, 255, 0)
            writer.write(frame)
        writer.release()
        assert path.exists() and path.stat().st_size > 0


def test_tracking_on_sample_video():
    if not SAMPLE.exists():
        print("  SKIP  sample.mp4 not present")
        return
    detector = Detector(model=MODEL, device="cpu", conf=0.25, tracker="bytetrack.yaml").load()
    capture = open_capture(str(SAMPLE))
    ok, frame = capture.read()
    assert ok
    ids_first, total = set(), 0
    for _ in range(30):
        ok, frame = capture.read()
        if not ok:
            break
        for det in detector.track(frame):
            total += 1
            if det.track_id is not None:
                ids_first.add(det.track_id)
    capture.release()
    assert total > 0, "expected detections on the sample video"
    assert ids_first, "expected tracking ids on the sample video"


def test_cli_end_to_end():
    if not SAMPLE.exists():
        print("  SKIP  sample.mp4 not present")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        cmd = [
            sys.executable, str(ROOT / "main.py"),
            "--source", str(SAMPLE),
            "--no-show", "--max-frames", "20", "--output", str(out),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, f"CLI failed:\n{proc.stdout}\n{proc.stderr}"
        assert out.exists() and out.stat().st_size > 0, "output video was not written"


def test_cli_reports_bad_source():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--source", "nope.mp4", "--no-show"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode != 0
    assert "Error" in (proc.stdout + proc.stderr)


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
