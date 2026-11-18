"""Real-time object detection and tracking.

Examples
--------
    python main.py --source 0
    python main.py --source "sample.mp4"
    python main.py --source "sample.mp4" --output "result.mp4" --tracker botsort.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence

import cv2

from detector import BUILTIN_TRACKERS, Detector, ModelError
from render import draw_detections, draw_help, draw_panel
from video import SourceError, describe_source, open_capture, open_writer, parse_source, read_first_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Detect and track objects in a webcam or video file using YOLO + ByteTrack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="0", help="Webcam index (0) or path/URL of a video file")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO weights (official name or local .pt path)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help=f"Tracker config: {', '.join(BUILTIN_TRACKERS)}")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, 0 / cuda:0")
    parser.add_argument("--classes", default="", help="Only track these classes, e.g. 'person,car' or '0,2'")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per frame")
    parser.add_argument("--half", action="store_true", help="FP16 inference (CUDA only)")
    parser.add_argument("--color-by", choices=("id", "class"), default="id", help="Colour boxes by track id or class")
    parser.add_argument("--output", default="", help="Optional path for the annotated output video")
    parser.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width (cameras)")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height (cameras)")
    parser.add_argument("--fps", type=float, default=0, help="Requested capture FPS (cameras, 0 = default)")
    parser.add_argument("--backend", default="auto", choices=("auto", "any", "dshow", "msmf"), help="Capture backend")
    parser.add_argument("--list-classes", action="store_true", help="Print the model's class names and exit")
    return parser


def _parse_class_filter(raw: str) -> List[str]:
    return [part for part in str(raw).replace(";", ",").split(",") if part.strip()]


def run(args: argparse.Namespace) -> int:
    source = parse_source(args.source)

    detector = Detector(
        model=args.model,
        device=args.device,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        half=args.half,
    )

    capture = None
    writer = None
    show = not args.no_show
    exit_code = 0
    try:
        # Validate the source first: cheap, and avoids a needless weight download.
        capture = open_capture(
            source,
            backend=args.backend,
            width=args.width or None,
            height=args.height or None,
            fps=args.fps or None,
        )
        ok, frame = read_first_frame(capture, source)
        if not ok:
            return 1

        info = describe_source(capture)
        print(
            f"Source  : {source} | {info['width']}x{info['height']} @ {info['fps']:.1f} FPS"
            + (f" | {info['frames']} frames" if info["frames"] > 0 else "")
        )

        detector.load()

        if args.list_classes:
            for index, name in sorted(detector.class_names.items()):
                print(f"{index:3d}  {name}")
            return 0

        if args.classes:
            detector.set_classes(_parse_class_filter(args.classes))

        print(f"Model   : {args.model} ({detector.device})")
        print(f"Tracker : {args.tracker}")

        if args.output:
            size = (frame.shape[1], frame.shape[0])
            writer = open_writer(args.output, info["fps"] if info["fps"] > 0 else 30.0, size)
            if writer is None:
                print(f"Warning: could not open writer for '{args.output}'.", file=sys.stderr)
            else:
                print(f"Output  : {args.output}")

        frames = 0
        fps_smooth = 0.0
        previous = time.perf_counter()

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            detections = detector.track(frame)
            tracked = sum(1 for det in detections if det.track_id is not None)

            now = time.perf_counter()
            delta = max(now - previous, 1e-6)
            previous = now
            fps_smooth = (1.0 / delta) if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * (1.0 / delta)

            draw_detections(frame, detections, color_by=args.color_by)
            draw_panel(
                frame,
                fps=fps_smooth,
                tracked=tracked,
                detected=len(detections),
                extra=[f"Model: {args.model}", f"Tracker: {args.tracker}"],
            )
            if show:
                draw_help(frame)

            if writer is not None:
                writer.write(frame)

            if show:
                try:
                    cv2.imshow("Object Detection & Tracking", frame)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
                except cv2.error as exc:
                    print(f"Warning: preview window unavailable ({exc}). Continuing headless.", file=sys.stderr)
                    show = False

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except SourceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (SourceError, ModelError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
