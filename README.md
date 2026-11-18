# AI Object Detection and Tracking (CLI)

Real-time multi-object detection and tracking for a **webcam** or a **video file**,
using a pretrained YOLO model and the ByteTrack tracker. Bounding boxes, class
labels, confidence scores, tracking IDs, live FPS and an object counter are drawn
on every frame.

![stack](https://img.shields.io/badge/python-3.10%2B-blue) ![stack](https://img.shields.io/badge/OpenCV-4.10%2B-green) ![stack](https://img.shields.io/badge/YOLO26-Ultralytics-orange)

This repository contains the standalone **Object Detection and Tracking** project (CLI).
The School Security Monitor desktop application is maintained separately and is not included here.

## 1. Technology selection

| Concern | Choice | Why |
|---|---|---|
| Detector | **YOLO26n** (`ultralytics` 8.4.183) | Latest stable Ultralytics release (YOLO27 is still unreleased). Pretrained COCO weights, auto-downloaded, strong CPU accuracy/speed balance (nano variant ≈ 2.5 M params). |
| Tracker | **ByteTrack** (`bytetrack.yaml`, built into Ultralytics) | Integrated into `model.track()` — no extra package, no `lap`/Cython build on Windows. BoT-SORT, OC-SORT, Deep OC-SORT, FastTracker and TrackTrack are also available via `--tracker`. |
| Video I/O & drawing | **OpenCV** (`opencv-python` ≥ 4.10) | `VideoCapture`/`VideoWriter`/`imshow` are the de-facto standard and work identically on Windows 10/11. |
| Runtime | **PyTorch** (CPU wheel by default, CUDA if present) | `device=auto` picks CUDA when available, otherwise CPU. |
| Extra dependencies | **none** | Tracking, NMS and drawing are all provided by the two packages above. |

**Licensing:** the Ultralytics package **and the YOLO26 weights are AGPL-3.0**
(free for open-source/personal use; commercial use requires an Ultralytics
Enterprise licence). If a permissive licence is mandatory, use the Apache-2.0
`rtdetr-l.pt` weights instead: `python main.py --source 0 --model rtdetr-l.pt`.

## 2. Features

- Webcam (`--source 0`) or video file/URL input, with clear errors for missing files,
  unavailable cameras and unreadable streams.
- Multi-object detection with a pretrained YOLO model (COCO, 80 classes).
- **ByteTrack** integration through the integrated `model.track(persist=True)` API —
  IDs stay stable across frames and survive short occlusions.
- Box + `ID n | class 0.87` label overlay, colour-coded per track ID.
- Live **FPS** and **tracked object count** panel.
- CPU by default, automatic CUDA/MPS use when a compatible GPU is present.
- Optional annotated video export (`--output result.mp4`).
- Clean shutdown: camera released, video writer closed, windows destroyed
  (`try/finally`, also on `Ctrl+C`).

## 3. Installation (Windows 10/11, PowerShell)

```powershell
# Recommended: CPU-only PyTorch (much smaller download, ~200 MB vs ~2.5 GB)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt
```

For NVIDIA GPUs, install the matching CUDA build of PyTorch first
(see <https://pytorch.org/get-started/locally/>) and then `requirements.txt`.

## 4. Usage

```powershell
# Webcam
python main.py --source 0

# Video file
python main.py --source "sample.mp4"

# Save the annotated result, only track people and cars, use BoT-SORT
python main.py --source "sample.mp4" --output "result.mp4" --classes "person,bus" --tracker botsort.yaml
```

### Options

| Option | Default | Description |
|---|---|---|
| `--source` | `0` | Camera index, video file or stream URL |
| `--model` | `yolo26n.pt` | Official weight name or local `.pt` path (`yolo26s.pt` for more accuracy) |
| `--tracker` | `bytetrack.yaml` | `bytetrack`/`botsort`/`ocsort`/`deepocsort`/`fasttrack`/`tracktrack` or a custom YAML |
| `--conf` / `--iou` | `0.25` / `0.7` | Detection confidence and NMS IoU thresholds |
| `--imgsz` | `640` | Inference size (320/480 for speed, 960+ for small objects) |
| `--device` | `auto` | `auto`, `cpu`, `0`, `cuda:0` |
| `--classes` | all | Filter by name (`person,car`) or id (`0,2`) |
| `--color-by` | `id` | Colour boxes by track ID or by class |
| `--output` | – | Write the annotated video to this path |
| `--no-show` | off | Run headless (no preview window) |
| `--max-frames` | `0` | Stop after N frames |
| `--list-classes` | – | Print the model's class names and exit |

Press `q` or `Esc` in the preview window to quit.

## 5. Tests

```powershell
python tests\smoke_test.py
```

The suite covers source parsing, invalid sources, missing models, unknown
trackers, model inference, video writing, tracking on `sample.mp4` and a full
CLI run with output export.

## 6. Limitations

- Performance depends on hardware and selected model/image size (CPU-only nano is typical for basic demos).
- ByteTrack keeps IDs for `track_buffer` (30) frames of occlusion; fast, dense
  crowds can still cause ID switches — use `--tracker botsort.yaml` there.
- YOLO26 is NMS-free (`end2end`), so overlapping identical-class objects can
  occasionally be merged.
- The preview window needs a working OpenCV GUI backend; on headless sessions use `--no-show`.
- `sample.mp4` is included as a test asset for the smoke tests.

## 7. Project layout

```
main.py              CLI entry point
detector.py          YOLO model loading + tracking wrapper
video.py             Source parsing/opening, video writing
render.py            Box, label, FPS panel drawing
tests/smoke_test.py  Smoke tests
requirements.txt     Pinned dependencies
sample.mp4           Sample video for tests
```
