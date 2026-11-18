"""Detection and tracking wrapper around Ultralytics YOLO.

The class exposes a very small surface: :meth:`Detector.load` prepares the
model and :meth:`Detector.track` returns a list of :class:`Detection` objects
for a single frame.  Tracking state is kept inside the model, so consecutive
frames automatically keep stable IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

#: Trackers bundled with Ultralytics that can be selected by name.
BUILTIN_TRACKERS = (
    "bytetrack.yaml",
    "botsort.yaml",
    "ocsort.yaml",
    "deepocsort.yaml",
    "fasttrack.yaml",
    "tracktrack.yaml",
)

#: File extensions that must point to an existing local file.
LOCAL_MODEL_SUFFIXES = (".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".xml")


class ModelError(RuntimeError):
    """Raised when the detection model cannot be loaded."""


@dataclass
class Detection:
    """One tracked object in a frame."""

    box: tuple  # (x1, y1, x2, y2) in pixels
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class Detector:
    """Load a YOLO model and run detection with multi-object tracking."""

    def __init__(
        self,
        model: str = "yolo26n.pt",
        device: str = "auto",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        classes: Optional[Sequence[int]] = None,
        max_det: int = 300,
        half: bool = False,
    ) -> None:
        self.model_name = model
        self.device_request = device
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.half = half
        self._class_indices: Optional[List[int]] = list(classes) if classes else None
        self._model = None
        self._names: dict = {}
        self.device: str = "cpu"

    # ------------------------------------------------------------------ setup
    def load(self) -> "Detector":
        """Load the model, resolving the device and class filter."""
        import torch  # imported lazily so `--help` stays fast
        import torchvision  # noqa: F401 - registers NMS ops for packaged builds
        from torchvision.ops import nms as _nms  # noqa: F401
        from ultralytics import YOLO

        self._validate_model_path()
        self._validate_tracker()

        try:
            self._model = YOLO(self.model_name, task="detect")
        except Exception as exc:  # noqa: BLE001 - surfaced as a friendly error
            raise ModelError(
                f"Could not load model '{self.model_name}'. "
                f"Check the file exists or that the weights can be downloaded. ({exc})"
            ) from exc

        self._names = dict(self._model.names)
        self.device = self._resolve_device(torch)
        if self.half and not self.device.startswith("cuda"):
            self.half = False
            print("Note: --half is only available on CUDA; ignoring it on this device.")
        return self

    def _validate_model_path(self) -> None:
        """Fail fast on missing local weight files.

        Plain names such as ``yolo26n.pt`` are official Ultralytics assets that
        are downloaded automatically, so only anything path-like is checked.
        """
        name = str(self.model_name)
        if Path(name).suffix.lower() not in LOCAL_MODEL_SUFFIXES:
            return
        looks_like_path = bool(Path(name).anchor) or "/" in name or "\\" in name
        if looks_like_path and not Path(name).is_file():
            raise ModelError(
                f"Model file not found: '{name}'. "
                "Point --model at an existing .pt file or use a built-in name "
                "such as yolo26n.pt (it is downloaded automatically)."
            )

    def _validate_tracker(self) -> None:
        tracker = str(self.tracker)
        if tracker in BUILTIN_TRACKERS:
            return
        if tracker.endswith(".yaml") and Path(tracker).is_file():
            return
        raise ModelError(
            f"Unknown tracker '{tracker}'. Use one of {', '.join(BUILTIN_TRACKERS)} "
            "or the path to a custom tracker YAML file."
        )

    def _resolve_device(self, torch) -> str:
        request = str(self.device_request).lower()
        if request in ("auto", ""):
            if torch.cuda.is_available():
                return "cuda:0"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if request == "cpu":
            return "cpu"
        if request.isdigit() or request.startswith("cuda"):
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU.")
                return "cpu"
            if request.isdigit():
                return f"cuda:{request}"
            return request if ":" in request else "cuda:0"
        return request

    # --------------------------------------------------------------- class ids
    @property
    def class_names(self) -> dict:
        return self._names

    def resolve_classes(self, names: Sequence[str]) -> List[int]:
        """Translate class names (or numeric strings) into class indices."""
        indices: List[int] = []
        unknown: List[str] = []
        for raw in names:
            key = str(raw).strip()
            if not key:
                continue
            if key.isdigit():
                index = int(key)
                if index in self._names:
                    indices.append(index)
                else:
                    unknown.append(key)
                continue
            for index, name in self._names.items():
                if str(name).lower() == key.lower():
                    indices.append(index)
                    break
            else:
                unknown.append(key)
        if unknown:
            raise ModelError(
                f"Unknown class name(s): {', '.join(unknown)}. "
                f"Valid COCO classes: {', '.join(str(v) for v in self._names.values())}"
            )
        return sorted(set(indices))

    def set_classes(self, names: Sequence[str]) -> List[int]:
        """Restrict detection to the given class names (or numeric ids)."""
        self._class_indices = self.resolve_classes(names)
        return self._class_indices

    # ---------------------------------------------------------------- tracking
    def track(self, frame: np.ndarray) -> List[Detection]:
        """Detect and track objects in a single BGR frame."""
        if self._model is None:
            raise ModelError("Detector.load() must be called before tracking.")
        results = self._model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self._class_indices,
            max_det=self.max_det,
            **({"quantize": 16} if self.half else {}),
            verbose=False,
        )
        if not results:
            return []
        return self._to_detections(results[0])

    def _to_detections(self, result) -> List[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy()
        if boxes.id is not None:
            track_ids = boxes.id.detach().cpu().numpy()
        else:
            track_ids = [None] * len(confidences)

        detections: List[Detection] = []
        for i in range(len(confidences)):
            class_id = int(class_ids[i])
            detections.append(
                Detection(
                    box=(int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])),
                    class_id=class_id,
                    class_name=str(self._names.get(class_id, class_id)),
                    confidence=float(confidences[i]),
                    track_id=None if track_ids[i] is None else int(track_ids[i]),
                )
            )
        return detections
