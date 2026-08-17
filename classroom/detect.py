"""Phase 6 - object evidence with mandatory abstention.

The measured phone footprint in this corpus is roughly 30x20 px at a *near*
seat, motion-blurred, at 0.08 bits per pixel. Published classroom-cheating CCTV
results top out near 51% accuracy. A detector that answers confidently at this
scale is wrong more often than it is useful.

So abstention is enforced here, structurally, rather than left as a threshold
someone can quietly lower: any detection whose *source-frame* footprint falls
below the validated floor is emitted as `insufficient` and never as a class
label. The footprint is measured after mapping back through the crop transform,
so magnifying a crop can never make an object look big enough to qualify.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .crops import CropTransform

# Emitted instead of a class label when the source pixels cannot support one.
INSUFFICIENT = "insufficient"
CLASSES = ("phone_like", "paper_like", "other", "person")


@dataclass
class Detection:
    cls: str
    confidence: float
    box: tuple[float, float, float, float]   # frame coordinates
    px_area: float                           # source-frame footprint
    abstained: bool
    t: float = 0.0

    @property
    def usable(self) -> bool:
        return not self.abstained and self.cls in CLASSES


def apply_abstention(
    detections: list[Detection], min_px_area: float
) -> list[Detection]:
    """Downgrade any detection below the pixel floor to `insufficient`."""
    out = []
    for d in detections:
        if d.px_area < min_px_area:
            out.append(Detection(INSUFFICIENT, d.confidence, d.box, d.px_area, True, d.t))
        else:
            out.append(d)
    return out


def temporal_confirm(
    per_frame: list[list[Detection]], required: int, iou_threshold: float = 0.3
) -> list[Detection]:
    """Keep only objects that persist across adjacent frames.

    A single-frame detection at this resolution is far more likely to be a
    compression artefact than an object. Requiring the same class in roughly the
    same place across consecutive frames is the cheapest available filter and
    costs no extra inference.
    """
    confirmed: list[Detection] = []
    for index, frame_dets in enumerate(per_frame):
        for det in frame_dets:
            if det.abstained:
                continue
            run = 1
            for other in per_frame[index + 1 :]:
                match = next(
                    (o for o in other
                     if o.cls == det.cls and iou(o.box, det.box) >= iou_threshold),
                    None,
                )
                if match is None:
                    break
                run += 1
                if run >= required:
                    break
            if run >= required and not any(
                c.cls == det.cls and iou(c.box, det.box) >= iou_threshold
                for c in confirmed
            ):
                confirmed.append(det)
    return confirmed


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def object_evidence(detections: list[Detection]) -> float:
    """Scalar object contribution for the priority score.

    Abstained detections contribute nothing -- they are neither evidence for nor
    against, and folding them in either direction would misrepresent what the
    footage supports.
    """
    usable = [d for d in detections if d.usable and d.cls in ("phone_like", "paper_like")]
    if not usable:
        return 0.0
    weights = {"phone_like": 1.0, "paper_like": 0.4}
    return float(max(weights.get(d.cls, 0.0) * d.confidence for d in usable))


def decode_boxes(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    raw_labels: np.ndarray,
    transform: CropTransform,
    class_names: list[str],
    confidence: float,
    t: float = 0.0,
) -> list[Detection]:
    """Map detector output from input coordinates back to the source frame."""
    out: list[Detection] = []
    for box, score, label in zip(raw_boxes, raw_scores, raw_labels):
        if float(score) < confidence:
            continue
        frame_box = transform.to_frame(tuple(float(v) for v in box[:4]))
        width = max(frame_box[2] - frame_box[0], 0.0)
        height = max(frame_box[3] - frame_box[1], 0.0)
        index = int(label)
        name = class_names[index] if 0 <= index < len(class_names) else "other"
        out.append(
            Detection(name, float(score), frame_box, width * height, False, t)
        )
    return out


class ONNXDetector:
    """D-FINE (or any ONNX detector) over seat crops.

    Loaded lazily so the geometry and abstention rules above are testable
    without weights present.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str],
        input_size: int = 640,
        providers: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.input_size = input_size
        self.providers = providers
        self._session = None

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"detector weights not found: {self.model_path}. "
                f"Export D-FINE to ONNX and place it there."
            )
        providers = self.providers or [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)

    def __call__(
        self, crop: np.ndarray, transform: CropTransform, confidence: float, t: float = 0.0
    ) -> list[Detection]:
        self.load()
        blob = crop.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        inputs = {self._session.get_inputs()[0].name: blob}
        boxes, scores, labels = self._session.run(None, inputs)[:3]
        return decode_boxes(
            np.asarray(boxes).reshape(-1, 4),
            np.asarray(scores).reshape(-1),
            np.asarray(labels).reshape(-1),
            transform, self.class_names, confidence, t,
        )
