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


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Mapping from the stock COCO vocabulary onto this system's classes. Used only
# before fine-tuning: a COCO "cell phone" is a large, clear handheld phone, not
# a 30x20 px object on distant CCTV, so anything routed through here is still
# subject to the abstention floor.
COCO_TO_CLASS = {
    "person": "person",
    "cell phone": "phone_like",
    "book": "paper_like",
    "laptop": "other",
    "remote": "other",
    "mouse": "other",
    "keyboard": "other",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_detr(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    transform: CropTransform,
    class_names: list[str],
    confidence: float,
    t: float = 0.0,
    class_map: dict[str, str] | None = None,
) -> list[Detection]:
    """Decode D-FINE / DETR-style output into frame-coordinate detections.

    `logits` is (queries, classes) pre-sigmoid -- these heads are trained with
    focal loss, so each class is an independent sigmoid rather than a softmax
    over classes, and there is no background class to discard.

    `pred_boxes` is (queries, 4) as normalised centre-x, centre-y, width,
    height relative to the network input.
    """
    logits = np.asarray(logits).reshape(-1, np.asarray(logits).shape[-1])
    boxes = np.asarray(pred_boxes).reshape(-1, 4)

    scores = _sigmoid(logits)
    best = scores.argmax(axis=1)
    best_score = scores[np.arange(len(scores)), best]

    size = transform.input_size
    out: list[Detection] = []
    for index in np.nonzero(best_score >= confidence)[0]:
        cx, cy, w, h = boxes[index]
        input_box = (
            (cx - w / 2) * size, (cy - h / 2) * size,
            (cx + w / 2) * size, (cy + h / 2) * size,
        )
        frame_box = transform.to_frame(input_box)
        width = max(frame_box[2] - frame_box[0], 0.0)
        height = max(frame_box[3] - frame_box[1], 0.0)

        label = int(best[index])
        raw = class_names[label] if 0 <= label < len(class_names) else "other"
        name = (class_map or {}).get(raw, raw) if class_map else raw
        out.append(
            Detection(name, float(best_score[index]), frame_box,
                      width * height, False, t)
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
        class_names: list[str] | None = None,
        input_size: int = 640,
        providers: list[str] | None = None,
        class_map: dict[str, str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.class_names = class_names if class_names is not None else COCO80
        self.input_size = input_size
        self.providers = providers
        self.class_map = class_map if class_map is not None else COCO_TO_CLASS
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
        # BGR from OpenCV -> RGB, CHW, scaled to [0, 1]. D-FINE's processor does
        # not apply ImageNet mean/std normalisation.
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        logits, pred_boxes = self._session.run(
            None, {self._session.get_inputs()[0].name: blob}
        )[:2]
        return decode_detr(
            logits[0], pred_boxes[0], transform, self.class_names,
            confidence, t, self.class_map,
        )
