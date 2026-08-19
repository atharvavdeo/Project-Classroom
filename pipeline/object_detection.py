"""The object detector A/B over one frozen crop manifest (PRD 10).

Four named configurations run on identical crop rows:

    rf_detr_native    RF-DETR, one explicit resize of the original crop
    dfine_native      D-FINE, the same
    rf_detr_sahi      RF-DETR with candidate-crop slicing
    dfine_sahi        D-FINE with the same slicing

PRD 10 requires the raw and merged results of each to be preserved separately
and forbids one configuration overwriting another's artifacts. That is enforced
structurally: a `DetectorResult` owns its own detections and its own directory,
and nothing in this module writes across configurations.

### Abstention is a class, not a missing row

PRD 10's taxonomy ends with `insufficient_visual_evidence`, and it is a real
output rather than the absence of one. A 22x16 px crop magnified to 640x640
produces confident-looking boxes from interpolated pixels; the honest label for
a phone-sized object in it is that the source cannot support the call. So the
abstention is applied on the crop's **native** footprint, before the detector's
score is consulted at all, and it is recorded as a detection row so a reviewer
can count how much of the recording the camera simply cannot answer for.

### Hard negatives are the measurement

PRD 10: "a detector calling mouse/keyboard/paper edge a phone is a failure
cluster, not a mere false score." `HARD_NEGATIVES` names them, and
`failure_clusters` groups confusions by (predicted, actual) so the report shows
*what* a detector gets wrong rather than only how often.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

# Provisional taxonomy (PRD 10). Requires product-owner confirmation before any
# detector selection or evaluation may be made -- see doubt.md.
PHONE = "phone"
CHIT = "secondary_paper_chit"
ANSWER_SHEET = "answer_sheet"
CALCULATOR = "calculator"
MOUSE = "mouse"
KEYBOARD = "keyboard"
HAND = "hand"
BAG = "bag"
MONITOR = "monitor_or_bezel"
STATIONERY = "stationery"
# Added beyond PRD 10's provisional list, and flagged for owner approval in
# doubt.md. An open textbook or notebook on an examination desk is
# unauthorised material in its own right; it is neither a chit nor nothing.
# The name states what is visible and claims nothing about permission.
BOOK_OR_NOTEBOOK = "book_or_notebook_like_object"
UNKNOWN = "unknown_object"
INSUFFICIENT = "insufficient_visual_evidence"

TAXONOMY = (PHONE, CHIT, ANSWER_SHEET, CALCULATOR, MOUSE, KEYBOARD, HAND, BAG,
            MONITOR, STATIONERY, BOOK_OR_NOTEBOOK, UNKNOWN, INSUFFICIENT)

# The two classes the product exists to surface. Reported separately, always:
# PRD 15 requires "Phone and chit results must be separate", because they fail
# in different ways -- a phone is a rigid bright rectangle, a chit is a soft
# low-contrast one, and a single averaged number hides whichever is worse.
# Reported separately, always. `book_or_notebook_like_object` joins them
# because an open book is unauthorised material a reviewer wants surfaced, and
# folding it into a total would hide both it and the chit count.
TARGET_CLASSES = (PHONE, CHIT, BOOK_OR_NOTEBOOK)

# Objects that look like a phone at 30x20 px on poor CCTV. A detector calling
# one of these a phone is the failure cluster PRD 10 names.
HARD_NEGATIVES = (MOUSE, KEYBOARD, CALCULATOR, ANSWER_SHEET, MONITOR, STATIONERY)

RF_DETR_NATIVE = "rf_detr_native"
DFINE_NATIVE = "dfine_native"
RF_DETR_SAHI = "rf_detr_sahi"
DFINE_SAHI = "dfine_sahi"
OBJECT_CONFIGS = (RF_DETR_NATIVE, DFINE_NATIVE, RF_DETR_SAHI, DFINE_SAHI)


class TaxonomyError(ValueError):
    """A detector emitted a class outside the approved taxonomy."""


@dataclass
class DetectConfig:
    # Score floor. Deliberately low: PRD 10 wants singletons retained and judged
    # by temporal confirmation, not thresholded away before they can be grouped.
    confidence: float = 0.25

    # Native source footprint, in px on the shorter side, below which no object
    # call is made. A phone measured on this corpus is ~30x20 px; under this a
    # crop cannot hold one plus the surround needed to tell it from a mouse.
    abstain_below_native_px: int = 32

    # Detections whose own source-projected footprint is under this are
    # abstentions too, even in a large crop: the object itself is too small.
    abstain_below_object_px: float = 120.0

    input_size: int = 640


@dataclass
class ObjectDetection:
    """One detection, in source coordinates, with its provenance."""

    crop_id: str
    config_id: str
    pts_ms: float
    cls: str
    confidence: float
    # Source-frame box. Never crop-space: a box left in crop coordinates is
    # wrong by the crop offset and still draws a plausible-looking overlay.
    x1: float
    y1: float
    x2: float
    y2: float
    native_px: float = 0.0
    abstained: bool = False
    abstain_reason: str = ""
    slice_index: int | None = None     # which SAHI slice produced it, if any
    merged_from: int = 1               # how many raw boxes merged into this
    seat_state: str = "unattributed"
    track_id: int | None = None
    event_id: str = ""

    def __post_init__(self) -> None:
        if self.cls not in TAXONOMY:
            raise TaxonomyError(
                f"{self.config_id} emitted class {self.cls!r}, which is not in "
                f"the provisional taxonomy. An unmapped class must be mapped or "
                f"routed to {UNKNOWN!r}, never passed through (PRD 10).")

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        return max(self.x2 - self.x1, 0.0) * max(self.y2 - self.y1, 0.0)

    @property
    def usable(self) -> bool:
        return not self.abstained and self.cls != INSUFFICIENT

    def to_row(self) -> dict:
        return {**asdict(self), "area_px": round(self.area, 1),
                "usable": self.usable}


def abstention(crop, config_id: str, reason: str) -> ObjectDetection:
    """An explicit `insufficient_visual_evidence` row.

    Emitted instead of, not alongside, a detection. PRD 10 requires the
    abstention to be output where the resolution does not support the object,
    and an abstention that is merely a missing row is indistinguishable from a
    crop nobody processed.
    """
    return ObjectDetection(
        crop_id=crop.crop_id, config_id=config_id, pts_ms=crop.pts_ms,
        cls=INSUFFICIENT, confidence=0.0,
        x1=crop.x1, y1=crop.y1, x2=crop.x2, y2=crop.y2,
        native_px=round(crop.native_px, 1), abstained=True,
        abstain_reason=reason, seat_state=crop.seat_state,
        track_id=crop.track_id, event_id=crop.event_id)


@dataclass
class DetectorResult:
    """One configuration's output. Owns its own rows; never shared."""

    config_id: str
    state: str = "available"
    crop_digest: str = ""
    raw: list[ObjectDetection] = field(default_factory=list)
    merged: list[ObjectDetection] = field(default_factory=list)
    crops_seen: int = 0
    crops_abstained: int = 0
    slices_run: int = 0
    seconds: float = 0.0
    peak_vram_mb: float | None = None
    checkpoint_sha256: str = ""
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.state == "available"

    def metrics(self) -> dict:
        if not self.available:
            return {"config_id": self.config_id, "state": self.state,
                    "reason": self.reason, "crop_digest": self.crop_digest,
                    "note": "this configuration did not run. Its row is retained "
                            "in the comparison and never filled from another."}
        by_class: dict[str, int] = {}
        for detection in self.merged:
            by_class[detection.cls] = by_class.get(detection.cls, 0) + 1
        usable = [d for d in self.merged if d.usable]
        return {
            "config_id": self.config_id,
            "state": self.state,
            "crop_digest": self.crop_digest,
            "checkpoint_sha256": self.checkpoint_sha256,
            "crops_seen": self.crops_seen,
            "crops_abstained": self.crops_abstained,
            "abstention_rate": round(self.crops_abstained / self.crops_seen, 4)
            if self.crops_seen else 0.0,
            "slices_run": self.slices_run,
            "raw_detections": len(self.raw),
            "merged_detections": len(self.merged),
            "usable_detections": len(usable),
            "by_class": dict(sorted(by_class.items())),
            "phone": by_class.get(PHONE, 0),
            "secondary_paper_chit": by_class.get(CHIT, 0),
            "hard_negative_detections": sum(by_class.get(c, 0)
                                            for c in HARD_NEGATIVES),
            "seconds": round(self.seconds, 3),
            "crops_per_second": round(self.crops_seen / self.seconds, 2)
            if self.seconds > 0 else None,
            "peak_vram_mb": self.peak_vram_mb,
        }


def detect(crops: list, runner, config_id: str,
           cfg: DetectConfig | None = None, crop_digest: str = "",
           checkpoint_sha256: str = "") -> DetectorResult:
    """Run one detector configuration over the frozen crop manifest.

    `runner` maps an `ObjectCrop` to an iterable of
    `(cls, confidence, x1, y1, x2, y2)` in **source** coordinates. Returning
    source coordinates is the runner's job because only the runner knows its own
    slice offsets; making it the caller's job is how boxes end up off by a
    letterbox pad.
    """
    cfg = cfg or DetectConfig()
    result = DetectorResult(config_id=config_id, crop_digest=crop_digest,
                            checkpoint_sha256=checkpoint_sha256)
    started = time.perf_counter()

    for crop in crops:
        result.crops_seen += 1

        # The resolution check runs before the model, not after. A detector
        # never sees a crop it cannot answer for, so it cannot produce a
        # confident box from interpolated pixels that a later filter must argue
        # with.
        if min(crop.native_w, crop.native_h) < cfg.abstain_below_native_px:
            result.crops_abstained += 1
            row = abstention(
                crop, config_id,
                f"native footprint {crop.native_w:.0f}x{crop.native_h:.0f} px is "
                f"under the {cfg.abstain_below_native_px} px floor; magnifying "
                f"it {crop.magnification:.1f}x adds interpolation, not detail")
            result.raw.append(row)
            result.merged.append(row)
            continue

        try:
            proposals = list(runner(crop))
        except Exception as exc:                        # noqa: BLE001
            result.state = "launch_failed"
            result.reason = f"{type(exc).__name__}: {exc}"
            break

        kept = []
        for cls, confidence, x1, y1, x2, y2 in proposals:
            if confidence < cfg.confidence:
                continue
            detection = ObjectDetection(
                crop_id=crop.crop_id, config_id=config_id, pts_ms=crop.pts_ms,
                cls=cls, confidence=round(float(confidence), 4),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                native_px=round(crop.native_px, 1),
                seat_state=crop.seat_state, track_id=crop.track_id,
                event_id=crop.event_id)
            # An object whose own source footprint is tiny is an abstention even
            # inside a large crop: the crop had the resolution, the object did
            # not occupy enough of it.
            if detection.area < cfg.abstain_below_object_px:
                detection.abstained = True
                detection.abstain_reason = (
                    f"object footprint {detection.area:.0f} px is under the "
                    f"{cfg.abstain_below_object_px:.0f} px floor for a call")
            kept.append(detection)

        result.raw.extend(kept)
        result.merged.extend(kept)

    result.seconds = time.perf_counter() - started
    return result


def unavailable(config_id: str, state: str, reason: str,
                crop_digest: str = "") -> DetectorResult:
    """A detector configuration that could not run, kept in the table (PRD 5)."""
    return DetectorResult(config_id=config_id, state=state, reason=reason,
                          crop_digest=crop_digest)


def failure_clusters(result: DetectorResult, truth_of=None) -> dict:
    """Group confusions by (predicted, actual), per PRD 10.

    Without a truth source this reports the *shape* of the output -- how many
    calls landed on hard-negative classes -- and says plainly that it is not a
    confusion matrix. Inventing an accuracy number from unlabelled data is the
    thing this whole pipeline is arranged to avoid.
    """
    if truth_of is None:
        by_class: dict[str, int] = {}
        for detection in result.merged:
            by_class[detection.cls] = by_class.get(detection.cls, 0) + 1
        return {
            "status": "unlabelled",
            "detail": "no reference labels were supplied, so no confusion can "
                      "be computed. The class mix below is the shape of the "
                      "output, not its accuracy.",
            "by_class": dict(sorted(by_class.items())),
            "hard_negative_calls": {c: by_class.get(c, 0) for c in HARD_NEGATIVES
                                    if by_class.get(c)},
        }

    clusters: dict[str, int] = {}
    for detection in result.merged:
        actual = truth_of(detection)
        if actual is None or actual == detection.cls:
            continue
        clusters[f"{detection.cls}<-{actual}"] = \
            clusters.get(f"{detection.cls}<-{actual}", 0) + 1
    phone_from_negatives = sum(
        count for key, count in clusters.items()
        if key.startswith(f"{PHONE}<-") and key.split("<-")[1] in HARD_NEGATIVES)
    return {
        "status": "labelled",
        "clusters": dict(sorted(clusters.items(), key=lambda kv: -kv[1])),
        "phone_called_on_hard_negative": phone_from_negatives,
        "note": "a phone called on a mouse, keyboard or paper edge is a failure "
                "cluster to fix, not a false positive to average away (PRD 10).",
    }


def compare(results: list[DetectorResult], crops: list,
            crop_digest: str, taxonomy_approved: bool = False) -> dict:
    """The matched detector comparison (PRD 10, PRD 15)."""
    mismatched = [r.config_id for r in results
                  if r.crop_digest and r.crop_digest != crop_digest]

    selection = (
        "none. PRD 3 blocks detector selection and evaluation until the product "
        "owner approves the taxonomy, and PRD 15 places selection in stage 12 "
        "against the reference manifest.")
    if taxonomy_approved:
        selection = (
            "none yet. The taxonomy is approved, but selection still requires "
            "the reference manifest at stage 12; no configuration may be called "
            "best from unlabelled output.")

    return {
        "crop_digest": crop_digest,
        "crop_rows": len(crops),
        "crop_equivalence": "verified" if not mismatched else "VIOLATED",
        "mismatched_configs": mismatched,
        "equivalence_note":
            "every configuration below processed byte-identical crop rows, so a "
            "difference in their output is attributable to the detector and its "
            "slicing (PRD 4)" if not mismatched else
            "configurations disagreed on the crop manifest. Their outputs are "
            "NOT comparable and no selection may be made from them.",
        "taxonomy_approved": taxonomy_approved,
        "configurations": [r.metrics() for r in results],
        "failure_clusters": {r.config_id: failure_clusters(r) for r in results
                             if r.available},
        "available": [r.config_id for r in results if r.available],
        "unavailable": [{"config_id": r.config_id, "state": r.state,
                         "reason": r.reason} for r in results if not r.available],
        "selection": selection,
    }


# --------------------------------------------------------------- runners --

# COCO name -> provisional taxonomy. Shared by both detector families so the
# mapping cannot differ between the arms of the comparison -- if it did, a
# class-count difference would be the mapping's doing rather than the model's.
COCO_TO_TAXONOMY = {
    "cell phone": PHONE,
    # COCO `book` does NOT mean `secondary_paper_chit`, and mapping it there was
    # a category error with real consequences. Rendering the top ten "chit"
    # detections on video 03 showed what the class actually fired on: a red
    # snack packet at 121x63 px and a yellow notebook at 116x69 px. Every chit
    # detection this pipeline had produced -- 232 rows on video 03, 27 on video
    # 01 -- was a notebook or a wrapper.
    #
    # A book or notebook on an examination desk is the candidate's own
    # stationery, quite possibly the answer sheet. Labelling it as a smuggled
    # note manufactures evidence against the person using it. COCO cannot tell
    # an answer sheet from a notebook from a chit, so the honest mapping is
    # `unknown_object`: the detector found *something paper-like* and the class
    # it assigned does not support any of the three readings.
    #
    # It is not `unknown_object` either. An open textbook or notebook on an
    # examination desk is unauthorised material in its own right, and burying it
    # under `unknown_object` discards a real observation. It gets its own
    # observational class, which says exactly what the detector saw and nothing
    # about what it means.
    #
    # A real chit claim needs a detector fine-tuned on labelled chits. Until
    # then chit recall on this pipeline is zero and chit precision is undefined.
    "book": BOOK_OR_NOTEBOOK,
    "mouse": MOUSE, "keyboard": KEYBOARD, "laptop": MONITOR, "tv": MONITOR,
    "remote": STATIONERY, "scissors": STATIONERY,
    "backpack": BAG, "handbag": BAG, "suitcase": BAG,
}


def rfdetr_runner(variant: str = "nano", crop_cfg=None, confidence: float = 0.25,
                  frame_reader=None, device: str = "cpu"):
    """Adapter over Roboflow RF-DETR (Apache-2.0).

    Runs on the same `ObjectCrop` rows as the D-FINE arm and returns boxes in
    **source** coordinates, so PRD 4's identical-input requirement holds across
    detector families as well as within one.

    RF-DETR takes the crop as an image rather than a letterboxed 640x640 canvas
    -- it does its own resizing internally. The crop is therefore cut at native
    resolution and the returned boxes are offset by the crop origin, which is
    the whole of the projection: no letterbox pad to undo, unlike the ONNX arm.
    """
    import numpy as _np
    import rfdetr

    from .crops import CropConfig

    crop_cfg = crop_cfg or CropConfig()
    classes = {
        "nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase,
        "large": rfdetr.RFDETRLarge,
    }
    if variant not in classes:
        raise ValueError(f"unknown RF-DETR variant {variant!r}; "
                         f"expected one of {sorted(classes)}")
    model = classes[variant]()

    from rfdetr.assets.coco_classes import COCO_CLASSES
    names = (COCO_CLASSES if isinstance(COCO_CLASSES, dict)
             else {i: n for i, n in enumerate(COCO_CLASSES)})

    def run(crop):
        if frame_reader is None:
            raise RuntimeError(
                "rfdetr_runner needs a frame_reader mapping a crop to its "
                "source frame; without one there is nothing to detect on")
        frame = frame_reader(crop)
        if frame is None:
            return []
        height, width = frame.shape[:2]
        x1, y1 = int(max(crop.x1, 0)), int(max(crop.y1, 0))
        x2, y2 = int(min(crop.x2, width)), int(min(crop.y2, height))
        if x2 <= x1 or y2 <= y1:
            return []
        patch = frame[y1:y2, x1:x2][:, :, ::-1]     # BGR -> RGB

        detections = model.predict(_np.ascontiguousarray(patch),
                                   threshold=confidence)
        out = []
        for box, score, class_id in zip(detections.xyxy, detections.confidence,
                                        detections.class_id):
            raw = names.get(int(class_id), "unknown")
            if raw == "person":
                continue
            cls = COCO_TO_TAXONOMY.get(raw, UNKNOWN)
            # Offset by the crop origin. A box left in crop space is wrong by
            # exactly (x1, y1) and still draws a plausible overlay.
            out.append((cls, float(score), float(box[0]) + x1, float(box[1]) + y1,
                        float(box[2]) + x1, float(box[3]) + y1))
        return out

    return run


def dfine_runner(model_path, crop_cfg=None, providers=None,
                 confidence: float = 0.25, frame_reader=None,
                 class_map: dict | None = None):
    """Adapter over the measured `classroom.detect.ONNXDetector` (D-FINE).

    The stock COCO head is mapped onto the provisional taxonomy. The mapping is
    deliberately narrow: COCO's `cell phone` is a large clear handheld phone,
    nothing like a 30x20 px object on this footage, so everything routed through
    it still meets the abstention floors above.
    """
    from classroom.detect import ONNXDetector

    from .crops import CropConfig, extract

    crop_cfg = crop_cfg or CropConfig()
    # `class_map={}` keeps the raw COCO names. The legacy default maps them onto
    # `phone_like`/`paper_like`/`other`, which would fold every hard negative
    # into one bucket and make the failure clusters PRD 10 asks for unmeasurable.
    detector = ONNXDetector(str(model_path), providers=providers,
                            input_size=crop_cfg.input_size, class_map={})
    mapping = class_map or COCO_TO_TAXONOMY

    def run(crop):
        if frame_reader is None:
            raise RuntimeError(
                "dfine_runner needs a frame_reader mapping a crop to its source "
                "frame; without one there is nothing to detect on")
        frame = frame_reader(crop)
        if frame is None:
            return []
        canvas, transform = extract(frame, crop, crop_cfg)
        out = []
        for detection in detector(canvas, transform, confidence, crop.pts_ms):
            cls = mapping.get(detection.cls, UNKNOWN)
            # `person` mapped to HAND above is a placeholder for a person part;
            # a whole person is not an object observation and is dropped here
            # rather than counted as one.
            if detection.cls == "person":
                continue
            out.append((cls, detection.confidence, *detection.box))
        return out

    return run
