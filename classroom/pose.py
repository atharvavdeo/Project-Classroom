"""Phase 6 - pose evidence.

RTMO via rtmlib (ONNXRuntime only, no mmcv build). Two constraints from PRD 8.2
are enforced here rather than left to the caller, because both are places where
a plausible-looking number would be actively misleading:

  * Wrist keypoints are masked when the hand drops below the desk line. That is
    exactly the interesting case, and the estimator will still emit a confident
    coordinate for a hand it cannot see. The absence is recorded as evidence
    instead of being imputed.

  * Head pitch is derived from the nose-to-shoulder offset in COCO-17 rather
    than from a separate head-pose model. It is the primary signal for the
    dominant behavioural pattern in this deployment (PRD 4.3), and it survives
    720p where a dedicated head-pose model would not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO-17 keypoint order.
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

KEYPOINT_CONF_MIN = 0.30


@dataclass
class PoseObservation:
    keypoints: np.ndarray          # (17, 3) -> x, y, confidence
    head_pitch: float | None       # >0 means looking down; None if unresolvable
    wrist_masked: bool             # a wrist fell below the desk line
    wrists_below: int              # how many (0, 1 or 2)
    seat_label: str | None = None
    track_id: int | None = None

    def visible(self, index: int) -> bool:
        return bool(self.keypoints[index, 2] >= KEYPOINT_CONF_MIN)


def head_pitch(keypoints: np.ndarray) -> float | None:
    """Downward head inclination, normalised by torso scale.

    Measures how far the nose sits below the shoulder line relative to the
    shoulder width, so it is invariant to how far the candidate is from the
    camera. Returns None when the required points are not visible -- an
    unresolvable pose must not silently read as zero, which would look like a
    confidently upright head.

    Positive means the nose has dropped toward the desk.
    """
    ls, rs, nose = keypoints[L_SHOULDER], keypoints[R_SHOULDER], keypoints[NOSE]
    if min(ls[2], rs[2], nose[2]) < KEYPOINT_CONF_MIN:
        return None

    shoulder_width = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
    if shoulder_width < 1e-3:
        return None

    shoulder_y = (ls[1] + rs[1]) / 2.0
    # Image y grows downward, so a nose below the shoulders is a positive drop.
    return float((nose[1] - shoulder_y) / shoulder_width)


def wrists_below_line(keypoints: np.ndarray, desk_line_y: float | None) -> tuple[int, bool]:
    """Count visible wrists below the desk line.

    Returns (count, mask_required). Occluded wrists are not counted as below --
    absence of evidence is not evidence of a hand in the lap.
    """
    if desk_line_y is None:
        return 0, False
    count = 0
    for index in (L_WRIST, R_WRIST):
        if keypoints[index, 2] >= KEYPOINT_CONF_MIN and keypoints[index, 1] > desk_line_y:
            count += 1
    return count, count > 0


def mask_wrists(keypoints: np.ndarray, desk_line_y: float | None) -> np.ndarray:
    """Zero the confidence of wrists below the desk line.

    Below the desk the wrist is occluded by the desk itself; whatever the
    estimator returns there is extrapolation, and downstream scoring must not
    treat it as a measurement.
    """
    out = keypoints.copy()
    if desk_line_y is None:
        return out
    for index in (L_WRIST, R_WRIST):
        if out[index, 1] > desk_line_y:
            out[index, 2] = 0.0
    return out


def observe(keypoints: np.ndarray, desk_line_y: float | None) -> PoseObservation:
    """Build a pose observation with the masking rules applied."""
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    below, needs_mask = wrists_below_line(keypoints, desk_line_y)
    return PoseObservation(
        keypoints=mask_wrists(keypoints, desk_line_y),
        head_pitch=head_pitch(keypoints),
        wrist_masked=needs_mask,
        wrists_below=below,
    )


# Default only. Measured on real footage from this corpus, an ordinary upright
# candidate at a high-mounted CCTV angle scores between -0.35 and -2.3: the
# camera looks down, so the nose sits *above* the shoulder line in image space
# for almost everyone, and shoulder width foreshortens. An absolute threshold
# is therefore camera-specific, exactly like the motion baseline in PRD 7.2.1.
# Calibrate per camera before trusting this signal; the default is a starting
# point, not a validated constant.
DEFAULT_PITCH_BASELINE = 0.35


def pitch_evidence(
    observations: list[PoseObservation], baseline: float = DEFAULT_PITCH_BASELINE
) -> float:
    """Fraction of the event spent with the head inclined past `baseline`.

    Frames where pitch is unresolvable are excluded from the denominator rather
    than counted as upright -- on this footage roughly a third of detected
    people have no resolvable pitch at any moment, and scoring those as "head
    up" would understate exactly the posture the system is looking for.
    """
    resolved = [o.head_pitch for o in observations if o.head_pitch is not None]
    if not resolved:
        return 0.0
    return float(np.mean([p > baseline for p in resolved]))


def calibrate_pitch_baseline(
    observations: list[PoseObservation], percentile: float = 90.0
) -> float | None:
    """Derive a camera-specific pitch threshold from normal footage.

    Run over a span believed ordinary: the returned value is the inclination
    only the top `100 - percentile`% of normal postures exceed, so it adapts to
    camera height and angle instead of assuming a mounting geometry.
    """
    resolved = [o.head_pitch for o in observations if o.head_pitch is not None]
    if len(resolved) < 20:
        return None
    return float(np.percentile(resolved, percentile))


def below_desk_evidence(observations: list[PoseObservation]) -> float:
    """Fraction of frames with at least one wrist confidently below the line."""
    if not observations:
        return 0.0
    return float(np.mean([o.wrists_below > 0 for o in observations]))


_RTMO_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"

# Both published one-stage sizes. The medium model was the one benchmarked
# first; the large one is here so the comparison below can be re-run against the
# strongest one-stage option rather than only the convenient one.
RTMO_MODELS = {
    "rtmo-m": _RTMO_BASE + "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
    "rtmo-l": _RTMO_BASE + "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",
}

# Retained under its original name for anything that imported it directly.
RTMO_MODEL = RTMO_MODELS["rtmo-m"]

# Measured on real corpus frames, CPU, ONNXRuntime, via tools/pose_ab.py
# (see PRD 8.2.1). Eight frames spread across all six cameras, every model
# seeing exactly the same ones:
#
#   model      people   pitch resolved   s/frame   people/frame
#   topdown        43              36      3.74            5.4
#   rtmo-m         23              15      1.00            2.9
#   rtmo-l         25              16      2.00            3.1
#
# The large one-stage model was the last open question in the stack, on the
# theory that RTMO-m was simply too small. It is not: going from m to l buys two
# more people and one more resolvable pitch for double the cost, leaving
# top-down still 1.72x ahead on person recall and 2.25x ahead on usable head
# pitch. The gap is architectural rather than a matter of capacity.
#
# The skeleton overlays settle what the counts cannot -- that the extra
# detections are distinct seated candidates deeper in each row, not duplicate
# skeletons stacked on one person. On the densest frame top-down finds 13 where
# rtmo-l finds 6, and the seven it adds are individually visible people.
#
# RTMO's real advantage is that its cost does not scale with crowd size. This
# pipeline runs pose on candidate clips only, so throughput was never the
# binding constraint, and paying for it with more than half the person recall is
# the wrong trade here.
#
# Top-down is therefore the default. Both one-stage sizes stay selectable so the
# comparison can be reproduced, not as a runtime fallback.
TOPDOWN = "topdown"
ONESTAGE = "rtmo-m"
MODELS = (TOPDOWN, "rtmo-m", "rtmo-l")


class PoseEstimator:
    """Multi-person 2D pose over a full frame.

    Loaded lazily so the geometry and masking rules above stay testable without
    downloading weights, and so a missing model surfaces as one clear error at
    load rather than a cascade at inference.
    """

    def __init__(
        self,
        model: str = TOPDOWN,
        backend: str = "onnxruntime",
        device: str = "cpu",
        confidence: float = 0.3,
    ):
        if model == "rtmo":
            model = ONESTAGE   # the medium model, named before there were two
        if model not in MODELS:
            raise ValueError(
                f"unknown pose model {model!r}; expected one of {MODELS}")
        self.model = model
        self.backend, self.device = backend, device
        self.confidence = confidence
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        if self.model == TOPDOWN:
            from rtmlib import Body

            self._model = Body(
                mode="balanced", backend=self.backend, device=self.device
            )
        else:
            from rtmlib import RTMO

            self._model = RTMO(
                RTMO_MODELS[self.model], model_input_size=(640, 640),
                backend=self.backend, device=self.device,
            )

    def __call__(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return one (17, 3) array per detected person."""
        self.load()
        keypoints, scores = self._model(frame)
        out = []
        for person_kp, person_sc in zip(keypoints, scores):
            arr = np.zeros((17, 3), dtype=np.float32)
            arr[:, :2] = np.asarray(person_kp)[:17, :2]
            arr[:, 2] = np.asarray(person_sc).reshape(-1)[:17]
            if float(arr[:, 2].max()) >= self.confidence:
                out.append(arr)
        return out


# Retained so existing imports keep working; the default model is now top-down.
RTMOPose = PoseEstimator
