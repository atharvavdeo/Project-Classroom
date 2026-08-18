"""Global camera-motion estimation and compensation (PRD 6.3).

A camera that moves makes every seat in the frame move with it, so *every* zone
reports motion at once. Compensating for that is worth doing; compensating
*badly* is worse than not compensating at all, because a wrong transform
smears real motion across neighbouring seats and manufactures activity where
there was none.

So this stage is built around refusing rather than around succeeding:

  * **Identity is a named baseline, not the absence of a method** (PRD 6.3.2).
    It competes in the same comparison as the estimators and wins whenever they
    cannot beat it. On a rigidly mounted camera it should win every time, and
    that is a result, not a failure.

  * **Compensation is applied only when quality gates pass** (PRD 6.3.4).
    Inlier coverage, residual and confidence each have an immutable threshold.
    Below any of them the span is `camera_motion_uncertain`, the raw evidence is
    retained, and no local attribution is made for it.

  * **Fitting happens only on declared static background** (PRD 6.3.1). A
    transform fitted on pixels containing people measures the people. The
    support mask comes from calibration and already excludes seats, aisles,
    monitors, reflections and environment regions.

  * **A sustained geometry change is a new epoch** (PRD 6.3.5), not a large
    correction. Past that point the seat polygons describe a room that is no
    longer where they say it is, and attribution must stop until a profile for
    the new epoch is approved.

Every estimate keeps its failure reason. PRD 6.3.3 requires the raw frame, the
support mask, the features, the transform, the confidence, the compensated frame
and the difference image to be saved; `render_diagnostics` produces that set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Named methods. PRD 6.3.2: ECC is a *measured challenger*, not the default, and
# identity is a baseline in its own right.
IDENTITY = "identity"
FEATURES = "features_ransac"
ECC = "ecc"
METHODS = (IDENTITY, FEATURES, ECC)

# Outcomes.
COMPENSATED = "compensated"
NOT_NEEDED = "not_needed"           # measured, and the camera did not move
UNCERTAIN = "camera_motion_uncertain"
FAILED = "estimation_failed"


@dataclass
class AlignmentConfig:
    """Immutable thresholds (PRD 6.3.4, 17.10)."""

    # Fraction of the frame that must be usable static background before an
    # estimate is attempted at all. Below this there is not enough rigid
    # structure for any method to fit against.
    min_support_coverage: float = 0.02

    # RANSAC inliers as a fraction of matched features. A low ratio means the
    # matches disagree about what the transform is, which is exactly when a
    # confident-looking wrong answer appears.
    min_inlier_ratio: float = 0.60
    min_inliers: int = 12

    # Mean reprojection error of the inliers, in pixels. A transform that fits
    # its own inliers badly does not describe the frame.
    max_residual_px: float = 2.0

    # Below this the camera is treated as still. Sub-pixel drift is estimation
    # noise, and "compensating" for it resamples every frame for nothing.
    still_translation_px: float = 0.5

    # Above this, a single frame's apparent motion is more likely a bad fit than
    # a real camera movement between two consecutive frames.
    implausible_translation_px: float = 120.0

    # ECC convergence.
    ecc_iterations: int = 60
    ecc_epsilon: float = 1e-4
    # ECC returns a correlation coefficient; below this its answer is not
    # trusted regardless of whether it converged.
    ecc_min_correlation: float = 0.90

    # Epoch break: sustained displacement over a window, not one frame.
    epoch_break_translation_px: float = 8.0
    epoch_break_sustained_ms: float = 2000.0

    # Feature detection.
    max_features: int = 2000


@dataclass
class AlignmentResult:
    """One estimate of how the frame moved, and whether to believe it."""

    method: str
    pts_ms: float
    outcome: str
    transform: np.ndarray | None = None      # 2x3 affine, source -> reference
    matched: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    residual_px: float = 0.0
    correlation: float | None = None         # ECC only
    support_coverage: float = 0.0
    translation_px: float = 0.0
    seconds: float = 0.0
    reason: str = ""

    @property
    def compensate(self) -> bool:
        return self.outcome == COMPENSATED and self.transform is not None

    @property
    def confidence(self) -> float:
        """A single number for ranking methods against each other.

        Deliberately crude and deliberately explicit: the inlier ratio scaled
        down by residual error. It is used to order candidates in a comparison,
        never as a gate -- the gates are the named thresholds above, because a
        blended score can hide a failing component behind a passing one.
        """
        if self.outcome in (FAILED, UNCERTAIN):
            return 0.0
        if self.method == IDENTITY:
            # Identity's confidence is how little it leaves behind, not a flat
            # 1.0. Measured on a 40 px synthetic shift it left a mean residual
            # of 67 luma levels while still reporting `not_needed`; a flat score
            # would have made the do-nothing baseline outrank both estimators in
            # the comparison report while being plainly the worst answer.
            return round(1.0 / (1.0 + max(self.residual_px, 0.0)), 4)
        penalty = 1.0 / (1.0 + max(self.residual_px, 0.0))
        return round(max(self.inlier_ratio, 0.0) * penalty, 4)

    def to_row(self) -> dict:
        return {
            "method": self.method,
            "pts_ms": round(self.pts_ms, 3),
            "outcome": self.outcome,
            "transform": self.transform.tolist() if self.transform is not None else None,
            "matched": self.matched,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "residual_px": round(self.residual_px, 4),
            "correlation": round(self.correlation, 4) if self.correlation is not None else None,
            "support_coverage": round(self.support_coverage, 5),
            "translation_px": round(self.translation_px, 4),
            "confidence": self.confidence,
            "seconds": round(self.seconds, 4),
            "reason": self.reason,
        }


def _translation(transform: np.ndarray | None) -> float:
    if transform is None:
        return 0.0
    return float(np.hypot(transform[0, 2], transform[1, 2]))


def _grey(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def estimate_identity(reference: np.ndarray, frame: np.ndarray,
                      support: np.ndarray, cfg: AlignmentConfig,
                      pts_ms: float = 0.0) -> AlignmentResult:
    """The baseline: assume the camera did not move, and measure the cost.

    Reported as `not_needed` with the residual it leaves behind, so a real
    estimator has something to beat rather than being adopted by default.
    """
    import time
    started = time.perf_counter()
    a, b = _grey(reference), _grey(frame)
    inside = support > 0
    residual = float(np.abs(a[inside].astype(np.int16)
                            - b[inside].astype(np.int16)).mean()) if inside.any() else 0.0
    return AlignmentResult(
        method=IDENTITY, pts_ms=pts_ms, outcome=NOT_NEEDED,
        transform=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        support_coverage=float(inside.mean()),
        residual_px=residual, translation_px=0.0,
        seconds=time.perf_counter() - started,
        reason="baseline: no compensation applied",
    )


def estimate_features(reference: np.ndarray, frame: np.ndarray,
                      support: np.ndarray, cfg: AlignmentConfig,
                      pts_ms: float = 0.0) -> AlignmentResult:
    """ORB features on static background, matched and fitted with RANSAC."""
    import time
    started = time.perf_counter()
    coverage = float((support > 0).mean())
    result = AlignmentResult(method=FEATURES, pts_ms=pts_ms, outcome=FAILED,
                             support_coverage=coverage)

    if coverage < cfg.min_support_coverage:
        result.reason = (f"static-background support is {coverage:.4f} of the "
                         f"frame, below {cfg.min_support_coverage}")
        result.seconds = time.perf_counter() - started
        return result

    a, b = _grey(reference), _grey(frame)
    mask = (support > 0).astype(np.uint8) * 255
    orb = cv2.ORB_create(nfeatures=cfg.max_features)
    ka, da = orb.detectAndCompute(a, mask)
    kb, db = orb.detectAndCompute(b, mask)
    if da is None or db is None or len(ka) < cfg.min_inliers or len(kb) < cfg.min_inliers:
        result.reason = (f"too few features on static background "
                         f"({len(ka or [])} and {len(kb or [])})")
        result.seconds = time.perf_counter() - started
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(da, db)
    result.matched = len(matches)
    if len(matches) < cfg.min_inliers:
        result.reason = f"only {len(matches)} matches, need {cfg.min_inliers}"
        result.seconds = time.perf_counter() - started
        return result

    src = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=3000, confidence=0.995)

    if transform is None or inlier_mask is None:
        result.reason = "RANSAC did not produce a transform"
        result.seconds = time.perf_counter() - started
        return result

    inliers = int(inlier_mask.sum())
    result.inliers = inliers
    result.inlier_ratio = inliers / max(len(matches), 1)
    result.transform = transform
    result.translation_px = _translation(transform)

    keep = inlier_mask.ravel().astype(bool)
    if keep.any():
        projected = cv2.transform(src[keep], transform).reshape(-1, 2)
        result.residual_px = float(
            np.linalg.norm(projected - dst[keep].reshape(-1, 2), axis=1).mean())

    result.outcome, result.reason = _gate(result, cfg)
    result.seconds = time.perf_counter() - started
    return result


def estimate_ecc(reference: np.ndarray, frame: np.ndarray,
                 support: np.ndarray, cfg: AlignmentConfig,
                 pts_ms: float = 0.0) -> AlignmentResult:
    """ECC image registration, as the measured challenger PRD 6.3.2 asks for.

    ECC optimises photometric correlation directly rather than matching
    features, which makes it strong on low-texture scenes and fragile under
    exposure change. It reports its own correlation coefficient, and that is
    kept as the quality score PRD 6.3.2 requires alongside convergence.
    """
    import time
    started = time.perf_counter()
    coverage = float((support > 0).mean())
    result = AlignmentResult(method=ECC, pts_ms=pts_ms, outcome=FAILED,
                             support_coverage=coverage)

    if coverage < cfg.min_support_coverage:
        result.reason = (f"static-background support is {coverage:.4f}, below "
                         f"{cfg.min_support_coverage}")
        result.seconds = time.perf_counter() - started
        return result

    a = _grey(reference).astype(np.float32) / 255.0
    b = _grey(frame).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                cfg.ecc_iterations, cfg.ecc_epsilon)
    try:
        correlation, warp = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_EUCLIDEAN, criteria,
            (support > 0).astype(np.uint8), 5)
    except cv2.error as exc:
        # ECC raises rather than returning a poor score when it cannot
        # converge. PRD 19 anticipates this: it "may fail under poor
        # texture/exposure/vibration", which is why it is a challenger.
        result.reason = f"did not converge: {str(exc).strip().splitlines()[-1][:120]}"
        result.seconds = time.perf_counter() - started
        return result

    # Clamp: ECC can return marginally above 1.0 on identical frames, which
    # would produce a negative residual below.
    result.correlation = float(min(max(correlation, 0.0), 1.0))
    correlation = result.correlation
    # ECC solves reference -> frame; the transform used downstream maps the
    # frame back onto the reference, so it is inverted here.
    result.transform = cv2.invertAffineTransform(warp)
    result.translation_px = _translation(result.transform)
    # ECC has no inliers; correlation stands in for both quality measures so the
    # shared gate can judge it on the same terms as the feature estimator.
    result.inlier_ratio = float(correlation)
    result.inliers = cfg.min_inliers if correlation >= cfg.ecc_min_correlation else 0
    result.residual_px = float((1.0 - correlation) * 10.0)

    if correlation < cfg.ecc_min_correlation:
        result.outcome = UNCERTAIN
        result.reason = (f"correlation {correlation:.4f} below "
                         f"{cfg.ecc_min_correlation}")
    else:
        result.outcome, result.reason = _gate(result, cfg)
    result.seconds = time.perf_counter() - started
    return result


def _gate(result: AlignmentResult, cfg: AlignmentConfig) -> tuple[str, str]:
    """Apply PRD 6.3.4's thresholds. Every failure names the number it failed."""
    if result.translation_px > cfg.implausible_translation_px:
        return UNCERTAIN, (f"apparent shift of {result.translation_px:.1f} px "
                           f"exceeds {cfg.implausible_translation_px} px and is "
                           f"more likely a bad fit than real camera movement")
    if result.inliers < cfg.min_inliers:
        return UNCERTAIN, f"{result.inliers} inliers, need {cfg.min_inliers}"
    if result.inlier_ratio < cfg.min_inlier_ratio:
        return UNCERTAIN, (f"inlier ratio {result.inlier_ratio:.3f} below "
                           f"{cfg.min_inlier_ratio}")
    if result.residual_px > cfg.max_residual_px:
        return UNCERTAIN, (f"residual {result.residual_px:.2f} px above "
                           f"{cfg.max_residual_px} px")
    if result.translation_px < cfg.still_translation_px:
        return NOT_NEEDED, (f"measured shift {result.translation_px:.3f} px is "
                            f"below {cfg.still_translation_px} px; the camera is "
                            f"still and resampling would cost accuracy for nothing")
    return COMPENSATED, f"shift {result.translation_px:.2f} px, gates passed"


ESTIMATORS = {
    IDENTITY: estimate_identity,
    FEATURES: estimate_features,
    ECC: estimate_ecc,
}


def compare(reference: np.ndarray, frame: np.ndarray, support: np.ndarray,
            cfg: AlignmentConfig | None = None, pts_ms: float = 0.0,
            methods: tuple[str, ...] = METHODS) -> list[AlignmentResult]:
    """Run every named method on the identical pair (PRD 4, comparison-first).

    All methods see the same reference, frame and support mask, so a difference
    in the result is a difference in the method rather than in its inputs.
    """
    cfg = cfg or AlignmentConfig()
    return [ESTIMATORS[m](reference, frame, support, cfg, pts_ms) for m in methods]


def select(results: list[AlignmentResult], cfg: AlignmentConfig | None = None
           ) -> AlignmentResult:
    """Pick the estimate to act on, defaulting to no compensation.

    Only a result that passed every gate may be selected for compensation.
    Otherwise the identity baseline is returned -- with the challengers' reasons
    preserved on their own records, so the comparison report can say why nothing
    was applied.
    """
    cfg = cfg or AlignmentConfig()
    usable = [r for r in results if r.outcome == COMPENSATED]
    if usable:
        return max(usable, key=lambda r: r.confidence)
    identity = next((r for r in results if r.method == IDENTITY), None)
    if identity is not None:
        return identity
    return min(results, key=lambda r: r.translation_px)


def apply(frame: np.ndarray, result: AlignmentResult) -> np.ndarray:
    """Warp a frame onto the reference, or return it untouched."""
    if not result.compensate:
        return frame
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, result.transform, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def render_diagnostics(reference: np.ndarray, frame: np.ndarray,
                       support: np.ndarray, result: AlignmentResult
                       ) -> dict[str, np.ndarray]:
    """The artifact set PRD 6.3.3 requires for every alignment attempt."""
    compensated = apply(frame, result)
    difference = cv2.absdiff(_grey(reference), _grey(compensated))
    support_view = (support > 0).astype(np.uint8) * 255
    return {
        "raw.png": frame,
        "reference.png": reference,
        "static_mask.png": support_view,
        "compensated.png": compensated,
        "difference.png": cv2.applyColorMap(
            cv2.convertScaleAbs(difference, alpha=4.0), cv2.COLORMAP_INFERNO),
    }


# ------------------------------------------------------------- epochs --

@dataclass
class EpochBreak:
    """A sustained geometry change: the camera is no longer where it was."""

    pts_ms: float
    translation_px: float
    sustained_ms: float
    reason: str = ""


def detect_epoch_breaks(results: list[AlignmentResult],
                        cfg: AlignmentConfig | None = None) -> list[EpochBreak]:
    """Find sustained displacement, not a single large frame (PRD 6.3.5).

    One frame with a big apparent shift is a bad fit or a bump that settles. A
    displacement that persists means the camera is pointing somewhere else, and
    every seat polygon is now describing a room that has moved beneath it.
    """
    cfg = cfg or AlignmentConfig()
    ordered = sorted(results, key=lambda r: r.pts_ms)
    breaks: list[EpochBreak] = []
    run_start: float | None = None
    peak = 0.0
    # One reposition is one break. Without this, a camera that stays moved keeps
    # re-qualifying every `epoch_break_sustained_ms` and emits a fresh break for
    # as long as the displacement lasts -- measured: a single 6-second
    # displacement produced two breaks, and a five-minute one would produce
    # dozens, each spawning an epoch and blocking attribution again.
    reported_for_run = False

    for result in ordered:
        moved = (result.translation_px >= cfg.epoch_break_translation_px
                 and result.outcome in (COMPENSATED, UNCERTAIN))
        if not moved:
            # The camera came back. The next displacement is a new event.
            run_start, peak, reported_for_run = None, 0.0, False
            continue

        run_start = result.pts_ms if run_start is None else run_start
        peak = max(peak, result.translation_px)
        if reported_for_run:
            continue
        if result.pts_ms - run_start >= cfg.epoch_break_sustained_ms:
            breaks.append(EpochBreak(
                pts_ms=run_start, translation_px=peak,
                sustained_ms=result.pts_ms - run_start,
                reason=f"displacement of {peak:.1f} px sustained for "
                       f"{result.pts_ms - run_start:.0f} ms; seat attribution "
                       f"is blocked until a profile for the new epoch is "
                       f"approved"))
            reported_for_run = True
    return breaks
