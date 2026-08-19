"""Named preprocessing comparisons (PRD 6, 6.3.7).

PRD 6.3.7 permits enhancement under one restriction, and the restriction is the
design:

> CLAHE, denoising, sharpening, or enhancement are named comparisons only;
> never unconditional full-video rewrites.

So nothing here is applied by default and nothing rewrites a source frame. Each
transform is a *named configuration* that is run enabled and disabled on the
same frames, and the difference is measured. PRD 6 also requires that every
enabled step stores raw input, transformed output, a difference image, its
configuration ID, the PTS, and metrics -- which is what `Comparison` carries.

### Why this is a comparison and not a fix

Enhancement is seductive on poor CCTV: CLAHE makes a dark frame *look* far more
readable, and it is easy to conclude the pipeline should always run it. But
contrast enhancement amplifies compression blocking along with the signal, and
on 30x20 px objects the blocking is the same size as the object. Whether it
helps is an empirical question about this footage, and PRD 6 requires it be
answered by measurement on known references rather than by eye.

Until a reference manifest exists, `evaluate` reports the *effect* of each
transform -- how much it changed the image, and how it moved detector-relevant
statistics -- and states plainly that effect is not benefit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

IDENTITY = "identity_no_preprocessing"
CLAHE = "clahe"
DENOISE = "denoise_nlmeans"
SHARPEN = "unsharp_mask"
PREPROCESS_CONFIGS = (IDENTITY, CLAHE, DENOISE, SHARPEN)


@dataclass
class PreprocessConfig:
    # CLAHE. A 2.0 clip limit on an 8x8 grid is the conventional starting point;
    # higher clip limits visibly amplify compression blocking on this corpus.
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    # Non-local means. h is the filter strength; above ~10 fine texture that a
    # 30x20 px object consists of is removed along with the noise.
    denoise_h: float = 6.0
    denoise_template: int = 7
    denoise_search: int = 21

    # Unsharp mask: blur radius and the weight of the high-frequency residual.
    sharpen_sigma: float = 1.5
    sharpen_amount: float = 0.8


def apply(frame: np.ndarray, config_id: str,
          cfg: PreprocessConfig | None = None) -> np.ndarray:
    """Apply one named transform. `identity` returns the frame untouched.

    Identity is a named configuration rather than a special case, so the
    comparison always has a baseline arm and nobody has to remember to add one.
    """
    cfg = cfg or PreprocessConfig()
    if config_id == IDENTITY:
        return frame
    import cv2

    if config_id == CLAHE:
        # Applied to luminance only. Running CLAHE per BGR channel shifts hue,
        # and a colour shift on a phone-versus-desk decision is a change in the
        # evidence, not an enhancement of it.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                                tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if config_id == DENOISE:
        return cv2.fastNlMeansDenoisingColored(
            frame, None, cfg.denoise_h, cfg.denoise_h,
            cfg.denoise_template, cfg.denoise_search)

    if config_id == SHARPEN:
        blurred = cv2.GaussianBlur(frame, (0, 0), cfg.sharpen_sigma)
        return cv2.addWeighted(frame, 1.0 + cfg.sharpen_amount,
                               blurred, -cfg.sharpen_amount, 0)

    raise ValueError(f"unknown preprocessing configuration {config_id!r}; "
                     f"expected one of {PREPROCESS_CONFIGS}")


def difference(raw: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    """The diagnostic difference image PRD 6 requires for every enabled step."""
    return np.abs(transformed.astype(np.int16)
                  - raw.astype(np.int16)).astype(np.uint8)


def statistics(frame: np.ndarray) -> dict:
    """Detector-relevant image statistics, measured the same way for every arm.

    `sharpness` is the variance of the Laplacian -- the standard focus measure,
    and the one the health monitor already uses for blur. `blocking` estimates
    JPEG/MPEG block artefacts by comparing gradient energy across the 8-pixel
    block grid against gradient energy inside blocks: a transform that raises
    this is amplifying compression structure, which at this object scale is
    indistinguishable from amplifying an object.
    """
    import cv2

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)

    columns = np.abs(np.diff(grey.astype(np.int16), axis=1))
    on_grid = columns[:, 7::8]
    off_grid = np.delete(columns, np.s_[7::8], axis=1)
    blocking = (float(on_grid.mean()) / float(off_grid.mean())
                if off_grid.size and off_grid.mean() > 0 else 0.0)

    return {
        "mean_luma": round(float(grey.mean()), 3),
        "std_luma": round(float(grey.std()), 3),
        "sharpness": round(float(laplacian.var()), 3),
        "blocking_ratio": round(blocking, 4),
        "saturated_high": round(float((grey >= 250).mean()), 5),
        "saturated_low": round(float((grey <= 5).mean()), 5),
    }


@dataclass
class Comparison:
    """One transform measured against identity on the same frames (PRD 6)."""

    config_id: str
    frames: int = 0
    seconds: float = 0.0
    mean_abs_difference: float = 0.0
    max_abs_difference: float = 0.0
    baseline: dict = field(default_factory=dict)
    transformed: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return asdict(self)


def compare(frames, config_id: str, cfg: PreprocessConfig | None = None,
            on_frame=None) -> Comparison:
    """Run one transform over frames and measure it against the untouched input.

    `on_frame(pts_ms, raw, transformed, difference)` is called per frame so the
    caller can persist the raw input, the transformed output and the difference
    image, which PRD 6 requires for every enabled step.
    """
    import time

    cfg = cfg or PreprocessConfig()
    comparison = Comparison(config_id=config_id)
    base_stats, out_stats, diffs = [], [], []
    started = time.perf_counter()

    for pts_ms, frame in frames:
        transformed = apply(frame, config_id, cfg)
        delta = difference(frame, transformed)
        base_stats.append(statistics(frame))
        out_stats.append(statistics(transformed))
        diffs.append((float(delta.mean()), float(delta.max())))
        comparison.frames += 1
        if on_frame is not None:
            on_frame(pts_ms, frame, transformed, delta)

    comparison.seconds = time.perf_counter() - started
    if not comparison.frames:
        return comparison

    def mean_of(rows, key):
        return round(float(np.mean([r[key] for r in rows])), 4)

    keys = base_stats[0].keys()
    comparison.baseline = {k: mean_of(base_stats, k) for k in keys}
    comparison.transformed = {k: mean_of(out_stats, k) for k in keys}
    comparison.delta = {k: round(comparison.transformed[k] - comparison.baseline[k], 4)
                        for k in keys}
    comparison.mean_abs_difference = round(float(np.mean([d[0] for d in diffs])), 4)
    comparison.max_abs_difference = round(float(np.max([d[1] for d in diffs])), 4)
    return comparison


def summarise(comparisons: list[Comparison]) -> dict:
    """Report the effect of each transform, and refuse to call effect benefit."""
    rows = {c.config_id: c.to_row() for c in comparisons}
    identity = next((c for c in comparisons if c.config_id == IDENTITY), None)

    verdict = (
        "No transform is recommended. These figures measure how much each "
        "transform CHANGED the image, not whether it helped. PRD 6 requires "
        "any transform to be evaluated enabled and disabled on known "
        "references, and no reference manifest exists for this corpus. A "
        "transform that raises sharpness while also raising blocking_ratio is "
        "amplifying compression structure, which at a 30x20 px object scale is "
        "indistinguishable from amplifying the object.")

    if identity is not None and identity.mean_abs_difference != 0.0:
        verdict = (f"IDENTITY IS NOT IDENTITY: it changed the image by "
                   f"{identity.mean_abs_difference}. The baseline arm is "
                   f"broken and no comparison here is valid.")

    return {
        "configurations": rows,
        "note": "every transform ran on the same frames as the identity "
                "baseline; none was applied to the pipeline's own inputs.",
        "recommendation": verdict,
    }
