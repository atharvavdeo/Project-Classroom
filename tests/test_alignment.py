"""Phase 2 gate: global alignment and epoch detection (PRD 6.3).

Driven by **known synthetic shifts applied to real corpus frames**, so the
correct answer is known to the pixel and a method that reports a plausible wrong
transform fails here rather than downstream.

The checks that matter are the refusals. Compensating badly is worse than not
compensating: a wrong transform smears real motion across neighbouring seats and
manufactures activity where there was none. So every quality gate is exercised
with an input designed to trip it, and `select()` must fall back to the identity
baseline rather than accept a poor estimate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.alignment import (  # noqa: E402
    COMPENSATED, ECC, FAILED, FEATURES, IDENTITY, METHODS, NOT_NEEDED, UNCERTAIN,
    AlignmentConfig, AlignmentResult, apply, compare, detect_epoch_breaks,
    render_diagnostics, select,
)

FRAMES_DIR = Path("data/frames")


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def shift(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                          borderMode=cv2.BORDER_REPLICATE)


def main() -> int:
    ok = True
    cfg = AlignmentConfig()

    candidates = sorted(FRAMES_DIR.glob("cam12_*.jpg")) or sorted(FRAMES_DIR.glob("*.jpg"))
    if not candidates:
        print("SKIP: no extracted frames under data/frames")
        return 0
    reference = cv2.imread(str(candidates[0]))
    h, w = reference.shape[:2]
    print(f"reference {w}x{h} from {candidates[0].name}")

    # Static background: the upper band, away from desks and people.
    support = np.zeros((h, w), np.uint8)
    support[0:int(h * 0.36), int(w * 0.31):] = 1
    print(f"support coverage {float((support > 0).mean()):.4f}\n")

    print("a still camera")
    results = compare(reference, reference.copy(), support, cfg)
    ok &= check("every named method runs", len(results) == len(METHODS))
    ok &= check("all report the camera did not move",
                all(r.outcome == NOT_NEEDED for r in results),
                str([(r.method, r.outcome) for r in results]))
    ok &= check("identity leaves no residual on an identical frame",
                next(r for r in results if r.method == IDENTITY).residual_px < 0.01)
    ok &= check("nothing is compensated", not any(r.compensate for r in results),
                "resampling a still frame costs accuracy for nothing")
    ok &= check("the selection is the identity baseline",
                select(results, cfg).method == IDENTITY)

    print("\na known 40 px shift on real pixels")
    dx, dy = 40.0, -15.0
    expected = float(np.hypot(dx, dy))
    moved = shift(reference, dx, dy)
    results = compare(reference, moved, support, cfg)
    by_method = {r.method: r for r in results}

    identity = by_method[IDENTITY]
    ok &= check("identity still reports no movement",
                identity.translation_px == 0.0)
    ok &= check("but records the residual it leaves behind",
                identity.residual_px > 10.0, f"{identity.residual_px:.1f}")
    # A flat confidence for identity would let the do-nothing baseline outrank
    # both estimators in the comparison report while being the worst answer.
    ok &= check("identity's confidence reflects that residual",
                identity.confidence < 0.1, f"{identity.confidence}")

    for method in (FEATURES, ECC):
        result = by_method[method]
        ok &= check(f"{method} recovers the shift within a pixel",
                    abs(result.translation_px - expected) < 1.5,
                    f"{result.translation_px:.2f} vs {expected:.2f}")
        ok &= check(f"{method} passes its gates",
                    result.outcome == COMPENSATED, result.reason)
        ok &= check(f"{method} beats identity on confidence",
                    result.confidence > identity.confidence,
                    f"{result.confidence} vs {identity.confidence}")

    chosen = select(results, cfg)
    ok &= check("an estimator is selected over the baseline",
                chosen.method in (FEATURES, ECC), chosen.method)

    corrected = apply(moved, chosen)
    inside = support > 0
    before = float(np.abs(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[inside].astype(np.int16)
                          - cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)[inside].astype(np.int16)).mean())
    after = float(np.abs(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[inside].astype(np.int16)
                         - cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)[inside].astype(np.int16)).mean())
    ok &= check("applying the transform reduces the difference",
                after < before * 0.25, f"{before:.1f} -> {after:.1f}")

    print("\nsub-pixel drift is not compensated")
    results = compare(reference, shift(reference, 0.2, 0.1), support, cfg)
    ok &= check("no method compensates for drift under the still threshold",
                not any(r.compensate for r in results),
                str([(r.method, r.outcome, round(r.translation_px, 3)) for r in results]))

    print("\nquality gates")
    thin = np.zeros((h, w), np.uint8)
    thin[0:4, 0:4] = 1
    results = compare(reference, shift(reference, 10, 0), thin, cfg)
    for method in (FEATURES, ECC):
        result = next(r for r in results if r.method == method)
        ok &= check(f"{method} refuses on insufficient background support",
                    result.outcome == FAILED and "support" in result.reason,
                    result.reason[:60])

    # A hand-built result exercising each threshold directly, so the gate is
    # tested rather than the estimator that happens to feed it.
    from pipeline.alignment import _gate

    outcome, reason = _gate(AlignmentResult(
        FEATURES, 0.0, COMPENSATED, inliers=200, inlier_ratio=0.95,
        residual_px=0.4, translation_px=500.0), cfg)
    ok &= check("an implausible shift is uncertain, not compensated",
                outcome == UNCERTAIN and "implausible" not in reason
                and "more likely a bad fit" in reason, reason[:70])

    outcome, _ = _gate(AlignmentResult(
        FEATURES, 0.0, COMPENSATED, inliers=4, inlier_ratio=0.95,
        residual_px=0.4, translation_px=5.0), cfg)
    ok &= check("too few inliers is uncertain", outcome == UNCERTAIN)

    outcome, _ = _gate(AlignmentResult(
        FEATURES, 0.0, COMPENSATED, inliers=200, inlier_ratio=0.30,
        residual_px=0.4, translation_px=5.0), cfg)
    ok &= check("a low inlier ratio is uncertain", outcome == UNCERTAIN)

    outcome, _ = _gate(AlignmentResult(
        FEATURES, 0.0, COMPENSATED, inliers=200, inlier_ratio=0.95,
        residual_px=9.0, translation_px=5.0), cfg)
    ok &= check("a high residual is uncertain", outcome == UNCERTAIN)

    print("\nselection defaults to doing nothing")
    poor = [
        AlignmentResult(IDENTITY, 0.0, NOT_NEEDED, transform=np.eye(2, 3),
                        residual_px=30.0),
        AlignmentResult(FEATURES, 0.0, UNCERTAIN, reason="inlier ratio 0.3"),
        AlignmentResult(ECC, 0.0, FAILED, reason="did not converge"),
    ]
    picked = select(poor, cfg)
    ok &= check("nothing usable means the identity baseline",
                picked.method == IDENTITY, picked.method)
    ok &= check("and no compensation is applied", not picked.compensate)
    ok &= check("the challengers keep their reasons",
                all(r.reason for r in poor[1:]),
                "PRD 6.3.3 requires the failure reason retained")

    print("\ndiagnostic artifacts (PRD 6.3.3)")
    results = compare(reference, shift(reference, 20, 8), support, cfg)
    artifacts = render_diagnostics(reference, shift(reference, 20, 8), support,
                                   select(results, cfg))
    for name in ("raw.png", "reference.png", "static_mask.png",
                 "compensated.png", "difference.png"):
        ok &= check(f"{name} rendered", name in artifacts
                    and artifacts[name] is not None)

    print("\nepoch breaks need sustained displacement")
    spike = [AlignmentResult(FEATURES, t, COMPENSATED, translation_px=0.1)
             for t in range(0, 6000, 200)]
    spike[10] = AlignmentResult(FEATURES, 2000.0, COMPENSATED, translation_px=30.0)
    ok &= check("one displaced frame is not an epoch break",
                not detect_epoch_breaks(spike, cfg),
                "a bump that settles is not a reposition")

    sustained = []
    for t in range(0, 3000, 200):
        sustained.append(AlignmentResult(FEATURES, float(t), COMPENSATED,
                                         translation_px=0.1))
    for t in range(3000, 9000, 200):
        sustained.append(AlignmentResult(FEATURES, float(t), COMPENSATED,
                                         translation_px=25.0))
    breaks = detect_epoch_breaks(sustained, cfg)
    # One reposition is one break. A camera that stays moved must not
    # re-qualify every window and spawn an epoch each time.
    ok &= check("sustained displacement is exactly one epoch break",
                len(breaks) == 1, f"{len(breaks)}")

    # ...but a camera that returns and moves again is two separate events.
    twice = []
    for t in range(0, 6000, 200):
        twice.append(AlignmentResult(FEATURES, float(t), COMPENSATED,
                                     translation_px=25.0))
    for t in range(6000, 9000, 200):
        twice.append(AlignmentResult(FEATURES, float(t), COMPENSATED,
                                     translation_px=0.1))
    for t in range(9000, 15000, 200):
        twice.append(AlignmentResult(FEATURES, float(t), COMPENSATED,
                                     translation_px=25.0))
    ok &= check("two separate displacements are two breaks",
                len(detect_epoch_breaks(twice, cfg)) == 2,
                f"{len(detect_epoch_breaks(twice, cfg))}")
    if breaks:
        ok &= check("the break is timed to where movement started",
                    abs(breaks[0].pts_ms - 3000.0) < 1e-6, f"{breaks[0].pts_ms}")
        ok &= check("it says attribution is blocked until reapproval",
                    "blocked" in breaks[0].reason, breaks[0].reason[:70])

    print("\nall methods receive identical inputs (PRD 4)")
    results = compare(reference, shift(reference, 12, 5), support, cfg)
    ok &= check("every method reports the same support coverage",
                len({round(r.support_coverage, 6) for r in results}) == 1,
                str([round(r.support_coverage, 4) for r in results]))
    ok &= check("every result serialises with its method and reason",
                all(r.to_row()["method"] and r.to_row()["reason"] for r in results))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
