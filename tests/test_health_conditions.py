"""Phase 1 gate: the ten health conditions (PRD 6.2).

Seven of the ten have no example anywhere in the available corpus. Those are
exercised by **injecting the condition into real decoded frames** and measuring
through the same code path ingest uses — `_frame_stats` on real pixels — rather
than by fabricating statistics. That way the test covers the measurement as well
as the threshold.

A fixture proves the detector fires. It does not prove the threshold is right
for real footage, and the gate checks that the distinction is preserved in the
output: an injected observation must be marked `fixture_only`.

The other half of this gate is the half that actually caught something. Every
detector is also run against **clean real footage with nothing injected**, and
must stay silent. That check is what exposed the first freeze detector
reporting `frozen_video` over 99.6% of a file with people walking through it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pipeline import health, ingest  # noqa: E402
from pipeline.health import HealthConfig, HealthObservation, HealthTimeline  # noqa: E402
from pipeline.timestamps import build_index  # noqa: E402

FRAMES_DIR = Path("data/frames")
CLEAN_VIDEO = Path("C:/DrishtiAI/04.CCTV Candidate Talking.mkv")

FPS = 25.0
INTERVAL_MS = 1000.0 / FPS


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def real_plane() -> np.ndarray | None:
    """One real luma plane from the extracted corpus frames."""
    candidates = sorted(FRAMES_DIR.glob("*.jpg"))
    if not candidates:
        return None
    image = cv2.imread(str(candidates[0]))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def stats_from(planes: list[np.ndarray], start_ms: float = 0.0):
    """Run real planes through the ingest measurement path."""
    out = []
    previous = None
    for i, plane in enumerate(planes):
        measured, previous = ingest._frame_stats(
            i, start_ms + i * INTERVAL_MS, plane, previous)
        out.append(measured)
    return out


def jitter(plane: np.ndarray, rng: np.random.Generator, sigma: float = 2.0
           ) -> np.ndarray:
    """Real footage is never two identical frames; sensor noise sees to that.

    Injected fixtures need it too, or a 'normal' stretch would read as frozen
    and the fixture would pass for the wrong reason.
    """
    noisy = plane.astype(np.float32) + rng.normal(0.0, sigma, plane.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main() -> int:
    ok = True
    cfg = HealthConfig()
    rng = np.random.default_rng(11)

    plane = real_plane()
    if plane is None:
        print("SKIP: no extracted frames under data/frames")
        return 0
    print(f"injecting into a real {plane.shape[1]}x{plane.shape[0]} luma plane\n")

    normal = [jitter(plane, rng) for _ in range(50)]

    print("blackout")
    dark = [jitter((plane * 0.02).astype(np.uint8), rng, 1.0) for _ in range(40)]
    found = health.detect_blackout(stats_from(normal + dark + normal), cfg)
    ok &= check("fires on a sustained near-black stretch", len(found) == 1,
                f"{len(found)}")
    if found:
        ok &= check("marked fixture_only",
                    found[0].validated_on == health.VALIDATED_ON_FIXTURE)
        ok &= check("excluded from baseline learning", found[0].excludes_from_baseline)
        ok &= check("an event may not bridge it", found[0].blocks_bridging)
        ok &= check("action is to retain and mark unavailable",
                    found[0].action == health.RETAIN_UNAVAILABLE, found[0].action)
    ok &= check("silent on clean frames",
                not health.detect_blackout(stats_from(normal), cfg))

    print("\nwhiteout")
    bright = [jitter(np.full_like(plane, 250), rng, 1.0) for _ in range(40)]
    found = health.detect_whiteout(stats_from(normal + bright + normal), cfg)
    ok &= check("fires on a sustained saturated stretch", len(found) == 1,
                f"{len(found)}")
    if found:
        ok &= check("pauses baseline adaptation",
                    found[0].action == health.PAUSE_BASELINE)
    ok &= check("silent on clean frames",
                not health.detect_whiteout(stats_from(normal), cfg))

    print("\nfrozen video")
    # A held frame: identical pixels, PTS still advancing. No jitter, because
    # that is exactly what a frozen encoder stops producing.
    held = [plane.copy() for _ in range(50)]
    found = health.detect_frozen(stats_from(normal + held + normal), cfg)
    ok &= check("fires when the image repeats while PTS advances",
                len(found) == 1, f"{len(found)}")
    if found:
        ok &= check("prevents false motion",
                    found[0].action == health.PREVENT_FALSE_MOTION)
        ok &= check("evidence records the observed difference",
                    "max_frame_delta" in found[0].evidence,
                    str(found[0].evidence))
    ok &= check("silent on clean jittered frames",
                not health.detect_frozen(stats_from(normal), cfg))

    print("\nblur")
    blurred = [jitter(cv2.GaussianBlur(plane, (21, 21), 6.0), rng)
               for _ in range(40)]
    combined = stats_from(normal + blurred + normal)
    found = health.detect_blur(combined, cfg)
    ok &= check("fires on a sustained soft stretch", len(found) >= 1, f"{len(found)}")
    if found:
        ok &= check("reduces model reliability rather than deleting the span",
                    found[0].action == health.REDUCE_MODEL_RELIABILITY)
        ok &= check("does not exclude the span from baseline learning",
                    not found[0].excludes_from_baseline,
                    "blur degrades object and pose, not motion")
        ok &= check("evidence names the per-camera baseline",
                    "baseline_sharpness" in found[0].evidence)
    ok &= check("silent on clean frames",
                not health.detect_blur(stats_from(normal), cfg))
    # The threshold is relative, so a uniformly soft camera must not be called
    # blurred: its own median is the reference.
    ok &= check("a uniformly soft camera is not blurred against itself",
                not health.detect_blur(stats_from(blurred), cfg),
                "PRD 6.2: sharpness relative to a per-camera baseline")

    print("\nexposure change")
    lifted = [jitter(np.clip(plane.astype(np.int16) + 40, 0, 255).astype(np.uint8),
                     rng) for _ in range(40)]
    found = health.detect_exposure_change(stats_from(normal + lifted), cfg)
    ok &= check("fires on a scene-wide luma step", len(found) >= 1, f"{len(found)}")
    if found:
        ok &= check("no baseline update through recovery",
                    found[0].action == health.FLAG_RECOVERY_NO_BASELINE)
        ok &= check("the recovery window extends past the step itself",
                    found[0].duration_ms >= cfg.exposure_recovery_ms,
                    f"{found[0].duration_ms} ms")
    ok &= check("silent on clean frames",
                not health.detect_exposure_change(stats_from(normal), cfg))

    print("\ndecode gap")
    records = [(i * INTERVAL_MS, i % 25 == 0, "I" if i % 25 == 0 else "P")
               for i in range(100)]
    records += [(records[-1][0] + 5000.0 + i * INTERVAL_MS, i % 25 == 0, "P")
                for i in range(100)]
    index = build_index(records)
    found = health.detect_decode_gaps(index)
    ok &= check("a missing stretch becomes one gap", len(found) == 1, f"{len(found)}")
    if found:
        ok &= check("the gap stays visible",
                    found[0].action == health.KEEP_GAP_VISIBLE)
        ok &= check("an event may not bridge it", found[0].blocks_bridging)
        ok &= check("gap duration is close to what was removed",
                    abs(found[0].duration_ms - 5000.0) < 2 * INTERVAL_MS,
                    f"{found[0].duration_ms:.1f} ms")
    clean_index = build_index([(i * INTERVAL_MS, False, "P") for i in range(200)])
    ok &= check("no gap on an unbroken sequence",
                not health.detect_decode_gaps(clean_index))

    print("\nvariable frame rate is not a gap")
    # File 04's real interval pattern: 120 ms with occasional 160 ms. Variable
    # pacing must not be reported as missing data.
    vfr = []
    t = 0.0
    for i in range(200):
        vfr.append((t, i % 8 == 0, "P"))
        t += 160.0 if i % 8 == 7 else 120.0
    vfr_index = build_index(vfr)
    ok &= check("VFR pacing produces no gaps", not vfr_index.gaps,
                f"{len(vfr_index.gaps)}")
    ok &= check("but it is recognised as variable rate", vfr_index.is_vfr,
                f"cv {vfr_index.interval_cv:.4f}")

    print("\ntimeline behaviour")
    observations = [
        HealthObservation(health.BLACKOUT, 1000.0, 2000.0),
        HealthObservation(health.BLUR, 1500.0, 3000.0),
        HealthObservation(health.ENVIRONMENTAL_MOTION, 8000.0, 9000.0),
    ]
    timeline = HealthTimeline(observations)
    ok &= check("conditions overlapping a window are found",
                len(timeline.over(1800.0, 1900.0)) == 2)
    ok &= check("a blackout window is not baseline eligible",
                not timeline.baseline_eligible(1000.0, 2000.0))
    ok &= check("a blur-only window is baseline eligible",
                timeline.baseline_eligible(2100.0, 2900.0),
                "blur degrades models, not the motion baseline")
    ok &= check("an event may not bridge a blackout",
                not timeline.may_bridge(900.0, 2100.0))
    ok &= check("an event may bridge environmental motion",
                timeline.may_bridge(8000.0, 9000.0),
                "it is penalised in ranking, not treated as missing data")
    # Overlapping conditions must not double-count the time they affect.
    ok &= check("affected time is a union, not a sum",
                abs(timeline.affected_ms() - 3000.0) < 1e-6,
                f"{timeline.affected_ms()} ms, not {1000 + 1500 + 1000}")

    print("\nvalidation refuses malformed observations")
    for label, build in (
        ("an unknown condition",
         lambda: HealthObservation("looks_odd", 0.0, 1.0)),
        ("a reversed interval",
         lambda: HealthObservation(health.BLUR, 2000.0, 1000.0)),
    ):
        raised = False
        try:
            build()
        except ValueError:
            raised = True
        ok &= check(f"refuses {label}", raised)

    ok &= check("every condition has a declared action",
                all(c in health.ACTION for c in health.CONDITIONS))

    print("\nclean real footage stays silent")
    if CLEAN_VIDEO.is_file():
        result = ingest.decode(CLEAN_VIDEO)
        timeline = HealthTimeline(
            health.observe(result, cfg, validated_on=health.VALIDATED_ON_SOURCE))
        counts = timeline.counts()
        ok &= check("no blackout on clean footage", counts[health.BLACKOUT] == 0)
        ok &= check("no whiteout on clean footage", counts[health.WHITEOUT] == 0)
        # This is the check that caught the first freeze detector reporting
        # 99.6% of a file with people moving through it.
        ok &= check("no frozen video on clean footage",
                    counts[health.FROZEN_VIDEO] == 0,
                    f"{counts[health.FROZEN_VIDEO]}")
        ok &= check("no decode gaps on a clean file", counts[health.DECODE_GAP] == 0)
        ok &= check("affected fraction is small",
                    timeline.affected_ms() / result.index.duration_ms < 0.05,
                    f"{timeline.affected_ms() / result.index.duration_ms:.4f}")
    else:
        print("  SKIP: clean source video not available")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
