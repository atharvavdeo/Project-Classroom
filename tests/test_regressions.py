"""Permanent regression tests for defects that were measured on real footage.

These are not restatements of a PRD rule. Each one carries **the numbers that
were observed when the defect fired**, so that removing a guard fails here with
the original evidence attached rather than with a generic assertion.

Every test in this file is closed by a measurement, and every one of these
defects produced *wrong output while reporting success*. None was found by
reading code.

A defect qualifies for this file when it satisfies all three:

  1. it was observed on real source footage, not on a fixture;
  2. it produced a plausible-looking wrong answer rather than an error;
  3. a future refactor could plausibly reintroduce it.

New defects of that class are added here; nothing is ever removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from classroom import motion, segment, verify  # noqa: E402
from classroom.config import BaselineConfig, MotionConfig, SegmentConfig  # noqa: E402
from classroom.motion import Baseline, ScoredWindow, Window, ZoneStats  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def one_window(key, residual, area_ratio, blocks, t_start=0.0):
    w = Window(t_start=t_start, t_end=t_start + 0.5, frames=8, global_dx=0.0,
               global_dy=0.0, global_ratio=0.0, luma_mean=130.0)
    w.zones[key] = ZoneStats(0, 0, 0, area_ratio, 0, residual, 0.0, blocks)
    return w


# ---------------------------------------------------------------------------
# R1 — an absolute moving-block count let an empty seat outscore an occupied one
# ---------------------------------------------------------------------------
#
# Observed on `03.CCTV Mobile Usage.mkv`, 281.9 s, 720p.
#
# An unoccupied desk row produced the TWELVE HIGHEST-SCORING EVENTS in the whole
# recording, peaking at 29.8 sigma, while the one genuinely occupied seat peaked
# at 9.5. The empty region is visually static across every sampled frame:
# monitors off, nobody present, nothing moving.
#
#     seat              median resid    MAD      mean area ratio   max z
#     R-desk (empty)       0.0130      0.0081        0.0011         29.8
#     S-61  (occupied)     0.1835      0.1616        0.0465          9.5
#
# Forty times less motion, three times the score. `min_moving_blocks = 1` is an
# absolute count, and a large zone accumulates stray blocks in proportion to its
# area, so one block of compression noise passed the guard and a near-zero MAD
# in the denominator turned it into an enormous standardised value.
#
# PRD 6.3.8 states the rule independently: "require moving area fraction plus
# persistence. Absolute moving-block count alone is forbidden."

R1_EMPTY = ("R-desk", "desk")
R1_BUSY = ("S-61", "desk")
R1_BASELINES = {
    R1_EMPTY: Baseline(0.0130, 0.0081, 0.13, None, None, 529),
    R1_BUSY: Baseline(0.1835, 0.1616, 0.95, None, None, 529),
}
# Measured extremes of each population across every window above the start
# threshold. The gap between them had nothing in it.
R1_EMPTY_MAX_AREA = 0.0333
R1_BUSY_MIN_AREA = 0.1458


def regression_r1(ok: bool) -> bool:
    print("\nR1  an empty seat must not outscore an occupied one")
    bcfg = BaselineConfig()

    quiet = motion.apply_baselines(
        [one_window(R1_EMPTY, 0.2609, 0.0011, 1)], R1_BASELINES,
        MotionConfig(), bcfg)[0]
    raw_z = (0.2609 - 0.0130) / 0.0081
    ok &= check("the raw z-score really was that large", raw_z > 25, f"{raw_z:.1f}")
    ok &= check("the area floor suppresses it",
                quiet.deviation[R1_EMPTY] <= 0.0, f"{quiet.deviation[R1_EMPTY]:.2f}")

    busy = motion.apply_baselines(
        [one_window(R1_BUSY, 0.9482, R1_BUSY_MIN_AREA, 76)], R1_BASELINES,
        MotionConfig(), bcfg)[0]
    ok &= check("a real event at the measured minimum area survives",
                busy.deviation[R1_BUSY] >= SegmentConfig().start_z,
                f"{busy.deviation[R1_BUSY]:.2f}")

    ok &= check("the floor sits inside the measured gap",
                R1_EMPTY_MAX_AREA < bcfg.min_area_ratio <= R1_BUSY_MIN_AREA,
                f"{R1_EMPTY_MAX_AREA} < {bcfg.min_area_ratio} <= {R1_BUSY_MIN_AREA}")
    ok &= check("an absolute block count alone cannot pass a zone",
                bcfg.min_area_ratio > 0.0,
                "PRD 6.3.8 forbids absolute moving-block count alone")
    return ok


# ---------------------------------------------------------------------------
# R2 — rolling-max smoothing manufactured duration from a single spike
# ---------------------------------------------------------------------------
#
# Real hand motion is intermittent: a reach pauses, the hand rests, the motion
# resumes, so the raw signal dips to baseline several times inside one action.
# Smoothing with a rolling maximum bridges those dips -- and also turns a
# DROPPED PEN into a three-second incident, because a rolling max cannot tell a
# gap between activity from a gap after an impulse.
#
# The fix is gap bridging, not smoothing: `candidate_bridge_s` measures time
# since activity was last above the stop threshold. An impulse never accumulates
# duration because there is no second burst to bridge to.
#
# PRD 8 states the rule independently: "Bridge only actual intermittent
# activity. Rolling-max smoothing that invents duration from a spike is
# prohibited."

def timeline(values: list[float], seat: str = "A") -> list[ScoredWindow]:
    out = []
    for i, z in enumerate(values):
        w = one_window((seat, "desk"), max(z, 0.0), 0.30, 40, t_start=i * 0.5)
        out.append(ScoredWindow(w, {(seat, "desk"): z}, {(seat, "desk"): "idle"}))
    return out


def regression_r2(ok: bool) -> bool:
    print("\nR2  an impulse must not become an event")
    cfg = SegmentConfig()

    # One window above the start threshold, nothing around it. A dropped pen.
    events = segment.segment(timeline([0] * 8 + [12.0] + [0] * 8), cfg)
    ok &= check("a single spike produces no event", len(events) == 0,
                f"{len(events)} events")

    # Two isolated spikes further apart than the bridge. Still not an action.
    spread = [0] * 6 + [12.0] + [0] * 8 + [12.0] + [0] * 6
    events = segment.segment(timeline(spread), cfg)
    ok &= check("two spikes beyond the bridge produce no event",
                len(events) == 0, f"{len(events)} events")

    # Genuine intermittent activity: bursts separated by dips shorter than the
    # bridge. This must survive, or the guard has simply gone deaf.
    intermittent = [0] * 4 + [8.0, 8.0, 1.0, 1.0, 8.0, 8.0, 1.0, 8.0, 8.0] + [0] * 6
    events = segment.segment(timeline(intermittent), cfg)
    ok &= check("intermittent real activity still becomes one event",
                len(events) == 1, f"{len(events)} events")
    if events:
        ok &= check("and it is not fragmented into several",
                    events[0].duration >= cfg.min_duration_s,
                    f"{events[0].duration:.2f}s")

    ok &= check("the bridge is a gap measure, not a smoothing window",
                cfg.candidate_bridge_s > 0 and cfg.min_duration_s > 0,
                f"bridge {cfg.candidate_bridge_s}s, min duration {cfg.min_duration_s}s")
    return ok


# ---------------------------------------------------------------------------
# R3 — the verifier described a person in a different seat, at confidence 1.0
# ---------------------------------------------------------------------------
#
# Observed live on `03.CCTV Mobile Usage.mkv`. Every event sent to the verifier
# belonged to an unoccupied desk row. Gemma returned fluent, detailed,
# internally coherent descriptions of "a candidate seated at a desk with a
# monitor, keyboard, and mouse" -- a man seated on the far LEFT of the frame, a
# different seat entirely -- at evidence_quality "sufficient", confidence 1.0.
#
# The contact sheet was the whole frame, six times. The seat under review was
# identified only as the string "seat": "R-desk" in the metadata JSON. A vision
# model cannot resolve a text label to a region of an image, so the attribution
# was never conveyed at all.
#
# Nothing could catch it: the output was schema-valid, the grammar was
# satisfied, the sanitizer found no verdict language, and the observations were
# true of somebody.
#
# PRD 11 states the rule independently: "a contact sheet of 4-8 chronological
# source frames with marked/labeled subject seat, magnified native subject
# crops... Whole-frame-only packets are prohibited."

R3_SEAT = [(700.0, 120.0), (1180.0, 120.0), (1180.0, 480.0), (700.0, 480.0)]


def regression_r3(ok: bool) -> bool:
    print("\nR3  the verifier packet must mark which seat is under review")
    frames = [np.full((720, 1280, 3), 90, dtype=np.uint8) for _ in range(6)]

    unmarked = verify.build_contact_sheet(frames, columns=3, tile=320)
    marked = verify.build_contact_sheet(frames, columns=3, tile=320,
                                        highlight=R3_SEAT)
    ok &= check("a marked sheet differs from an unmarked one",
                not np.array_equal(unmarked, marked))

    scale = 320 / 1280
    y0 = (320 - int(720 * scale)) // 2
    red = (marked[:, :, 2].astype(int) - marked[:, :, 1]) > 100
    ok &= check("every tile carries the mark",
                all(red[r * 320:(r + 1) * 320, c * 320:(c + 1) * 320].sum() > 0
                    for r in range(2) for c in range(3)))

    ys, xs = np.nonzero(red[:320, :320])
    expected = (y0 + 120 * scale, y0 + 480 * scale, 700 * scale, 1180 * scale)
    drawn = (ys.min(), ys.max(), xs.min(), xs.max())
    ok &= check("the mark lands on the seat's mapped rectangle",
                all(abs(a - b) <= 2 for a, b in zip(drawn, expected)),
                f"{drawn} vs {tuple(round(v, 1) for v in expected)}")

    sheet = verify.build_event_sheet(frames, R3_SEAT, columns=3, tile=320)
    ok &= check("a magnified crop of that seat is appended",
                sheet.shape[0] == marked.shape[0] + 320,
                f"{sheet.shape} vs {marked.shape}")

    ok &= check("the prompt says what the mark means",
                "red box" in verify.SYSTEM and "not the subject" in verify.SYSTEM)
    ok &= check("an empty marked box has a defined answer",
                "if the box is empty" in verify.SYSTEM
                and "insufficient" in verify.SYSTEM)
    ok &= check("the prompt forbids verdict language",
                "Never state or imply" in verify.SYSTEM)
    return ok


def main() -> int:
    ok = True
    print("Regression tests for defects measured on real footage.")
    print("Each carries the numbers observed when the defect fired.")
    ok = regression_r1(ok)
    ok = regression_r2(ok)
    ok = regression_r3(ok)
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
