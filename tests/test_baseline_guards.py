"""Phase 3 gate: the four baseline guards (PRD 6.3.8, 6.3.9).

Every guard here exists because its absence was measured producing wrong output
on real footage, and each test carries the numbers from that measurement. A guard
removed by a future refactor fails here with its original evidence attached.

Also covers PRD 6.3.9: baseline learning excludes quality-affected spans and
records which windows were excluded and why. A baseline that learns through a
blackout learns that darkness is normal.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.baseline import (  # noqa: E402
    IDLE, TYPING, UNKNOWN, Baseline, BaselineConfig, ZoneWindow,
    environmental_zones, learn, regime_of, score,
)

WINDOW_MS = 500.0


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def windows(seat: str, residuals, area=0.30, blocks=40, zone="desk", start=0.0):
    return [ZoneWindow(seat, zone, start + i * WINDOW_MS,
                       start + (i + 1) * WINDOW_MS, r,
                       area[i] if isinstance(area, list) else area,
                       blocks[i] if isinstance(blocks, list) else blocks)
            for i, r in enumerate(residuals)]


def main() -> int:
    ok = True
    cfg = BaselineConfig(min_windows=20, learn_ms=1_000_000.0)

    print("guard 1: the spread floor is relative to the seat's own median")
    # Real 720p CCTV: median residual ~0.011, MAD ~0.0016. An absolute floor of
    # 0.05 divides every z-score by 30 and the gate goes blind. Measured that
    # way: fixtures passed, real footage produced zero events with a peak z of
    # 1.5 where the true value was ~47.
    real = windows("S", [0.011 + (i % 5) * 0.0008 for i in range(60)])
    baselines = learn(real, cfg)
    base = baselines[("S", "desk")]
    ok &= check("learns the real scale", 0.005 < base.median < 0.02,
                f"median {base.median:.5f}")
    ok &= check("and a spread of the same order", base.spread < 0.005,
                f"spread {base.spread:.5f}")
    deviation, _ = base.deviation(0.085, cfg)
    ok &= check("a genuine excursion scores high, not 1.5", deviation > 20.0,
                f"z {deviation:.1f}")

    print("\nguard 2: an absolute floor stops a static seat exploding")
    # A static seat's median and MAD both approach zero, so a purely relative
    # floor collapses and a trivial fluctuation over a near-zero denominator
    # reports an enormous deviation from nothing. Measured: z = 16.6 with zero
    # moving blocks.
    static = windows("T", [0.0000] * 60, area=0.0, blocks=0)
    base = learn(static, cfg)[("T", "desk")]
    deviation, _ = base.deviation(0.002, cfg)
    ok &= check("a static seat cannot reach the start threshold",
                deviation < 4.0, f"z {deviation:.2f} on a 0.002 fluctuation")

    print("\nguard 3: area fraction, not an absolute block count (PRD 6.3.8)")
    # Measured on 03.CCTV Mobile Usage: an unoccupied desk row produced the
    # twelve highest-scoring events at 29.8 sigma against 9.5 for the occupied
    # seat, on 0.1% area motion against 4.65%.
    empty = windows("R-desk", [0.0130] * 60, area=0.0011, blocks=1)
    empty += windows("R-desk", [0.2609], area=0.0011, blocks=1, start=30_000.0)
    scored = score(empty, learn(empty, cfg), cfg)
    spike = next(s for s in scored if s.window.residual > 0.2)
    ok &= check("one stray block does not become an event",
                spike.deviation == 0.0, f"deviation {spike.deviation}")
    ok &= check("and the guard that fired is named",
                spike.suppressed_by == "below_area_fraction", spike.suppressed_by)

    occupied = windows("S-61", [0.1835] * 60, area=0.0465, blocks=76)
    occupied += windows("S-61", [0.9482] * 4, area=0.2625, blocks=76, start=30_000.0)
    scored = score(occupied, learn(occupied, cfg), cfg)
    real_event = [s for s in scored if s.window.residual > 0.9]
    ok &= check("a real event at the measured area survives",
                all(s.deviation > 4.0 for s in real_event),
                f"{[round(s.deviation, 1) for s in real_event]}")
    ok &= check("and is not suppressed",
                all(not s.suppressed for s in real_event))

    print("\nguard 4: persistence as well as area (PRD 6.3.8)")
    # One window above the area threshold is not activity, whatever its area.
    flicker = windows("U", [0.01] * 30, area=0.01, blocks=1)
    flicker += windows("U", [0.9], area=0.60, blocks=90, start=20_000.0)
    flicker += windows("U", [0.01] * 10, area=0.01, blocks=1, start=25_000.0)
    scored = score(flicker, learn(flicker, cfg), cfg)
    single = next(s for s in scored if s.window.area_ratio > 0.5)
    ok &= check("a single high-area window is not persistent",
                single.suppressed_by == "not_persistent", single.suppressed_by)

    sustained = windows("V", [0.01] * 30, area=0.01, blocks=1)
    sustained += windows("V", [0.9] * 4, area=0.60, blocks=90, start=20_000.0)
    scored = score(sustained, learn(sustained, cfg), cfg)
    run = [s for s in scored if s.window.area_ratio > 0.5]
    ok &= check("a run of them is", all(not s.suppressed for s in run),
                f"{[s.suppressed_by for s in run]}")

    print("\nthe regime guard: events must not become normal")
    # Typing in a CBT hall is near-continuous, so a working seat sits above its
    # idle centre most of the time. Below the fraction, the excursions are
    # events -- and treating them as a regime trains the baseline to ignore
    # exactly what the system exists to find. Measured: recall fell to 1/3.
    rare = windows("W", [0.010] * 56 + [0.9] * 4)
    base = learn(rare, cfg)[("W", "desk")]
    ok &= check("four excursions in sixty windows are not a regime",
                base.typing_median is None,
                f"typing_median {base.typing_median}")

    # A realistic CBT distribution: idle around 0.010, typing around 0.030,
    # roughly a 3x separation with overlap. Measured behaviour of the split
    # across the band:
    #
    #     distribution                     median    MAD     above cut   regime
    #     10% excursions (events)          0.0099  0.0014      0.10       no
    #     40% typing, 3x separation        0.0115  0.0043      0.35      yes
    #     60% typing, 3x separation        0.0255  0.0091      0.00       no
    #
    # The 60% row is not a miss. Once a seat types most of the time, typing IS
    # its baseline -- the median has moved into the typing mode and the idle
    # windows are the minority tail below it. There is no second regime to
    # learn, and inventing one would mean scoring typing against itself.
    import random
    random.seed(3)
    busy = windows("X", [round(random.gauss(0.010, 0.002), 5) for _ in range(36)]
                   + [round(random.gauss(0.030, 0.005), 5) for _ in range(24)])
    base = learn(busy, cfg)[("X", "desk")]
    ok &= check("a realistic typing minority is a regime",
                base.typing_median is not None,
                f"typing_median {base.typing_median}")
    ok &= check("and a typing window is judged against the typing centre",
                regime_of(ZoneWindow("X", "desk", 0, 500, 0.032, 0.3, 40), base, cfg)
                == TYPING)
    ok &= check("while a quiet one is judged against idle",
                regime_of(ZoneWindow("X", "desk", 0, 500, 0.010, 0.3, 40), base, cfg)
                == IDLE)

    # A seat that types most of the time has typing as its baseline already.
    random.seed(4)
    mostly = windows("X2", [round(random.gauss(0.010, 0.002), 5) for _ in range(24)]
                     + [round(random.gauss(0.030, 0.005), 5) for _ in range(36)])
    base2 = learn(mostly, cfg)[("X2", "desk")]
    ok &= check("a seat that types most of the time needs no second regime",
                base2.typing_median is None and base2.median > 0.02,
                f"median {base2.median:.4f}, typing_median {base2.typing_median}")

    print("\nquality-affected spans are excluded and recorded (PRD 6.3.9)")
    mixed = windows("Y", [0.01] * 60)

    def eligible(start_ms, end_ms):
        if 5_000.0 <= start_ms < 12_000.0:
            return False, "blackout"
        return True, ""

    base = learn(mixed, cfg, eligible=eligible)[("Y", "desk")]
    ok &= check("excluded windows are counted", base.excluded_windows == 14,
                f"{base.excluded_windows}")
    ok &= check("with the reason kept",
                all(e.reason == "blackout" for e in base.exclusions),
                "PRD 6.3.9 requires included and excluded windows recorded")
    ok &= check("excluded time is recorded", base.excluded_ms == 7000.0,
                f"{base.excluded_ms}")
    ok &= check("the baseline learned from the rest",
                base.sample_windows == 46, f"{base.sample_windows}")

    print("\na baseline with too little to learn from refuses")
    thin = windows("Z", [0.01] * 5)
    base = learn(thin, cfg)[("Z", "desk")]
    ok &= check("marked degenerate", base.degenerate)
    ok &= check("and says why", "eligible window" in base.degenerate_reason,
                base.degenerate_reason)
    deviation, regime = base.deviation(99.0, cfg)
    ok &= check("a degenerate baseline scores nothing", deviation == 0.0)
    ok &= check("and reports an unknown regime", regime == UNKNOWN)

    mostly_excluded = learn(windows("Q", [0.01] * 30), cfg,
                            eligible=lambda s, e: (s > 12_000.0, "blackout"))
    base = mostly_excluded[("Q", "desk")]
    ok &= check("a mostly-excluded seat is degenerate, not confidently wrong",
                base.degenerate, base.degenerate_reason)
    ok &= check("and names the quality exclusion",
                "excluded by quality" in base.degenerate_reason,
                base.degenerate_reason)

    print("\npersistent zones are environmental, not behavioural")
    fan = windows("Fan", [0.5] * 100, area=0.40, blocks=50)
    candidate = windows("Seat", [0.5] * 10, area=0.40, blocks=50)
    found = environmental_zones(fan + candidate, cfg, total_ms=50_000.0)
    ok &= check("a zone active most of the recording is environmental",
                ("Fan", "desk") in found, str(found))
    ok &= check("a candidate's occasional activity is not",
                ("Seat", "desk") not in found, str(found))
    ok &= check("persistence is reported, not just a flag",
                found[("Fan", "desk")] >= 0.5, str(found))

    print("\nno baseline means no score, said out loud")
    orphan = score(windows("Nobody", [0.5] * 3), {}, cfg)
    ok &= check("an unknown seat-zone scores zero",
                all(s.deviation == 0.0 for s in orphan))
    ok &= check("with the reason named",
                all(s.suppressed_by == "no_baseline" for s in orphan))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
