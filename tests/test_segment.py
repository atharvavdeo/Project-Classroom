"""Phase 4 validation gate.

Driven by synthetic deviation timelines rather than video, so each state-machine
transition can be exercised in isolation. The acceptance criterion that matters
most is fragmentation: one sustained action must produce exactly one event.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import segment  # noqa: E402
from classroom.config import SegmentConfig  # noqa: E402
from classroom.motion import ScoredWindow, Window, ZoneStats  # noqa: E402

WINDOW = 0.5
SEAT = "A"


def timeline(values: list[float], below: list[float] | None = None) -> list[ScoredWindow]:
    """Build a scan output whose seat deviation follows `values`."""
    out: list[ScoredWindow] = []
    for i, z in enumerate(values):
        b = below[i] if below else 0.0
        window = Window(
            t_start=i * WINDOW, t_end=(i + 1) * WINDOW, frames=12,
            global_dx=0.0, global_dy=0.0, global_ratio=0.0, luma_mean=130.0,
        )
        window.zones[(SEAT, "desk")] = ZoneStats(0, 0, 0, 0, 0, max(z, 0.0), b)
        out.append(ScoredWindow(window, {(SEAT, "desk"): z}, {(SEAT, "desk"): "idle"}))
    return out


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    ok = True
    cfg = SegmentConfig(
        start_z=4.0, stop_z=2.0, min_duration_s=1.2,
        merge_gap_s=2.5, max_duration_s=90.0,
    )

    print("hysteresis: sustained action produces exactly one event")
    # 10 windows (5 s) above start, surrounded by quiet.
    events = segment.segment(timeline([0] * 6 + [6] * 10 + [0] * 6), cfg)
    ok &= check("one event, not fragmented", len(events) == 1, f"{len(events)}")
    if events:
        e = events[0]
        ok &= check("starts at the first hot window", abs(e.t_start - 3.0) < 1e-6,
                    f"{e.t_start}")
        ok &= check("ends when activity stops", abs(e.t_end - 8.0) < 1e-6, f"{e.t_end}")
        ok &= check("peak deviation captured", e.peak_deviation == 6.0)
        ok &= check("driving zone recorded", e.driving_zone == "desk")

    print("\nhysteresis: the gap between thresholds prevents fragmentation")
    # Dips to 3.0 -- below start (4.0) but above stop (2.0). Must stay one event.
    events = segment.segment(
        timeline([0] * 4 + [6, 6, 3, 3, 6, 6, 3, 6, 6] + [0] * 6), cfg
    )
    ok &= check("dips above stop_z do not split", len(events) == 1, f"{len(events)}")

    print("\nimpulse rejection")
    ok &= check("single spike rejected",
                len(segment.segment(timeline([0] * 5 + [9] + [0] * 5), cfg)) == 0)
    ok &= check("two-window spike below min_duration rejected",
                len(segment.segment(timeline([0] * 5 + [9, 9] + [0] * 5), cfg)) == 0)
    ok &= check("three-window burst clears min_duration",
                len(segment.segment(timeline([0] * 5 + [9, 9, 9] + [0] * 5), cfg)) == 1)

    print("\nmerge gap")
    # Two bursts separated by 2 quiet windows (1.0 s) -- inside the 2.5 s gap.
    near = timeline([0] * 4 + [6] * 4 + [0, 0] + [6] * 4 + [0] * 5)
    ok &= check("bursts inside the merge gap become one event",
                len(segment.segment(near, cfg)) == 1,
                f"{len(segment.segment(near, cfg))}")
    # Separated by 8 quiet windows (4.0 s) -- beyond the gap.
    far = timeline([0] * 4 + [6] * 4 + [0] * 8 + [6] * 4 + [0] * 5)
    far_events = segment.segment(far, cfg)
    ok &= check("bursts beyond the merge gap split", len(far_events) == 2,
                f"{len(far_events)}")
    if len(far_events) == 2:
        ok &= check("first event ends where activity stopped",
                    abs(far_events[0].t_end - 4.0) < 1e-6, f"{far_events[0].t_end}")
        ok &= check("events do not overlap",
                    far_events[0].t_end <= far_events[1].t_start)

    print("\nmaximum duration forces a split")
    short_cfg = SegmentConfig(start_z=4.0, stop_z=2.0, min_duration_s=1.2,
                              merge_gap_s=2.5, max_duration_s=3.0)
    long_events = segment.segment(timeline([0] * 2 + [6] * 30 + [0] * 2), short_cfg)
    ok &= check("long stretch is split", len(long_events) > 1, f"{len(long_events)}")
    ok &= check("no event exceeds max duration",
                all(e.duration <= short_cfg.max_duration_s + WINDOW for e in long_events),
                f"max {max(e.duration for e in long_events):.2f}s")

    print("\nsuppressed windows never start an event")
    # apply_baselines clamps deviation to <=0 on exposure steps and camera bumps.
    ok &= check("clamped deviations produce nothing",
                len(segment.segment(timeline([0, -1, 0, -3, 0] * 4), cfg)) == 0)

    print("\nrunning to end of stream")
    tail = segment.segment(timeline([0] * 4 + [6] * 6), cfg)
    ok &= check("event still closed at stream end", len(tail) == 1, f"{len(tail)}")
    if tail:
        ok &= check("tail event ends at last hot window",
                    abs(tail[0].t_end - 5.0) < 1e-6, f"{tail[0].t_end}")

    print("\nbelow-desk evidence carried onto the event")
    with_below = timeline([0] * 4 + [6] * 6 + [0] * 4,
                          below=[0.0] * 4 + [0.0, 0.9, 0.4, 0.0, 0.0, 0.0] + [0.0] * 4)
    ev = segment.segment(with_below, cfg)
    ok &= check("peak below-desk fraction retained",
                bool(ev) and abs(ev[0].below_desk_peak - 0.9) < 1e-6,
                f"{ev[0].below_desk_peak if ev else 'none'}")

    print("\nmultiple seats segment independently")
    multi: list[ScoredWindow] = []
    a_vals = [0] * 4 + [6] * 6 + [0] * 6
    b_vals = [0] * 10 + [6] * 6 + [0] * 0
    for i in range(16):
        w = Window(t_start=i * WINDOW, t_end=(i + 1) * WINDOW, frames=12,
                   global_dx=0.0, global_dy=0.0, global_ratio=0.0, luma_mean=130.0)
        w.zones[("A", "desk")] = ZoneStats(0, 0, 0, 0, 0, max(a_vals[i], 0), 0)
        w.zones[("B", "desk")] = ZoneStats(0, 0, 0, 0, 0, max(b_vals[i], 0), 0)
        multi.append(ScoredWindow(
            w, {("A", "desk"): a_vals[i], ("B", "desk"): b_vals[i]},
            {("A", "desk"): "idle", ("B", "desk"): "idle"},
        ))
    both = segment.segment(multi, cfg)
    ok &= check("one event per seat", len(both) == 2, f"{len(both)}")
    ok &= check("seats correctly attributed",
                {e.seat_label for e in both} == {"A", "B"})
    ok &= check("output ordered by start time",
                all(a.t_start <= b.t_start for a, b in zip(both, both[1:])))

    print("\nsimultaneous seats produce overlapping events")
    sim: list[ScoredWindow] = []
    for i in range(16):
        z = 6.0 if 4 <= i < 10 else 0.0
        w = Window(t_start=i * WINDOW, t_end=(i + 1) * WINDOW, frames=12,
                   global_dx=0.0, global_dy=0.0, global_ratio=0.0, luma_mean=130.0)
        w.zones[("A", "desk")] = ZoneStats(0, 0, 0, 0, 0, z, 0)
        w.zones[("B", "desk")] = ZoneStats(0, 0, 0, 0, 0, z, 0)
        sim.append(ScoredWindow(w, {("A", "desk"): z, ("B", "desk"): z},
                                {("A", "desk"): "idle", ("B", "desk"): "idle"}))
    overlap = segment.segment(sim, cfg)
    ok &= check("two overlapping events, not one merged", len(overlap) == 2,
                f"{len(overlap)}")
    ok &= check("they share the same span",
                abs(overlap[0].t_start - overlap[1].t_start) < 1e-6)

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
