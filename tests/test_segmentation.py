"""Phase 3 gate: event segmentation (PRD 8).

Covers the scenarios PRD 16 Phase 3 names: sustained subtle motion, impulse,
adjacent seats, maximum-duration split, and epoch break — plus the boundary
rules and the requirement that every transition records its reason.

The impulse checks are the important ones. Manufacturing duration from a spike
is forbidden outright by PRD 8, and it reappeared in this module during
development: measuring persistence as elapsed wall time rather than sustained
activity let a dropped pen accumulate 1200 ms of "persistence" out of the quiet
windows that followed it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.baseline import Scored, ZoneWindow  # noqa: E402
from pipeline.segmentation import (  # noqa: E402
    ACTIVE, CANDIDATE, CLOSE_BOUNDARY, CLOSE_MAX_DURATION, CLOSE_TROUGH,
    PRIORITY_LOW, PRIORITY_NORMAL, BoundaryPolicy, SegmentConfig, segment,
    split_long, summarise,
)

WINDOW_MS = 500.0


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def w(i: int, deviation: float, seat: str = "A", zone: str = "desk",
      area: float = 0.30, blocks: int = 40, suppressed: str = "") -> Scored:
    window = ZoneWindow(seat, zone, i * WINDOW_MS, (i + 1) * WINDOW_MS,
                        0.5, area, blocks)
    return Scored(window, deviation, "idle", suppressed)


def quiet(a: int, b: int, **kw):
    return [w(i, 0.0, **kw) for i in range(a, b)]


def main() -> int:
    ok = True
    cfg = SegmentConfig()

    print("sustained action produces exactly one event")
    stream = quiet(0, 6) + [w(i, 6.0) for i in range(6, 16)] + quiet(16, 22)
    events, transitions = segment(stream, cfg)
    ok &= check("one event, not fragmented", len(events) == 1, f"{len(events)}")
    if events:
        ok &= check("normal priority", events[0].priority == PRIORITY_NORMAL)
        ok &= check("starts at the first hot window",
                    events[0].start_pts_ms == 3000.0, f"{events[0].start_pts_ms}")
        ok &= check("peak recorded", events[0].peak_deviation == 6.0)

    print("\nan impulse never becomes a ranked event (PRD 8)")
    events, _ = segment(quiet(0, 8) + [w(8, 12.0)] + quiet(9, 18), cfg)
    ok &= check("a single spike yields no normal-priority event",
                not [e for e in events if e.priority == PRIORITY_NORMAL],
                f"{[(e.priority, e.duration_ms) for e in events]}")
    ok &= check("but it is retained as low priority",
                len([e for e in events if e.priority == PRIORITY_LOW]) == 1,
                "PRD 8: keep short/subtle candidates, mark lower priority")
    if events:
        ok &= check("and carries no tail of stillness",
                    events[0].duration_ms <= WINDOW_MS,
                    f"{events[0].duration_ms} ms")

    events, _ = segment(quiet(0, 6) + [w(6, 12.0)] + quiet(7, 16)
                        + [w(16, 12.0)] + quiet(17, 24), cfg)
    ok &= check("two spikes beyond the bridge stay two low-priority records",
                len(events) == 2 and all(e.priority == PRIORITY_LOW for e in events),
                f"{[(e.priority, e.duration_ms) for e in events]}")

    print("\ngenuine intermittent activity survives bridging")
    intermittent = quiet(0, 4) + [w(4, 8.0), w(5, 8.0), w(6, 1.0), w(7, 1.0),
                                  w(8, 8.0), w(9, 8.0), w(10, 1.0), w(11, 8.0),
                                  w(12, 8.0)] + quiet(13, 20)
    events, _ = segment(intermittent, cfg)
    ok &= check("one event, not several", len(events) == 1, f"{len(events)}")
    ok &= check("at normal priority",
                events and events[0].priority == PRIORITY_NORMAL)

    print("\nstarting needs motion, not just an unusual number (PRD 8)")
    # A deviation of 30 on 0.1% area: the empty-desk-row case.
    events, _ = segment(quiet(0, 6) + [w(i, 30.0, area=0.001, blocks=1)
                                       for i in range(6, 16)] + quiet(16, 20), cfg)
    ok &= check("no event from a huge z on negligible area", not events,
                f"{len(events)}")

    events, _ = segment(quiet(0, 6) + [w(i, 30.0, area=0.30, blocks=0)
                                       for i in range(6, 16)] + quiet(16, 20), cfg)
    ok &= check("no event with zero moving blocks", not events, f"{len(events)}")

    print("\na suppressed window is recorded, not silently dropped")
    stream = quiet(0, 4) + [w(i, 8.0, area=0.001, blocks=1,
                              suppressed="below_area_fraction")
                            for i in range(4, 8)] + quiet(8, 12)
    events, transitions = segment(stream, cfg)
    ok &= check("no event", not events)
    noted = [t for t in transitions if "suppressed" in t.reason]
    ok &= check("but the suppression is in the transition log", noted,
                f"{len(noted)} transition(s)")
    ok &= check("naming which guard fired",
                noted and "below_area_fraction" in noted[0].reason,
                noted[0].reason if noted else "")

    print("\nadjacent seats never merge")
    both = []
    for i in range(20):
        deviation = 6.0 if 6 <= i < 16 else 0.0
        both.append(w(i, deviation, seat="A"))
        both.append(w(i, deviation, seat="B"))
    events, _ = segment(both, cfg)
    ok &= check("one event per seat", len(events) == 2, f"{len(events)}")
    ok &= check("both seats represented",
                {e.seat_id for e in events} == {"A", "B"})
    ok &= check("they overlap in time rather than merging",
                events[0].start_pts_ms == events[1].start_pts_ms)

    print("\nno bridging across a boundary (PRD 8)")
    walls = BoundaryPolicy([(5000.0, 5500.0, "decode_gap")])
    stream = quiet(0, 4) + [w(i, 8.0) for i in range(4, 20)]
    events, _ = segment(stream, cfg, boundaries=walls)
    ok &= check("the event is split at the wall", len(events) >= 2,
                f"{len(events)}")
    closed_at_wall = [e for e in events if e.close_reason.startswith(CLOSE_BOUNDARY)]
    ok &= check("the split names the boundary", closed_at_wall,
                f"{[e.close_reason for e in events]}")
    ok &= check("and names which kind",
                closed_at_wall and "decode_gap" in closed_at_wall[0].close_reason,
                closed_at_wall[0].close_reason if closed_at_wall else "")
    ok &= check("nothing spans the wall",
                all(not (e.start_pts_ms < 5000.0 < e.end_pts_ms) for e in events),
                f"{[(e.start_pts_ms, e.end_pts_ms) for e in events]}")

    print("\nan epoch break is a boundary too")
    epoch = BoundaryPolicy([(4000.0, 4001.0, "camera_repositioned epoch e1->e2")])
    events, _ = segment(quiet(0, 2) + [w(i, 8.0) for i in range(2, 16)], cfg,
                        boundaries=epoch)
    ok &= check("an event does not span a reposition", len(events) >= 2,
                f"{len(events)}")
    ok &= check("the reason survives to the event",
                any("camera_repositioned" in e.close_reason for e in events),
                f"{[e.close_reason for e in events]}")

    print("\nmaximum duration forces a split with lineage")
    long_cfg = SegmentConfig(max_duration_ms=3000.0)
    events, _ = segment(quiet(0, 2) + [w(i, 8.0) for i in range(2, 40)], long_cfg)
    ok &= check("a long stretch does not become one unusable event",
                len(events) >= 2, f"{len(events)}")
    ok &= check("closed for the stated reason",
                any(e.close_reason == CLOSE_MAX_DURATION for e in events),
                f"{[e.close_reason for e in events]}")

    from pipeline.segmentation import Event
    parent = Event("evt_A_desk_0001", "A", "desk", 0.0, 250_000.0, 10_000.0, 9.0)
    children = split_long(parent, SegmentConfig(max_duration_ms=90_000.0))
    ok &= check("splitting produces children", len(children) == 3, f"{len(children)}")
    ok &= check("each names its parent",
                all(c.parent_event_id == parent.event_id for c in children))
    ok &= check("the children cover the parent exactly",
                children[0].start_pts_ms == parent.start_pts_ms
                and children[-1].end_pts_ms == parent.end_pts_ms,
                "PRD 8: retain parent/child linkage")

    print("\na sustained trough splits an event")
    trough_cfg = SegmentConfig(trough_ms=2000.0, trough_z=0.5)
    stream = ([w(i, 8.0) for i in range(0, 8)]
              + [w(i, 0.1) for i in range(8, 16)]
              + [w(i, 8.0) for i in range(16, 24)])
    events, _ = segment(stream, trough_cfg)
    ok &= check("two actions separated by stillness are two events",
                len(events) == 2, f"{len(events)}")
    ok &= check("the first closes on the trough",
                events and events[0].close_reason == CLOSE_TROUGH,
                f"{[e.close_reason for e in events]}")

    print("\nevery transition records its reasoning (PRD 8)")
    events, transitions = segment(quiet(0, 4) + [w(i, 6.0) for i in range(4, 14)]
                                  + quiet(14, 20), cfg)
    ok &= check("transitions were logged", len(transitions) >= 3,
                f"{len(transitions)}")
    ok &= check("each carries a reason", all(t.reason for t in transitions))
    ok &= check("each carries the thresholds it was judged against",
                all("start_z" in t.thresholds for t in transitions))
    ok &= check("each carries the feature values",
                any(t.area_ratio > 0 for t in transitions))
    ok &= check("the state path is recorded",
                {t.to_state for t in transitions} >= {CANDIDATE, ACTIVE},
                str({t.to_state for t in transitions}))

    print("\nsummary reports retention, not just detections")
    events, _ = segment(quiet(0, 4) + [w(i, 6.0) for i in range(4, 14)]
                        + quiet(14, 30) + [w(30, 12.0)] + quiet(31, 40), cfg)
    report = summarise(events, total_ms=20_000.0)
    ok &= check("normal and low priority counted separately",
                report["normal_priority"] >= 1
                and report["low_priority_retained"] >= 1, str(report))
    ok &= check("candidate fraction uses normal-priority time only",
                report["candidate_fraction"] is not None
                and report["candidate_fraction"] < 1.0,
                f"{report['candidate_fraction']}")
    ok &= check("close reasons are broken out", report["by_close_reason"],
                str(report["by_close_reason"]))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
