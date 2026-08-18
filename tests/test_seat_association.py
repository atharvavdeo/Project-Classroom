"""Phase 4 gate: person-to-seat association (PRD 7).

The two prohibitions in PRD 7 are the whole test:

> Never assign by nearest-box centre alone. Never force a seat across shared
> desk/aisle boundaries.

Both describe producing an answer because an answer was wanted. So the cases
here are chosen to be exactly the ones where an answer is available and wrong:

  * a **leaning candidate** whose box centre sits over the neighbour while their
    seated footprint has not moved at all. Centre-based attribution gets this
    backwards, and it is not a rare pose in an examination hall.
  * a **declared ambiguous boundary**, where the operator has already recorded
    that the geometry cannot separate two seats. No score, however clean, may
    override an operator's recorded judgement.
  * **track continuity contradicting geometry**. Continuity may settle a tie
    between candidates the geometry proposed; it may never supply a seat the
    geometry scored at zero. That asymmetry is the point, and it is easy to
    implement backwards because "the track is usually in S-61" feels like strong
    evidence right up until the person stands up and walks away.
  * an **unapproved calibration**, which PRD 3 says blocks seat-attributed
    evidence outright.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.calibration import (  # noqa: E402
    AMBIGUOUS_SEAT, APPROVED, DRAFT, UNATTRIBUTED, CalibrationProfile, Seat,
    STATIC_BACKGROUND, MaskRegion,
)
from pipeline.person_detection import DETECTED, PersonBox  # noqa: E402
from pipeline.seat_association import (  # noqa: E402
    BY_CONTINUITY, BY_GEOMETRY, AssociationConfig, associate, associate_box,
    associate_track, summarise,
)
from pipeline.tracking import SEAT_ONLY, Track, TrackingResult, track  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def seat(seat_id, x1, x2, ambiguous_with=None):
    """A seat 300 px tall with all four required zones."""
    return Seat(
        seat_id=seat_id,
        zones={
            "seat_context": rect(x1, 300, x2, 600),
            "desk_work_surface": rect(x1, 500, x2, 600),
            "upper_body": rect(x1, 300, x2, 500),
            "lap_bag": rect(x1, 560, x2, 600),
        },
        desk_line_y=520.0,
        ambiguous_with=list(ambiguous_with or []),
    )


def profile(seats, state=APPROVED):
    out = CalibrationProfile(
        camera_id="cam-t", epoch_id="e1", version=1, frame_w=1280, frame_h=720,
        operator="test", reviewer="test-reviewer", approval_state=state,
        seats=seats,
        masks=[MaskRegion(STATIC_BACKGROUND, rect(0, 0, 1280, 200))])
    return out


def box(x1, y1, x2, y2, pts=0.0, track_id=None, row=""):
    return PersonBox(row_id=row or f"r{int(pts)}", pts_ms=pts,
                     x1=x1, y1=y1, x2=x2, y2=y2, score=0.9,
                     origin=DETECTED, track_id=track_id)


def main() -> int:
    ok = True

    # Two adjacent seats, no declared ambiguity: 100..400 and 400..700.
    left, right = seat("S-A", 100, 400), seat("S-B", 400, 700)
    left.adjacent, right.adjacent = ["S-B"], ["S-A"]
    clean = profile([left, right])
    blocking = clean.blocking_issues
    if blocking:
        print("  FAIL  test fixture itself does not validate:")
        for issue in blocking:
            print(f"        {issue}")
        return 1

    print("a squarely seated candidate resolves to their seat")
    seated = associate_box(box(150, 320, 350, 600), clean)
    ok &= check("resolved", seated.resolved, seated.state)
    ok &= check("to S-A", seated.seat_id == "S-A", str(seated.scores))
    ok &= check("by geometry", seated.rule == BY_GEOMETRY, seated.rule)

    print("\nthe leaning candidate: centre over the neighbour, footprint at home")
    # Box spans 250..560. Its centre x is 405, inside S-B. Its lower third
    # (y 500..600) still lies mostly in 250..400, which is S-A.
    leaning = box(250, 300, 560, 600)
    centre_x = (leaning.x1 + leaning.x2) / 2.0
    ok &= check("box centre is over S-B", 400 <= centre_x <= 700,
                f"centre x = {centre_x:.0f}")
    result = associate_box(leaning, clean)
    # The footprint straddles both seats, so the honest answer is either S-A or
    # a refusal -- never S-B, which is what the centre alone would have said.
    ok &= check("never attributed to the seat the centre is over",
                result.seat_id != "S-B", f"{result.state} {result.scores}")
    ok &= check("both seats retained as candidates", len(result.scores) == 2,
                str(result.scores))
    ok &= check("a reason in words is recorded", bool(result.reason),
                result.reason)

    print("\na declared ambiguous boundary is ambiguous whatever the scores say")
    a2, b2 = seat("S-C", 100, 420, ["S-D"]), seat("S-D", 380, 700, ["S-C"])
    a2.adjacent, b2.adjacent = ["S-D"], ["S-C"]
    overlapped = profile([a2, b2])
    ok &= check("fixture validates with the boundary declared",
                not overlapped.blocking_issues,
                str([str(i) for i in overlapped.blocking_issues]))
    straddling = associate_box(box(300, 300, 500, 600), overlapped)
    ok &= check("state is ambiguous_seat", straddling.state == AMBIGUOUS_SEAT,
                straddling.state)
    ok &= check("no seat is named", straddling.seat_id is None)
    ok &= check("all candidate scores retained", len(straddling.scores) >= 2,
                str(straddling.scores))

    print("\nan unapproved calibration blocks seat attribution (PRD 3)")
    draft = profile([seat("S-A", 100, 400)], state=DRAFT)
    refused = associate_box(box(150, 320, 350, 600), draft)
    ok &= check("unattributed", refused.state == UNATTRIBUTED, refused.state)
    ok &= check("the reason names the approval state",
                "not approved" in refused.reason, refused.reason)

    print("\ncontinuity settles a tie between candidates the geometry proposed")
    # Four boxes in S-A, then one straddling box the geometry cannot separate.
    boxes = [box(150, 320, 350, 600, pts=p, track_id=1) for p in
             (0.0, 500.0, 1000.0, 1500.0)]
    # The same straddling geometry proved ambiguous above: S-A 0.486 vs
    # S-B 0.518, a 0.032 margin under the 0.15 floor.
    boxes.append(box(250, 300, 560, 600, pts=2000.0, track_id=1))
    candidate = Track(track_id=1, boxes=boxes)
    associations = associate_track(candidate, clean)
    settled = associations[-1]
    ok &= check("the ambiguous frame is resolved", settled.resolved,
                f"{settled.state} via {settled.rule}")
    ok &= check("by continuity, and labelled as such",
                settled.rule == BY_CONTINUITY, settled.rule)
    ok &= check("to the seat the track sits in", settled.seat_id == "S-A",
                str(settled.scores))
    ok &= check("the reason states both sources of evidence",
                "ambiguous" in settled.reason and "track" in settled.reason,
                settled.reason)

    print("\ncontinuity never supplies a seat the geometry scored at zero")
    # Same track, but the last box is far away over S-B alone. Continuity says
    # S-A; the geometry says S-B is the only candidate. They contradict, and a
    # contradiction is not resolved by preferring the more numerous evidence.
    walked = [box(150, 320, 350, 600, pts=p, track_id=2) for p in
              (0.0, 500.0, 1000.0, 1500.0)]
    walked.append(box(500, 300, 690, 600, pts=2000.0, track_id=2))
    moved = associate_track(Track(track_id=2, boxes=walked), clean)
    last = moved[-1]
    ok &= check("the far box is not dragged back to the track's seat",
                last.seat_id != "S-A", f"{last.state} -> {last.seat_id}")
    ok &= check("it is attributed on its own geometry or refused",
                last.rule in (BY_GEOMETRY, "refused"), last.rule)

    print("\na short track has no majority worth applying")
    tiny = Track(track_id=3, boxes=[box(330, 300, 530, 600, pts=0.0, track_id=3),
                                    box(330, 300, 530, 600, pts=500.0, track_id=3)])
    short = associate_track(tiny, clean, AssociationConfig(min_resolved_boxes=3))
    ok &= check("two agreeing frames do not make a majority",
                all(a.rule != BY_CONTINUITY for a in short),
                str([a.rule for a in short]))

    print("\nthe no-track baseline gets geometry only, never continuity")
    no_track = TrackingResult(config_id=SEAT_ONLY, boxes=list(boxes))
    plain = associate(no_track, clean)
    ok &= check("no association came from continuity",
                all(a.rule != BY_CONTINUITY for a in plain),
                "seat_only_no_track must have no continuity rule available")
    ok &= check("it still resolves the unambiguous frames",
                sum(1 for a in plain if a.resolved) >= 4)

    print("\nhealth conditions that invalidate the polygons block attribution")
    blocked = associate_box(box(150, 320, 350, 600), clean,
                            health_state=["camera_repositioned"])
    ok &= check("unattributed under a reposition",
                blocked.state == UNATTRIBUTED, blocked.state)
    ok &= check("the reason names the condition",
                "camera_repositioned" in blocked.reason, blocked.reason)

    print("\nthe summary makes refusals visible rather than hiding them")
    report = summarise(associations + moved + plain)
    for field in ("resolved", "ambiguous_seat", "unattributed", "by_rule",
                  "continuity_share", "median_margin"):
        ok &= check(f"summary reports {field}", field in report)
    ok &= check("an empty association set is not a perfect result",
                "not a perfect-attribution result" in summarise([])["note"])

    print("\ntracking preserves box provenance end to end")
    tracked = track([box(150, 320, 350, 600, pts=p) for p in (0.0, 500.0, 1000.0)])
    ok &= check("one track over three co-located boxes",
                len(tracked.tracks) == 1, str(tracked.metrics()))
    ok &= check("every box carries an origin",
                all(b.origin in ("detected", "interpolated") for b in tracked.boxes))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
