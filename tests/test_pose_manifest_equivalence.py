"""Phase 4 gate: identical manifest rows for every pose model (PRD 4, 9).

PRD 4 calls this mandatory and says why:

> Run tracking and RTMPose/RTMO/AlphaPose on the identical pose manifest. [...]
> otherwise an input/crop/threshold change can be incorrectly credited to a
> model.

The failure this prevents is not exotic. It is a run where one configuration was
given crops from a slightly later commit, scores three points better, and gets
selected. Nothing in the numbers looks wrong. So equivalence is made checkable
rather than asserted: every configuration records the digest of the rows it was
actually handed, and a mismatch turns the comparison into a diagnostic rather
than a result.

Also covered here, because both are places where a missing measurement can be
dressed up as a measurement:

  * the four observability states of PRD 9. `occluded` (a hand under a desk),
    `not_resolvable_at_source_scale` (a camera that cannot show a hand at all),
    and `not_detected` are three different facts, and collapsing them loses the
    one that decides whether better analysis or a better camera is the answer.
  * PRD 9's floor on single observations: "A single neck turn or low-confidence
    wrist must never create a high-priority event."
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.calibration import (  # noqa: E402
    AMBIGUOUS_SEAT, APPROVED, STATIC_BACKGROUND, CalibrationProfile,
    MaskRegion, Seat,
)
from pipeline.person_detection import DETECTED, PersonBox, SampleRow  # noqa: E402
from pipeline.pose import (  # noqa: E402
    ALPHAPOSE, L_SHOULDER, L_WRIST, NOSE, NOT_DETECTED, NOT_RESOLVABLE,
    OCCLUDED, R_SHOULDER, R_WRIST, RTMO, RTMPOSE, VISIBLE, PoseConfig, PoseRow,
    compare, evidence, freeze, manifest_digest, observe, pose_row_id, run,
    unavailable,
)
from pipeline.seat_association import BY_GEOMETRY, SeatAssociation  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_profile() -> CalibrationProfile:
    return CalibrationProfile(
        camera_id="cam-t", epoch_id="e1", version=1, frame_w=1280, frame_h=720,
        operator="test", reviewer="test-reviewer", approval_state=APPROVED,
        seats=[Seat(seat_id="S-A",
                    zones={"seat_context": rect(100, 300, 400, 600),
                           "desk_work_surface": rect(100, 500, 400, 600),
                           "upper_body": rect(100, 300, 400, 500),
                           "lap_bag": rect(100, 560, 400, 600)},
                    desk_line_y=520.0)],
        masks=[MaskRegion(STATIC_BACKGROUND, rect(0, 0, 1280, 200))])


def sample(row_id: str, pts: float) -> SampleRow:
    return SampleRow(row_id=row_id, event_id="EV-1", seat_id="S-A",
                     source_video="v.mkv", source_video_sha256="abc123",
                     pts_ms=pts, calibration_version="cam-t/e1/v1",
                     camera_epoch="e1")


def association(row_id: str, pts: float, box, state="S-A") -> SeatAssociation:
    return SeatAssociation(row_id=row_id, pts_ms=pts, track_id=1,
                           seat_id=state if state == "S-A" else None,
                           state=state, rule=BY_GEOMETRY, reason="test",
                           scores={"S-A": 0.9}, box=box)


def keypoints(confidence=0.9, wrist_y=480.0, wrist_conf=None,
              nose_y=330.0) -> np.ndarray:
    """A plausible seated COCO-17 pose. Shoulders at y=360, x 180 and 300."""
    kp = np.zeros((17, 3), dtype=float)
    kp[NOSE] = (240.0, nose_y, confidence)
    kp[L_SHOULDER] = (300.0, 360.0, confidence)
    kp[R_SHOULDER] = (180.0, 360.0, confidence)
    kp[L_WRIST] = (290.0, wrist_y,
                   confidence if wrist_conf is None else wrist_conf)
    kp[R_WRIST] = (190.0, wrist_y,
                   confidence if wrist_conf is None else wrist_conf)
    return kp


def main() -> int:
    ok = True
    profile = make_profile()

    print("row identity is content-derived and stable across runs")
    first = pose_row_id("abc123", 1000.0, (100.0, 300.0, 300.0, 600.0))
    again = pose_row_id("abc123", 1000.0, (100.0, 300.0, 300.0, 600.0))
    other = pose_row_id("abc123", 1000.0, (101.0, 300.0, 300.0, 600.0))
    ok &= check("the same row hashes the same twice", first == again, first)
    ok &= check("a different crop is a different row", first != other)

    print("\nthe manifest freezes from associations, skipping interpolated boxes")
    rows_by_id = {f"r{i}": sample(f"r{i}", i * 500.0) for i in range(6)}
    associations = [association(f"r{i}", i * 500.0,
                                (150.0, 300.0, 350.0, 600.0)) for i in range(6)]
    # An interpolated box has no manifest row behind it, so it cannot be posed.
    associations.append(association("", 250.0, (150.0, 300.0, 350.0, 600.0)))
    manifest = freeze(associations, rows_by_id)
    ok &= check("only real rows are frozen", len(manifest) == 6,
                f"{len(manifest)} rows from {len(associations)} associations")
    ok &= check("rows are PTS ordered",
                [r.pts_ms for r in manifest] == sorted(r.pts_ms for r in manifest))
    ok &= check("every row carries the video hash",
                all(r.source_video_sha256 == "abc123" for r in manifest))
    ok &= check("every row carries seat candidates, not just a seat",
                all(isinstance(r.seat_candidates, dict) for r in manifest))

    print("\nreference annotations in an upstream row are refused, not carried")
    leaked = sample("rX", 0.0)
    leaked.gate_state = "pass"
    object.__setattr__(leaked, "reference_id", "REF-1")
    try:
        freeze([association("rX", 0.0, (150.0, 300.0, 350.0, 600.0))],
               {"rX": leaked})
        ok &= check("freeze refuses a leaked reference field", False)
    except ValueError as exc:
        ok &= check("freeze refuses a leaked reference field", True, str(exc)[:70])

    print("\nevery configuration is handed byte-identical rows")
    digest = manifest_digest(manifest)
    ok &= check("the digest is stable", manifest_digest(manifest) == digest,
                digest[:16])
    ok &= check("order does not change the digest",
                manifest_digest(list(reversed(manifest))) == digest)

    calls: dict[str, list[str]] = {}

    def runner_for(config_id):
        def run_row(row: PoseRow):
            calls.setdefault(config_id, []).append(row.pose_row_id)
            return [keypoints()]
        return run_row

    results = [run(manifest, runner_for(c), c, profile) for c in (RTMPOSE, RTMO)]
    ok &= check("both configurations saw the same row IDs",
                calls[RTMPOSE] == calls[RTMO],
                f"{len(calls[RTMPOSE])} rows each")
    ok &= check("both recorded the same digest",
                len({r.manifest_digest for r in results}) == 1)

    print("\na configuration given different rows is caught, not compared")
    shifted = list(manifest[:-1])
    diverged = run(shifted, runner_for("diverged"), "diverged_config", profile)
    report = compare(results + [diverged], manifest)
    ok &= check("equivalence is reported VIOLATED",
                report["manifest_equivalence"] == "VIOLATED",
                report["manifest_equivalence"])
    ok &= check("the offending configuration is named",
                "diverged_config" in report["mismatched_configs"])
    ok &= check("the note forbids selection from these numbers",
                "NOT comparable" in report["equivalence_note"])

    print("\na clean comparison names no winner and keeps the absent")
    blocked = unavailable(ALPHAPOSE, "licence_blocked",
                          "checkpoint and licence unconfirmed", manifest)
    clean = compare(results + [blocked], manifest)
    ok &= check("equivalence verified", clean["manifest_equivalence"] == "verified")
    ok &= check("the blocked configuration is still a row",
                ALPHAPOSE in [e["config_id"] for e in clean["unavailable"]])
    ok &= check("its state and reason survive",
                clean["unavailable"][0]["state"] == "licence_blocked")
    ok &= check("no configuration is called best",
                "none" in clean["selection"].lower(), clean["selection"][:60])

    print("\nthe four observability states stay distinguishable (PRD 9)")
    big = PoseRow(pose_row_id="p1", row_id="r1", event_id="EV-1",
                  source_video_sha256="abc123", pts_ms=0.0, frame_index=None,
                  box=(150.0, 300.0, 350.0, 600.0), seat_state="S-A")

    seen = observe(keypoints(), big, RTMPOSE, profile)
    ok &= check("a confident wrist above the desk is visible",
                seen.state(L_WRIST) == VISIBLE, seen.state(L_WRIST))

    # Desk line is y=520. A low-confidence wrist below it is the desk in the
    # way, which is a different fact from the estimator having failed.
    hidden = observe(keypoints(wrist_y=560.0, wrist_conf=0.05), big, RTMPOSE,
                     profile)
    ok &= check("a low-confidence wrist below the desk is occluded",
                hidden.state(L_WRIST) == OCCLUDED, hidden.state(L_WRIST))

    absent = keypoints()
    absent[L_WRIST] = (0.0, 0.0, 0.0)
    gone = observe(absent, big, RTMPOSE, profile)
    ok &= check("a wrist with no response at all is not_detected",
                gone.state(L_WRIST) == NOT_DETECTED, gone.state(L_WRIST))

    # A 40 px person cannot support a hand keypoint whatever the estimator says.
    tiny = PoseRow(pose_row_id="p2", row_id="r2", event_id="EV-1",
                   source_video_sha256="abc123", pts_ms=0.0, frame_index=None,
                   box=(150.0, 300.0, 190.0, 340.0), seat_state="S-A")
    unresolvable = observe(keypoints(), tiny, RTMPOSE, profile)
    ok &= check("a confident keypoint on a tiny person is not_resolvable",
                unresolvable.state(L_WRIST) == NOT_RESOLVABLE,
                unresolvable.state(L_WRIST))
    ok &= check("and the confidence did not rescue it",
                unresolvable.state(NOSE) == NOT_RESOLVABLE)
    ok &= check("no pose context is derived from it",
                unresolvable.head_pitch is None and
                not unresolvable.wrist_in_lap_context, unresolvable.note[:60])
    ok &= check("all four states are distinct",
                len({VISIBLE, OCCLUDED, NOT_DETECTED, NOT_RESOLVABLE}) == 4)

    print("\nlap context needs the wrist clearly past the desk line")
    edge = observe(keypoints(wrist_y=524.0), big, RTMPOSE, profile)
    ok &= check("a hand on the desk edge is not a hand in a lap",
                not edge.wrist_in_lap_context and edge.wrist_in_desk_context)
    deep = observe(keypoints(wrist_y=580.0), big, RTMPOSE, profile)
    ok &= check("a hand well below the desk line is lap context",
                deep.wrist_in_lap_context)

    print("\na single observation never becomes a sustained signal (PRD 9)")
    one = evidence([deep], "EV-1")
    ok &= check("one lap frame is not sustained", not one.sustained, one.reason)
    many = evidence([deep, deep, deep], "EV-1")
    ok &= check("three are", many.sustained, many.reason)
    ok &= check("the reach count counts transitions, not frames",
                many.repeated_reach == 1,
                f"{many.repeated_reach} reaches from 3 consecutive lap frames")
    alternating = evidence([deep, edge, deep, edge, deep], "EV-1")
    ok &= check("an in-out-in pattern is three reaches",
                alternating.repeated_reach == 3,
                str(alternating.repeated_reach))

    print("\nhead dwell is baseline-relative, never a universal angle (PRD 17.6)")
    upright = observe(keypoints(nose_y=330.0), big, RTMPOSE, profile)
    down = observe(keypoints(nose_y=395.0), big, RTMPOSE, profile)
    ok &= check("pitch is resolved for both",
                upright.head_pitch is not None and down.head_pitch is not None,
                f"{upright.head_pitch:.3f} vs {down.head_pitch:.3f}")
    ok &= check("looking down reads as a larger pitch",
                down.head_pitch > upright.head_pitch)
    # The same absolute pitch is a dwell against a tight baseline and normal
    # against a loose one. That is the whole point of the rule.
    tight = evidence([down, down, down], "EV-1", baseline_pitch=-0.2,
                     baseline_mad=0.02)
    loose = evidence([down, down, down], "EV-1", baseline_pitch=-0.2,
                     baseline_mad=1.0)
    ok &= check("a dwell against a tight per-seat baseline",
                tight.head_dwell_frames == 3, str(tight.head_dwell_frames))
    ok &= check("the identical pose is normal against a loose one",
                loose.head_dwell_frames == 0, str(loose.head_dwell_frames))
    ok &= check("with no baseline supplied, no dwell is claimed",
                evidence([down, down], "EV-1").head_dwell_frames == 0)

    print("\ncross-seat ambiguity is reported, not resolved")
    a = observe(keypoints(), big, RTMPOSE, profile)
    other_seat = PoseRow(pose_row_id="p3", row_id="r3", event_id="EV-1",
                         source_video_sha256="abc123", pts_ms=500.0,
                         frame_index=None, box=(150.0, 300.0, 350.0, 600.0),
                         seat_state="S-B")
    b = observe(keypoints(), other_seat, RTMPOSE, profile)
    mixed = evidence([a, b], "EV-1")
    ok &= check("two seats across one event is flagged",
                mixed.cross_seat_ambiguity)
    ok &= check("and the event's seat is ambiguous, not the first one seen",
                mixed.seat_state == AMBIGUOUS_SEAT, mixed.seat_state)

    print("\na runner that raises is recorded, not swallowed")
    def broken(row):
        raise RuntimeError("checkpoint absent")
    failed = run(manifest, broken, "broken_config", profile)
    ok &= check("state is launch_failed", failed.state == "launch_failed",
                failed.state)
    ok &= check("the exception text survives into metrics",
                "checkpoint absent" in failed.metrics()["reason"])
    ok &= check("an unavailable config reports no flattering zeros",
                "note" in failed.metrics())

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
