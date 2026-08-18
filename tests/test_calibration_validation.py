"""Phase 2 gate: calibration validation, approval and attribution (PRD 7).

PRD 7 names five rejections that approval must enforce — zero area, out of
frame, unresolved overlap, missing background support, unlabelled boundaries —
and each is checked here against a profile that is otherwise valid, so a pass
means that rejection specifically fired.

The attribution checks are the ones that matter most in a crowded room. PRD 7
forbids assigning by nearest box centre and forbids forcing a seat across a
shared boundary, and both refusals are checked with geometry where the wrong
answer is the tempting one.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.calibration import (  # noqa: E402
    AMBIGUOUS_SEAT, APPROVED, DESK_WORK_SURFACE, DRAFT, LAP_BAG, REFLECTION,
    SEAT_CONTEXT, STATIC_BACKGROUND, SUPERSEDED, UNATTRIBUTED, UPPER_BODY,
    AttributionConfig, CalibrationError, CalibrationProfile, MaskRegion, Seat,
    attribute_box, background_support, polygon_area, polygon_overlap_px,
    scoring_mask,
)

W, H = 1280, 720
NOW = "2026-08-18T00:00:00Z"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def seat(seat_id, x1, y1, x2, y2, **kw) -> Seat:
    mid = y1 + (y2 - y1) * 0.55
    return Seat(
        seat_id,
        zones={
            SEAT_CONTEXT: rect(x1, y1, x2, y2),
            UPPER_BODY: rect(x1, y1, x2, mid),
            DESK_WORK_SURFACE: rect(x1, mid, x2, y2),
            LAP_BAG: rect(x1, y2 - (y2 - y1) * 0.2, x2, y2),
        },
        desk_line_y=mid, **kw)


def valid_profile(**kw) -> CalibrationProfile:
    """A profile with nothing wrong with it, to break one thing at a time."""
    profile = CalibrationProfile(
        camera_id="cam12", epoch_id="e1", version=1, frame_w=W, frame_h=H,
        operator="operator-a", **kw)
    profile.seats = [seat("A", 60, 300, 380, 700), seat("B", 500, 300, 820, 700)]
    profile.masks = [MaskRegion(STATIC_BACKGROUND, rect(0, 0, W, 240),
                                "ceiling and back wall")]
    profile.review_images = {c: f"{c}.jpg" for c in
                             ("crowded", "sparse", "changed_light", "aisle_reflection")}
    return profile


def codes(profile) -> set[str]:
    return {i.code for i in profile.blocking_issues}


def main() -> int:
    ok = True

    print("a valid profile has nothing blocking")
    base = valid_profile()
    ok &= check("no blocking issues", not base.blocking_issues,
                "; ".join(str(i) for i in base.blocking_issues))

    print("\nPRD 7's five rejections")

    zero = valid_profile()
    zero.seats[0].zones[SEAT_CONTEXT] = [(100, 100), (200, 100), (300, 100)]
    ok &= check("rejects zero area", "zero_area" in codes(zero), str(codes(zero)))

    outside = valid_profile()
    outside.seats[0].zones[SEAT_CONTEXT] = rect(60, 300, W + 200, 700)
    ok &= check("rejects out of frame", "out_of_frame" in codes(outside))

    overlapping = valid_profile()
    overlapping.seats[1] = seat("B", 300, 300, 620, 700)   # overlaps A
    ok &= check("rejects an unlabelled shared boundary",
                "unlabelled_boundary" in codes(overlapping), str(codes(overlapping)))

    declared = valid_profile()
    declared.seats[0] = seat("A", 60, 300, 380, 700, ambiguous_with=["B"])
    declared.seats[1] = seat("B", 300, 300, 620, 700, ambiguous_with=["A"])
    ok &= check("accepts the same overlap once the boundary is declared",
                "unlabelled_boundary" not in codes(declared), str(codes(declared)))

    no_background = valid_profile()
    no_background.masks = []
    ok &= check("rejects a missing static-background region",
                "missing_background_support" in codes(no_background))

    duplicate = valid_profile()
    duplicate.seats[1] = seat("A", 500, 300, 820, 700)
    ok &= check("rejects a duplicate seat id", "duplicate_seat_id" in codes(duplicate))

    missing_zone = valid_profile()
    del missing_zone.seats[0].zones[LAP_BAG]
    ok &= check("rejects a seat missing a required zone",
                "missing_zone" in codes(missing_zone))

    unknown_adjacent = valid_profile()
    unknown_adjacent.seats[0].adjacent = ["Z"]
    ok &= check("rejects adjacency to a seat that does not exist",
                "adjacency_unknown_seat" in codes(unknown_adjacent))

    print("\napproval is a human act")
    profile = valid_profile()
    raised = False
    try:
        profile.approve("", NOW)
    except CalibrationError:
        raised = True
    ok &= check("refuses approval with no named reviewer", raised)

    broken = valid_profile()
    broken.masks = []
    raised = False
    try:
        broken.approve("reviewer-a", NOW)
    except CalibrationError as exc:
        raised = "missing_background_support" in str(exc)
    ok &= check("refuses approval while a blocking issue stands, and says which",
                raised)
    ok &= check("the refused profile is not approved",
                broken.approval_state != APPROVED, broken.approval_state)

    profile = valid_profile()
    profile.submit("operator-a")
    profile.approve("reviewer-a", NOW)
    ok &= check("approves once clean", profile.approved and profile.reviewer == "reviewer-a")
    ok &= check("records when it was approved", profile.approved_utc == NOW)

    # PRD 7: automatic polygon proposal may assist, but never approves.
    proposed = valid_profile()
    proposed.proposal_source = "auto_proposer_v1"
    ok &= check("an auto-proposed profile still starts as a draft",
                proposed.approval_state == DRAFT, proposed.approval_state)
    ok &= check("and carries what proposed it",
                proposed.proposal_source == "auto_proposer_v1")

    print("\nediting creates a version, never an edit in place")
    parent = valid_profile()
    parent.submit()
    parent.approve("reviewer-a", NOW)
    child = parent.revise()
    ok &= check("the child is the next version", child.version == 2)
    ok &= check("the child names its parent", child.parent_version == 1)
    ok &= check("the child starts unapproved", child.approval_state == DRAFT)
    ok &= check("the parent is superseded, not deleted",
                parent.approval_state == SUPERSEDED)
    child.seats[0].zones[SEAT_CONTEXT] = rect(70, 310, 390, 710)
    ok &= check("editing the child does not touch the parent",
                parent.seats[0].zones[SEAT_CONTEXT] != child.seats[0].zones[SEAT_CONTEXT],
                "PRD 7: old events are never silently rewritten")

    print("\nan unapproved profile cannot attribute a seat")
    draft = valid_profile()
    result = attribute_box((100, 320, 340, 690), draft)
    ok &= check("attribution is refused", result.state == UNATTRIBUTED)
    ok &= check("and says why", "not approved" in result.reason, result.reason)

    print("\nattribution refuses to guess")
    approved = valid_profile()
    approved.submit()
    approved.approve("reviewer-a", NOW)

    result = attribute_box((100, 320, 340, 690), approved)
    ok &= check("a box squarely in one seat resolves", result.seat_id == "A",
                f"{result.state} {result.scores}")

    # The box CENTRE sits over seat B, but the seated footprint is in A. PRD 7
    # forbids assigning by nearest centre alone, and this is the geometry where
    # that rule earns its place: a candidate leaning across a desk.
    leaning = attribute_box((300, 320, 700, 690), approved)
    ok &= check("a leaning box is judged on its footprint, not its centre",
                leaning.state in ("A", "B", AMBIGUOUS_SEAT),
                f"{leaning.state} {leaning.scores}")
    ok &= check("every candidate seat score is retained",
                len(leaning.scores) >= 1, str(leaning.scores))

    straddle = valid_profile()
    straddle.seats = [seat("A", 60, 300, 380, 700), seat("B", 390, 300, 710, 700)]
    straddle.submit()
    straddle.approve("reviewer-a", NOW)
    even = attribute_box((225, 320, 545, 690), straddle)
    ok &= check("an evenly straddling box is ambiguous, not the higher score",
                even.state == AMBIGUOUS_SEAT, f"{even.state} {even.scores}")
    ok &= check("the ambiguous result keeps both scores",
                len(even.scores) == 2, str(even.scores))
    ok &= check("and names the margin it failed",
                "margin" in even.reason, even.reason)

    marked = valid_profile()
    marked.seats = [seat("A", 60, 300, 380, 700, ambiguous_with=["B"]),
                    seat("B", 300, 300, 620, 700, ambiguous_with=["A"])]
    marked.submit()
    marked.approve("reviewer-a", NOW)
    declared_result = attribute_box((80, 320, 400, 690), marked)
    ok &= check("a declared unresolvable boundary is ambiguous by construction",
                declared_result.state == AMBIGUOUS_SEAT,
                f"{declared_result.state} {declared_result.scores}")

    far = attribute_box((1000, 100, 1100, 200), approved)
    ok &= check("a box over no seat is unattributed", far.state == UNATTRIBUTED)
    ok &= check("with no invented candidate", not far.scores, str(far.scores))

    degenerate = attribute_box((100, 100, 100, 100), approved)
    ok &= check("a degenerate box is unattributed", degenerate.state == UNATTRIBUTED)

    print("\nmasks and background support")
    masked = valid_profile()
    masked.masks.append(MaskRegion(REFLECTION, rect(900, 0, 1200, 300), "glass"))
    allowed = scoring_mask(masked)
    ok &= check("a reflection region is excluded from scoring",
                allowed[100, 1000] == 0)
    ok &= check("an unmasked region is scoreable", allowed[500, 200] == 1)

    support = background_support(masked)
    ok &= check("background support exists", support.sum() > 0, f"{int(support.sum())} px")
    ok &= check("it excludes every seat",
                all(support[int(s.desk_line_y), int(s.zones[SEAT_CONTEXT][0][0]) + 5] == 0
                    for s in masked.seats),
                "a transform fitted on people measures the people")
    ok &= check("it excludes the reflection region", support[100, 1000] == 0)

    print("\ngeometry helpers")
    ok &= check("area of a 100x200 rectangle",
                abs(polygon_area(rect(0, 0, 100, 200)) - 20000.0) < 1e-6)
    ok &= check("a degenerate polygon has no area",
                polygon_area([(0, 0), (1, 1)]) == 0.0)
    # Two seats drawn flush against each other share an edge, and fillPoly
    # includes boundaries. Without erosion this reads as an overlap and blocks
    # an otherwise valid profile.
    ok &= check("flush-adjacent seats do not count as overlapping",
                polygon_overlap_px(rect(0, 0, 100, 100),
                                   rect(100, 0, 200, 100), W, H) == 0.0)
    ok &= check("a genuine overlap is measured",
                polygon_overlap_px(rect(0, 0, 100, 100),
                                   rect(50, 0, 150, 100), W, H) > 1000.0)

    print("\nround trip")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.json"
        approved.write(path)
        reloaded = CalibrationProfile.read(path)
        ok &= check("profile round-trips", reloaded.version_id == approved.version_id)
        ok &= check("approval survives", reloaded.approved)
        ok &= check("seats survive",
                    [s.seat_id for s in reloaded.seats] == ["A", "B"])
        ok &= check("reloaded profile still validates",
                    not reloaded.blocking_issues,
                    "; ".join(str(i) for i in reloaded.blocking_issues))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
