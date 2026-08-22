"""Phase 5 gate: temporal confirmation and its four statuses (PRD 10).

One line of PRD 10 is the whole reason this module is not a filter:

> A singleton is retained, not silently deleted.

The obvious implementation drops anything seen once, and it is wrong in exactly
the case the product exists for: a phone visible for a third of a second between
a pocket and a lap *is* a singleton. So a single sighting gets a status, keeps
its artifacts, and ranks below corroborated evidence. It is never removed, and
`deleted` is reported as a number so that claim is checkable rather than
asserted in a comment.

The second property tested here is that **conflict is a status, not a
tie-break**. Three frames calling a region `phone` and two calling it `mouse` is
not resolved by majority. Resolving it manufactures agreement the evidence does
not contain, and the disagreement is itself the signal — usually a small bright
rectangle that neither class fits. The full class distribution therefore travels
with every group, including the ones that agreed.

Also asserted: PRD 15's "Zero detections must yield valid metrics rather than
crash."
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.object_detection import (  # noqa: E402
    CHIT, INSUFFICIENT, MOUSE, PHONE, ObjectDetection,
)
from pipeline.temporal_confirmation import (  # noqa: E402
    CONFLICTING, SINGLE_FRAME, SUPPORTED, ConfirmConfig, confirm, summarise,
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def det(cls, pts, box=(200.0, 300.0, 232.0, 322.0), confidence=0.7,
        event="EV-1", seat="S-A", track=1, config="dfine_native",
        abstain_reason="") -> ObjectDetection:
    return ObjectDetection(
        crop_id=f"c{int(pts)}", config_id=config, pts_ms=pts, cls=cls,
        confidence=confidence, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
        seat_state=seat, track_id=track, event_id=event,
        abstained=cls == INSUFFICIENT, abstain_reason=abstain_reason)


def main() -> int:
    ok = True
    cfg = ConfirmConfig()

    print("a singleton is retained with its own status, never deleted")
    groups = confirm([det(PHONE, 1000.0)], cfg)
    ok &= check("one group survives", len(groups) == 1, str(len(groups)))
    ok &= check("its status is single_frame_object_candidate",
                groups[0].status == SINGLE_FRAME, groups[0].status)
    ok &= check("it is not marked supported", not groups[0].supported)
    ok &= check("its crop is still reachable",
                groups[0].crop_ids == ["c1000"], str(groups[0].crop_ids))
    ok &= check("the reason states why it is kept",
                "never deleted" in groups[0].reason, groups[0].reason[:60])
    ok &= check("the summary reports zero deletions",
                summarise(groups)["deleted"] == 0)
    ok &= check("and counts the retained singleton",
                summarise(groups)["retained_singletons"] == 1)

    print("\ntwo consistent sightings are temporally supported")
    supported = confirm([det(PHONE, 1000.0), det(PHONE, 1500.0)], cfg)
    ok &= check("one group", len(supported) == 1, str(len(supported)))
    ok &= check("status is temporally_supported_object_evidence",
                supported[0].status == SUPPORTED, supported[0].status)
    ok &= check("support frames are counted",
                supported[0].support_frames == 2)
    ok &= check("duration is measured across the group",
                supported[0].duration_ms == 500.0,
                f"{supported[0].duration_ms} ms")
    ok &= check("spatial consistency is recorded",
                supported[0].spatial_consistency > 0.9,
                f"{supported[0].spatial_consistency}")

    print("\na disagreeing group stays conflicting; no majority vote resolves it")
    mixed = confirm([det(PHONE, 1000.0), det(PHONE, 1500.0),
                     det(MOUSE, 2000.0), det(MOUSE, 2500.0)], cfg)
    # Class is part of the group key, so these are two groups, and their
    # disagreement is visible as two competing claims on the same region.
    ok &= check("competing classes are two groups, not one silently merged",
                len(mixed) == 2, str(len(mixed)))
    ok &= check("each keeps its own class distribution",
                all(g.class_distribution for g in mixed),
                str([g.class_distribution for g in mixed]))

    print("\nan explicitly split group reports the split rather than a winner")
    # Same class key would hide it, so this exercises the majority rule directly
    # by handing one group a mixed distribution through a shared key.
    split = [det(PHONE, 1000.0), det(PHONE, 1500.0), det(PHONE, 2000.0)]
    split[1].cls = MOUSE
    split[2].cls = CHIT
    from pipeline.temporal_confirmation import _confirmed
    outcome = _confirmed(split, cfg)
    ok &= check("status is conflicting_object_evidence",
                outcome.status == CONFLICTING, outcome.status)
    ok &= check("the full distribution is reported",
                len(outcome.class_distribution) == 3,
                str(outcome.class_distribution))
    ok &= check("the reason says the disagreement was not voted away",
                "rather than resolved by vote" in outcome.reason,
                outcome.reason[:70])

    print("\na long gap starts a new group, not one long sighting")
    apart = confirm([det(PHONE, 1000.0), det(PHONE, 90000.0)], cfg)
    ok &= check("10 s and 90 s are two events",
                len(apart) == 2, str(len(apart)))
    ok &= check("neither is reported as an 89-second sighting",
                all(g.duration_ms == 0.0 for g in apart),
                str([g.duration_ms for g in apart]))

    print("\na sighting that moves elsewhere on the desk is a separate group")
    moved = confirm([det(PHONE, 1000.0, (200.0, 300.0, 232.0, 322.0)),
                     det(PHONE, 1500.0, (600.0, 300.0, 632.0, 322.0))], cfg)
    ok &= check("spatially disjoint sightings do not merge",
                len(moved) == 2, str(len(moved)))

    print("\na held object that drifts stays one group")
    drifting = confirm([det(PHONE, 1000.0, (200.0, 300.0, 232.0, 322.0)),
                        det(PHONE, 1500.0, (206.0, 304.0, 238.0, 326.0)),
                        det(PHONE, 2000.0, (212.0, 308.0, 244.0, 330.0))], cfg)
    ok &= check("drift within the IoU floor is one object",
                len(drifting) == 1, str(len(drifting)))
    ok &= check("it is supported by three frames",
                drifting[0].support_frames == 3)

    print("\nconfigurations never share a group")
    two_configs = confirm([det(PHONE, 1000.0, config="dfine_native"),
                           det(PHONE, 1000.0, config="dfine_sahi")], cfg)
    ok &= check("two configurations produce two groups",
                len(two_configs) == 2,
                "one configuration's evidence must never be counted as "
                "another's corroboration (PRD 17.9)")

    print("\nseats never share a group either")
    two_seats = confirm([det(PHONE, 1000.0, seat="S-A"),
                         det(PHONE, 1500.0, seat="S-B")], cfg)
    ok &= check("two seats produce two groups", len(two_seats) == 2)

    print("\nabstentions are grouped and reported, not discarded")
    with_abstentions = confirm([
        det(PHONE, 1000.0), det(PHONE, 1500.0),
        det(INSUFFICIENT, 2000.0, abstain_reason="native footprint 22x16 px"),
        det(INSUFFICIENT, 2500.0, abstain_reason="native footprint 22x16 px"),
    ], cfg)
    abstained = [g for g in with_abstentions if g.status == INSUFFICIENT]
    ok &= check("abstentions form their own group", len(abstained) == 1,
                str(len(abstained)))
    ok &= check("both abstaining crops are counted",
                abstained[0].support_frames == 2)
    ok &= check("the abstention reason survives",
                "22x16" in abstained[0].reason, abstained[0].reason[:50])
    ok &= check("abstentions do not corrupt the real group",
                any(g.status == SUPPORTED for g in with_abstentions))

    print("\nzero detections yield valid metrics rather than a crash (PRD 15)")
    empty = confirm([], cfg)
    ok &= check("no groups, no exception", empty == [], str(empty))
    report = summarise(empty)
    ok &= check("the summary is a valid dict", isinstance(report, dict))
    ok &= check("it says zero detections is not evidence of nothing happening",
                "not evidence that nothing happened" in report["note"],
                report["note"][:60])

    print("\nthe summary exposes status counts and the retention claim")
    everything = confirm([det(PHONE, 1000.0), det(PHONE, 1500.0),
                          det(CHIT, 5000.0),
                          det(INSUFFICIENT, 9000.0)], cfg)
    report = summarise(everything)
    for field in ("by_status", "by_class", "retained_singletons", "deleted",
                  "supported_duration_ms", "median_support_frames"):
        ok &= check(f"summary reports {field}", field in report)
    ok &= check("phone and chit are separately countable (PRD 15)",
                PHONE in report["by_class"] and CHIT in report["by_class"],
                str(report["by_class"]))
    ok &= check("nothing was deleted across the whole set",
                report["deleted"] == 0)
    ok &= check("every input is accounted for in some group",
                sum(g.support_frames for g in everything) == 4,
                str(sum(g.support_frames for g in everything)))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
