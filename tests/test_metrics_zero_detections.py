"""Phase 7 gate: metrics for the run that found nothing (PRD 15).

> Zero detections must yield valid metrics rather than crash.

The zero-detection run is not an edge case to be defended against; on this
corpus and source scale it is a likely result, and it is the one most worth
reporting correctly. Two ways to get it wrong:

**Crashing.** A ZeroDivisionError in the metrics means the honest outcome is the
one outcome the pipeline cannot report.

**Reporting 0.000.** A precision of 0.000 is a claim: the detector was asked and
was wrong. With no detections it was never asked, and the defined answer is
`None` with the counts beside it. Every ratio here is `None` when undefined,
because a zero that reads as a measurement is worse than an absent number.

The other half of this gate is the **seven outcomes**. PRD 15 requires
`not_routed_to_candidate` and `quality_unavailable` to be distinguishable from
`FN`. Collapsing them produces a recall figure that blames the detector for the
motion gate and for the camera, and optimising against it leads straight to
loosening the gate until everything is a candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evaluate import (  # noqa: E402
    AMBIGUOUS, FN, FP, INSUFFICIENT, NOT_ROUTED, QUALITY_UNAVAILABLE,
    STAGE_CROP, STAGE_DETECTOR, STAGE_MOTION, TP, MatchConfig, RunView, compare,
    evaluate, metrics,
)
from pipeline.reference_manifest import ReferenceManifest, ReferenceRow  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def reference(reference_id="REF-1", pts=12000.0, cls="phone",
              box=(200.0, 500.0, 232.0, 522.0), visibility="visible"):
    return ReferenceRow(reference_id=reference_id, source_video_sha256="a" * 64,
                        source_pts_ms=pts, cls=cls,
                        annotation_source="reviewer-a",
                        bbox_xyxy_source=box, visibility=visibility)


def manifest_of(rows, errors=()):
    return ReferenceManifest(rows=list(rows), errors=list(errors))


def detection(pts=12000.0, cls="phone", box=(201.0, 501.0, 233.0, 523.0),
              status="temporally_supported_object_evidence", event="EV-1"):
    return {"pts_ms": pts, "cls": cls, "x1": box[0], "y1": box[1],
            "x2": box[2], "y2": box[3], "confidence": 0.6, "status": status,
            "event_id": event, "crop_id": "c1"}


def main() -> int:
    ok = True
    cfg = MatchConfig()

    print("zero references and zero detections still produce metrics")
    empty_view = RunView(config_id="dfine_native", duration_ms=282000.0)
    empty = manifest_of([])
    outcomes = evaluate(empty, empty_view, cfg)
    report = metrics(outcomes, empty, empty_view, cfg)
    ok &= check("no exception", isinstance(report, dict))
    ok &= check("precision is None, not 0.0",
                report["overall"]["precision"] is None,
                "0.000 says the detector was wrong; None says it was never asked")
    ok &= check("recall is None", report["overall"]["recall"] is None)
    ok &= check("F1 is None", report["overall"]["f1"] is None)
    ok &= check("counts are present and zero", report["overall"]["TP"] == 0)
    ok &= check("phone and chit are still reported separately",
                set(report["per_class"]) == {"phone", "secondary_paper_chit"},
                str(list(report["per_class"])))
    ok &= check("the note explains the None values",
                "never asked" in report["note"], report["note"][:60])

    print("\nreferences with zero detections give recall 0, not None")
    view = RunView(config_id="dfine_native", duration_ms=282000.0,
                   candidate_spans=[(11000.0, 13000.0)],
                   crops=[{"pts_ms": 12000.0, "crop_id": "c1",
                           "native_w": 120.0, "native_h": 120.0}])
    one = manifest_of([reference()])
    outcomes = evaluate(one, view, cfg)
    report = metrics(outcomes, one, view, cfg)
    ok &= check("one FN", report["overall"]["FN"] == 1)
    ok &= check("recall is 0.0, a real measurement",
                report["overall"]["recall"] == 0.0,
                "here the detector WAS asked and returned nothing")
    ok &= check("precision is None; nothing was proposed",
                report["overall"]["precision"] is None)
    ok &= check("the loss is attributed to the detector",
                report["stage_loss_waterfall"][STAGE_DETECTOR] == 1,
                str(report["stage_loss_waterfall"]))

    print("\nthe seven outcomes stay distinguishable (PRD 15)")
    # not routed: no candidate covers the timestamp
    unrouted = RunView(config_id="c", duration_ms=282000.0, candidate_spans=[])
    out = evaluate(manifest_of([reference()]), unrouted, cfg)
    ok &= check("a reference motion never routed is not_routed_to_candidate",
                out[0].outcome == NOT_ROUTED, out[0].outcome)
    ok &= check("and the loss is attributed to motion, not the detector",
                out[0].loss_stage == STAGE_MOTION, str(out[0].loss_stage))

    # quality unavailable: the data was not there
    unavailable = RunView(config_id="c", duration_ms=282000.0,
                          candidate_spans=[(11000.0, 13000.0)],
                          failed_spans=[(11500.0, 12500.0, "decode_gap")])
    out = evaluate(manifest_of([reference()]), unavailable, cfg)
    ok &= check("a reference inside a decode gap is quality_unavailable",
                out[0].outcome == QUALITY_UNAVAILABLE, out[0].outcome)
    ok &= check("no stage is blamed for absent data",
                out[0].loss_stage is None, str(out[0].loss_stage))

    # insufficient: the crop could not carry an object call
    abstaining = RunView(config_id="c", duration_ms=282000.0,
                         candidate_spans=[(11000.0, 13000.0)],
                         abstentions=[{"pts_ms": 12000.0,
                                       "abstain_reason": "native footprint "
                                                         "22x16 px"}])
    out = evaluate(manifest_of([reference()]), abstaining, cfg)
    ok &= check("an abstained crop is insufficient_visual_evidence",
                out[0].outcome == INSUFFICIENT, out[0].outcome)
    ok &= check("the loss is attributed to the crop",
                out[0].loss_stage == STAGE_CROP)
    ok &= check("the abstention reason survives into the outcome",
                "22x16" in out[0].reason, out[0].reason[:60])

    # ambiguous: found, wrong class
    wrong_class = RunView(config_id="c", duration_ms=282000.0,
                          candidate_spans=[(11000.0, 13000.0)],
                          detections=[detection(cls="mouse")],
                          crops=[{"pts_ms": 12000.0, "crop_id": "c1",
                                  "native_w": 120.0, "native_h": 120.0}])
    out = [o for o in evaluate(manifest_of([reference()]), wrong_class, cfg)
           if o.reference_id]
    ok &= check("a phone referenced and a mouse detected is ambiguous",
                out[0].outcome == AMBIGUOUS, out[0].outcome)
    ok &= check("not a true positive", out[0].outcome != TP)
    ok &= check("and not the same failure as detecting nothing",
                out[0].outcome != FN)

    # true positive
    found = RunView(config_id="c", duration_ms=282000.0,
                    candidate_spans=[(11000.0, 13000.0)],
                    detections=[detection()],
                    ranks={"EV-1": 3},
                    crops=[{"pts_ms": 12000.0, "crop_id": "c1",
                            "native_w": 120.0, "native_h": 120.0}])
    out = [o for o in evaluate(manifest_of([reference()]), found, cfg)
           if o.reference_id]
    ok &= check("a matching detection is a TP", out[0].outcome == TP)
    ok &= check("the IoU is recorded", out[0].matched_iou > 0.5,
                str(out[0].matched_iou))
    ok &= check("the timestamp delta is recorded",
                out[0].matched_pts_delta_ms == 0.0)
    ok &= check("the rank of the reference is recorded (PRD 15)",
                out[0].rank == 3, str(out[0].rank))

    print("\nall seven outcomes are distinct values")
    ok &= check("seven distinct", len({TP, FP, FN, AMBIGUOUS, INSUFFICIENT,
                                       QUALITY_UNAVAILABLE, NOT_ROUTED}) == 7)

    print("\nthe recalls answer different questions over different denominators")
    # R1 found, R2 routed but missed, R3 never routed, R4 inside a decode gap.
    mixed = manifest_of([reference("R1", 12000.0), reference("R2", 50000.0),
                         reference("R3", 70000.0), reference("R4", 90000.0)])
    view = RunView(config_id="c", duration_ms=282000.0,
                   candidate_spans=[(11000.0, 13000.0), (49000.0, 51000.0)],
                   failed_spans=[(89000.0, 91000.0, "decode_gap")],
                   detections=[detection(12000.0)],
                   crops=[{"pts_ms": 12000.0, "crop_id": "c1",
                           "native_w": 120.0, "native_h": 120.0},
                          {"pts_ms": 50000.0, "crop_id": "c2",
                           "native_w": 120.0, "native_h": 120.0}])
    report = metrics(evaluate(mixed, view, cfg), mixed, view, cfg)
    overall = report["overall"]
    ok &= check("one of each outcome",
                overall["TP"] == 1 and overall["FN"] == 1
                and overall["not_routed_to_candidate"] == 1
                and overall["quality_unavailable"] == 1,
                f"TP {overall['TP']} FN {overall['FN']} "
                f"NR {overall['not_routed_to_candidate']} "
                f"QU {overall['quality_unavailable']}")
    ok &= check("strict recall is 1 of all 4 annotated",
                overall["recall"] == 0.25, str(overall["recall"]))
    ok &= check("routing recall is 2 of the 3 the source could support",
                overall["candidate_routing_recall"] == round(2 / 3, 4),
                f"{overall['candidate_routing_recall']}; the decode-gap "
                f"reference is neither routed nor not-routed")
    ok &= check("recall given routing is 1 of the 2 routed",
                overall["recall_conditional_on_routing"] == 0.5,
                str(overall["recall_conditional_on_routing"]))
    ok &= check("full-pipeline recall is 1 of the 3 supportable",
                overall["full_pipeline_recall"] == round(1 / 3, 4),
                str(overall["full_pipeline_recall"]))
    ok &= check("all four differ, so none can stand in for another",
                len({overall["recall"], overall["candidate_routing_recall"],
                     overall["recall_conditional_on_routing"],
                     overall["full_pipeline_recall"]}) == 4)
    ok &= check("the waterfall blames motion for the unrouted reference",
                report["stage_loss_waterfall"][STAGE_MOTION] == 1,
                str(report["stage_loss_waterfall"]))

    print("\nfalse positives are counted only against annotated classes")
    unlabelled = RunView(config_id="c", duration_ms=282000.0,
                         candidate_spans=[(11000.0, 13000.0)],
                         detections=[detection(cls="keyboard", pts=40000.0)])
    out = evaluate(manifest_of([reference()]), unlabelled, cfg)
    ok &= check("a keyboard where only phones are annotated is not an FP",
                sum(1 for o in out if o.outcome == FP) == 0,
                "counting it would invent a precision the references cannot "
                "support")
    phone_fp = RunView(config_id="c", duration_ms=282000.0,
                       candidate_spans=[(11000.0, 13000.0)],
                       detections=[detection(cls="phone", pts=40000.0)])
    out = evaluate(manifest_of([reference()]), phone_fp, cfg)
    ok &= check("an unreferenced phone is an FP",
                sum(1 for o in out if o.outcome == FP) == 1)

    print("\nan incomplete manifest taints every metric derived from it")
    from pipeline.reference_manifest import ValidationError
    tainted = manifest_of([reference()],
                          [ValidationError("REF-9", "source_pts_ms", "negative")])
    report = metrics(evaluate(tainted, found, cfg), tainted, found, cfg)
    ok &= check("complete is False", report["complete"] is False)
    ok &= check("the note says INCOMPLETE",
                "INCOMPLETE" in report["completeness_note"],
                report["completeness_note"][:50])

    print("\ncompare refuses a selection the numbers do not support")
    ok &= check("no configurations at all",
                "no configuration" in compare({})["recommendation"].lower())
    good = metrics(evaluate(manifest_of([reference()]), found, cfg),
                   manifest_of([reference()]), found, cfg)
    ok &= check("taxonomy unapproved blocks selection",
                "PRD 3" in compare({"dfine_native": good})["recommendation"],
                compare({"dfine_native": good})["recommendation"][:50])
    ok &= check("an incomplete manifest blocks selection",
                "incomplete" in compare(
                    {"dfine_native": {**good, "complete": False}},
                    taxonomy_approved=True)["recommendation"].lower())
    ok &= check("no forced winner is stated in the note",
                "no forced winner" in compare(
                    {"dfine_native": good}, taxonomy_approved=True)["note"])

    print("\na margin inside the noise yields no winner")
    a = {**good, "overall": {**good["overall"], "f1": 0.62}}
    b = {**good, "overall": {**good["overall"], "f1": 0.60}}
    close = compare({"a": a, "b": b}, taxonomy_approved=True)
    ok &= check("declared no clear winner",
                "No clear winner" in close["recommendation"],
                close["recommendation"][:60])
    far = compare({"a": {**good, "overall": {**good["overall"], "f1": 0.85}},
                   "b": b}, taxonomy_approved=True)
    ok &= check("a wide margin does name the leader",
                "measures best" in far["recommendation"],
                far["recommendation"][:60])
    ok &= check("and still bounds the claim to this corpus",
                "not a general claim" in far["recommendation"])

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
