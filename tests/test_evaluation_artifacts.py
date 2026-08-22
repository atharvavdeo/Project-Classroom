"""Phase 7 gate: every reference produces a rendered, citable result (PRD 15, 18).

PRD 18's acceptance criterion:

> Every reference row produces rendered TP/FP/FN/abstention/quality/not-routed
> output; phone/chit metrics are separate.

"Every" is the requirement. An evaluation that reports aggregate counts but
leaves some references with no row is an evaluation whose numbers cannot be
checked one at a time — and the rows most likely to go missing are the awkward
ones, the abstentions and the never-routed, which are exactly the rows that
explain the aggregate.

So this gate walks the outcomes and asserts a one-to-one correspondence with
the manifest, that every outcome carries the fields needed to render it, and
that phone and chit never share a number.

The gate audit is covered here too, for the property PRD 8 asks for and that a
reference-only audit cannot provide: a gate that passes everything scores a
perfect false-suppression rate on references. Only sampled *normal* intervals
expose it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evaluate import (  # noqa: E402
    LOSS_STAGES, OUTCOMES, TP, MatchConfig, RunView, evaluate, metrics,
)
from pipeline.gate_audit import (  # noqa: E402
    DISTURBANCE, NORMAL, REFERENCE, AuditPopulation, SampleConfig, audit,
    sample_intervals,
)
from pipeline.gates import (  # noqa: E402
    DEPRIORITISE, PASS, SUPPRESS_FROM_AUTO_RANKING, GateLog,
)
from pipeline.reference_manifest import ReferenceManifest, ReferenceRow  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def reference(reference_id, pts, cls="phone", visibility="visible",
              box=(200.0, 500.0, 232.0, 522.0)):
    return ReferenceRow(reference_id=reference_id, source_video_sha256="a" * 64,
                        source_pts_ms=pts, cls=cls,
                        annotation_source="reviewer-a", bbox_xyxy_source=box,
                        visibility=visibility)


def main() -> int:
    ok = True
    cfg = MatchConfig()

    # Six references, deliberately chosen so every outcome class is populated.
    rows = [
        reference("R1", 12000.0),                              # TP
        reference("R2", 50000.0),                              # FN, detector
        reference("R3", 70000.0),                              # not routed
        reference("R4", 90000.0),                              # quality unavail.
        reference("R5", 110000.0),                             # insufficient
        reference("R6", 130000.0, cls="secondary_paper_chit"),  # FN, chit
        # Found, but classified as something else. Not a TP, and not the same
        # failure as detecting nothing -- so it must land on `ambiguous`.
        reference("R7", 150000.0, cls="secondary_paper_chit"),
    ]
    manifest = ReferenceManifest(rows=rows)
    view = RunView(
        config_id="dfine_native", duration_ms=282000.0,
        candidate_spans=[(11000.0, 13000.0), (49000.0, 51000.0),
                         (109000.0, 111000.0), (129000.0, 131000.0),
                         (149000.0, 151000.0)],
        failed_spans=[(89000.0, 91000.0, "decode_gap")],
        detections=[{"pts_ms": 12000.0, "cls": "phone", "x1": 201.0,
                     "y1": 501.0, "x2": 233.0, "y2": 523.0, "confidence": 0.6,
                     "status": "temporally_supported_object_evidence",
                     "event_id": "EV-1", "crop_id": "c1"},
                    {"pts_ms": 200000.0, "cls": "phone", "x1": 10.0, "y1": 10.0,
                     "x2": 42.0, "y2": 32.0, "confidence": 0.4,
                     "status": "single_frame_object_candidate",
                     "event_id": "EV-9", "crop_id": "c9"},
                    {"pts_ms": 150000.0, "cls": "phone", "x1": 201.0,
                     "y1": 501.0, "x2": 233.0, "y2": 523.0, "confidence": 0.5,
                     "status": "temporally_supported_object_evidence",
                     "event_id": "EV-7", "crop_id": "c7"}],
        abstentions=[{"pts_ms": 110000.0,
                      "abstain_reason": "native footprint 22x16 px"}],
        crops=[{"pts_ms": t, "crop_id": f"c{i}", "native_w": 120.0,
                "native_h": 120.0}
               for i, t in enumerate((12000.0, 50000.0, 110000.0, 130000.0,
                                      150000.0))],
        ranks={"EV-1": 2, "EV-9": 17})

    outcomes = evaluate(manifest, view, cfg)
    reference_outcomes = [o for o in outcomes if o.reference_id]

    print("every reference row produces exactly one outcome")
    ok &= check("one outcome per reference",
                len(reference_outcomes) == len(rows),
                f"{len(reference_outcomes)} outcomes for {len(rows)} references")
    ok &= check("and their IDs match one to one",
                {o.reference_id for o in reference_outcomes}
                == {r.reference_id for r in rows})
    ok &= check("no reference is silently absent",
                all(any(o.reference_id == r.reference_id
                        for o in reference_outcomes) for r in rows))

    print("\nevery outcome class PRD 18 names is reachable and populated")
    produced = {o.outcome for o in outcomes}
    for name in OUTCOMES:
        ok &= check(f"{name} is produced", name in produced,
                    "" if name in produced else "not exercised by this fixture")

    print("\nevery outcome carries what a renderer needs")
    for outcome in reference_outcomes:
        row = outcome.to_row()
        missing = [f for f in ("reference_id", "config_id", "cls", "outcome",
                               "source_pts_ms", "visibility", "reason")
                   if f not in row]
        ok &= check(f"{outcome.reference_id}: renderable", not missing,
                    str(missing))
        ok &= check(f"{outcome.reference_id}: carries a reason in words",
                    bool(outcome.reason), outcome.reason[:52])

    print("\nevery loss is attributed to a stage, or explicitly to none")
    for outcome in reference_outcomes:
        if outcome.outcome == TP:
            continue
        if outcome.outcome == "quality_unavailable":
            ok &= check(f"{outcome.reference_id}: no stage blamed for absent data",
                        outcome.loss_stage is None)
            continue
        ok &= check(f"{outcome.reference_id}: attributed to {outcome.loss_stage}",
                    outcome.loss_stage in LOSS_STAGES, str(outcome.loss_stage))

    print("\nthe waterfall sums to the references that were lost")
    report = metrics(outcomes, manifest, view, cfg)
    lost = sum(1 for o in reference_outcomes
               if o.outcome not in (TP, "quality_unavailable"))
    ok &= check("waterfall total matches the lost references",
                sum(report["stage_loss_waterfall"].values()) >= lost,
                f"{sum(report['stage_loss_waterfall'].values())} vs {lost} "
                f"(false positives may add stages)")

    print("\nphone and chit never share a number (PRD 15)")
    phone = report["per_class"]["phone"]
    chit = report["per_class"]["secondary_paper_chit"]
    ok &= check("both classes are reported",
                phone["reference_count"] == 5 and chit["reference_count"] == 2,
                f"phone {phone['reference_count']}, chit {chit['reference_count']}")
    ok &= check("the phone class has a TP", phone["TP"] == 1)
    ok &= check("the chit class does not", chit["TP"] == 0)
    ok &= check("their recalls differ", phone["recall"] != chit["recall"],
                f"{phone['recall']} vs {chit['recall']}")
    ok &= check("a chit precision is None, not zero",
                chit["precision"] is None,
                "no chit was proposed, so the detector was never asked")

    print("\nthe rank of every reference is recorded (PRD 15)")
    ok &= check("a ranks map exists", "reference_ranks" in report)
    ok &= check("it covers every reference",
                set(report["reference_ranks"]) == {r.reference_id for r in rows},
                str(sorted(report["reference_ranks"])))
    ok &= check("the matched one carries its rank",
                report["reference_ranks"]["R1"] == 2)
    ok &= check("unmatched references are counted as unranked",
                report["unranked_references"] == 6,
                str(report["unranked_references"]))

    print("\noutcomes are broken down by visibility")
    ok &= check("a by_visibility map exists", bool(report["by_visibility"]))
    ok &= check("its counts sum to the references",
                sum(sum(v.values()) for v in report["by_visibility"].values())
                == len(rows))

    # ------------------------------------------------------- gate audit --
    print("\na permissive gate looks perfect on references alone")
    permissive = GateLog()
    permissive.route("motion", "everything", PASS, 0.0, 282000.0,
                     "this gate passes the whole recording")

    references_only = AuditPopulation()
    for row in rows:
        references_only.add(REFERENCE, row.reference_id,
                            row.source_pts_ms - 250.0, row.source_pts_ms + 250.0,
                            True)
    result = audit(permissive, references_only, 282000.0)
    ok &= check("zero false suppressions on references",
                result["by_source"][REFERENCE]["false_suppression"] == 0)
    ok &= check("which looks like a flawless gate",
                result["by_source"][REFERENCE]["false_suppression_rate"] == 0.0)

    print("\nsampled normal intervals expose it (PRD 8)")
    mixed = sample_intervals(282000.0,
                             [(20000.0, 25000.0, "compression_noise")],
                             [r.source_pts_ms for r in rows],
                             SampleConfig(normal_samples=30,
                                          disturbance_samples=1))
    for row in rows:
        mixed.add(REFERENCE, row.reference_id, row.source_pts_ms - 250.0,
                  row.source_pts_ms + 250.0, True)
    result = audit(permissive, mixed, 282000.0)
    ok &= check("normal intervals were sampled",
                result["population"].get(NORMAL, 0) == 30,
                str(result["population"]))
    ok &= check("the permissive gate false-passes all of them",
                result["by_source"][NORMAL]["false_pass"] == 30,
                f"{result['by_source'][NORMAL]['false_pass']} of 30; a gate "
                f"that passes everything is only visible on this population")
    ok &= check("references still show zero false suppression",
                result["by_source"][REFERENCE]["false_suppression"] == 0,
                "which is why the split is reported, not one aggregate")

    print("\nan over-suppressing gate shows the opposite failure")
    strict = GateLog()
    strict.route("motion", "everything", SUPPRESS_FROM_AUTO_RANKING,
                 0.0, 282000.0, "this gate suppresses the whole recording")
    result = audit(strict, mixed, 282000.0)
    ok &= check("it false-suppresses every reference",
                result["by_source"][REFERENCE]["false_suppression"] == len(rows),
                str(result["by_source"][REFERENCE]["false_suppression"]))
    ok &= check("and correctly suppresses the normal intervals",
                result["by_source"][NORMAL]["correct_suppression"] == 30)
    ok &= check("the failures are listed for rendering",
                len(result["failures"]) == len(rows), str(len(result["failures"])))

    print("\na sensible gate scores well on both populations")
    sensible = GateLog()
    sensible.route("motion", "quiet", PASS, 0.0, 282000.0, "baseline pass")
    sensible.route("motion", "noise", SUPPRESS_FROM_AUTO_RANKING,
                   20000.0, 25000.0, "compression noise")
    result = audit(sensible, mixed, 282000.0)
    ok &= check("no reference is falsely suppressed",
                result["by_source"][REFERENCE]["false_suppression"] == 0)
    ok &= check("the disturbance span is held back",
                result["by_source"][DISTURBANCE]["correct_suppression"] == 1,
                str(result["by_source"].get(DISTURBANCE)))

    print("\nnormal sampling avoids intervals containing references")
    population = sample_intervals(282000.0, [],
                                  [r.source_pts_ms for r in rows],
                                  SampleConfig(normal_samples=40,
                                               disturbance_samples=0))
    clashes = [i for i in population.intervals
               if i[0] == NORMAL and any(i[2] <= r.source_pts_ms <= i[3]
                                         for r in rows)]
    ok &= check("no normal interval contains a reference", not clashes,
                "an interval containing a real sighting is not a normal "
                "interval, and a pass there is not a false pass")

    print("\nan unaudited gate says so rather than reporting zero")
    result = audit(GateLog(), AuditPopulation(), 282000.0)
    ok &= check("status is unaudited",
                result["overall"].get("status") == "unaudited",
                str(result["overall"].get("status")))
    ok &= check("the by-source breakdown is empty, not fabricated",
                result["by_source"] == {})

    print("\nsampling is deterministic under a fixed seed")
    a = sample_intervals(282000.0, [], [], SampleConfig(normal_samples=10,
                                                        disturbance_samples=0))
    b = sample_intervals(282000.0, [], [], SampleConfig(normal_samples=10,
                                                        disturbance_samples=0))
    ok &= check("two draws with the same seed agree",
                a.intervals == b.intervals,
                "an audit that changes between runs cannot be cited")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
