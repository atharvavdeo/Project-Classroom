"""Phase 3 gate: the five-state gate protocol and its audit (PRD 8, 15).

The property under test is the one the protocol exists for: **gates route
evidence, they do not erase it.** Nothing here may reduce the record count, and
the distinction between "suppressed but searchable" and "absent" must survive
into the audit.

The audit's headline is false suppression — relevant evidence held back. That is
the expensive error, and a gate that reports zero of them because nobody ever
checked must say `unaudited` rather than zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.gates import (  # noqa: E402
    DEPRIORITISE, FAILED, PASS, RANKABLE, RETAINED,
    SUPPRESS_FROM_AUTO_RANKING, UNCERTAIN, GateAudit, GateError, GateLog,
    GateRecord,
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    ok = True

    print("every state is representable and distinct")
    ok &= check("five states", len({PASS, DEPRIORITISE, UNCERTAIN,
                                    SUPPRESS_FROM_AUTO_RANKING, FAILED}) == 5)
    ok &= check("only pass and deprioritise reach ranking",
                RANKABLE == {PASS, DEPRIORITISE}, str(RANKABLE))
    # The distinction that is easy to collapse and expensive to lose.
    ok &= check("everything except failed is retained and searchable",
                RETAINED == {PASS, DEPRIORITISE, UNCERTAIN,
                             SUPPRESS_FROM_AUTO_RANKING},
                "suppressed evidence exists; failed evidence does not")

    print("\na decision without a reason is not a decision")
    raised = False
    try:
        GateRecord("motion", "compression_noise", PASS, 0.0, 500.0, "")
    except GateError:
        raised = True
    ok &= check("a record with no reason is refused", raised,
                "a state with no explanation cannot be reviewed or appealed")

    raised = False
    try:
        GateRecord("motion", "x", "probably_fine", 0.0, 500.0, "why")
    except GateError:
        raised = True
    ok &= check("an invented state is refused", raised)

    raised = False
    try:
        GateRecord("motion", "x", PASS, 500.0, 0.0, "why")
    except GateError:
        raised = True
    ok &= check("a reversed interval is refused", raised)

    print("\nrouting retains, never discards")
    log = GateLog()
    log.route("motion", "compression_noise", DEPRIORITISE, 0.0, 500.0,
              "area 0.0011 below the 0.05 floor",
              inputs={"area_ratio": 0.0011}, thresholds={"min_area_ratio": 0.05})
    log.route("health", "blackout", FAILED, 1000.0, 3000.0,
              "near-black for 2000 ms; no analysis is claimed")
    log.route("alignment", "camera_motion_uncertain", SUPPRESS_FROM_AUTO_RANKING,
              4000.0, 6000.0, "inlier ratio 0.31 below 0.60")
    log.route("motion", "environmental_motion", UNCERTAIN, 7000.0, 8000.0,
              "periodic at 0.9 Hz; conflicting classification")
    log.route("motion", "seat_activity", PASS, 9000.0, 10_000.0,
              "area 0.26, sustained over 4 windows")

    ok &= check("every decision is a record", len(log.records) == 5)
    ok &= check("nothing was discarded",
                log.summary()["discarded_records"] == 0)
    ok &= check("four of five remain searchable",
                log.summary()["retained_records"] == 4,
                "only the failed span is absent")

    print("\nthe most restrictive state applies to an interval")
    ok &= check("a failed span is failed", log.worst(1500.0, 2000.0) == FAILED)
    ok &= check("a suppressed span is suppressed",
                log.worst(4500.0, 5000.0) == SUPPRESS_FROM_AUTO_RANKING)
    ok &= check("a clean span passes", log.worst(9200.0, 9500.0) == PASS)
    ok &= check("an unmentioned span passes",
                log.worst(50_000.0, 51_000.0) == PASS,
                "silence is not suppression")

    print("\nranking penalties")
    ok &= check("a passing span is unpenalised", log.multiplier(9200.0, 9500.0) == 1.0)
    ok &= check("a deprioritised span is penalised but present",
                0.0 < log.multiplier(100.0, 400.0) < 1.0,
                f"{log.multiplier(100.0, 400.0)}")
    ok &= check("a suppressed span reaches ranking at zero",
                log.multiplier(4500.0, 5000.0) == 0.0)
    # Two independent weak signals over the same span are weaker than either.
    stacked = GateLog()
    stacked.route("a", "x", DEPRIORITISE, 0.0, 1000.0, "weak")
    stacked.route("b", "y", DEPRIORITISE, 0.0, 1000.0, "also weak")
    ok &= check("two penalties compound rather than taking the minimum",
                stacked.multiplier(0.0, 1000.0) < 0.35,
                f"{stacked.multiplier(0.0, 1000.0)}")

    print("\ncoverage is a union, not a sum")
    overlapping = GateLog()
    overlapping.route("a", "blackout", FAILED, 0.0, 2000.0, "dark")
    overlapping.route("b", "decode_gap", FAILED, 1000.0, 3000.0, "missing")
    ok &= check("overlapping failures count once",
                overlapping.coverage_ms(FAILED) == 3000.0,
                f"{overlapping.coverage_ms(FAILED)}, not 4000")

    print("\nan unaudited gate says so")
    audit = GateAudit(log)
    metrics = audit.metrics(total_ms=60_000.0)
    ok &= check("status is unaudited", metrics["status"] == "unaudited")
    ok &= check("and it explains what that means",
                "not a zero-error result" in metrics["detail"])
    ok &= check("failed coverage is still reported",
                metrics["failed_coverage_ms"] == 2000.0,
                f"{metrics['failed_coverage_ms']}")
    ok &= check("uncertain retention is still reported",
                metrics["uncertain_retained"] == 1)

    print("\nwith references, the four quantities are measured (PRD 8)")
    audit = GateAudit(log)
    # Relevant evidence that reached ranking: correct pass.
    audit.judge("ref_1", 9200.0, 9500.0, relevant=True)
    # Relevant evidence held back: FALSE SUPPRESSION, the expensive error.
    audit.judge("ref_2", 4500.0, 5000.0, relevant=True)
    audit.judge("ref_3", 1500.0, 2000.0, relevant=True)
    # Irrelevant evidence held back: correct suppression.
    audit.judge("ref_4", 7200.0, 7500.0, relevant=False)
    # Irrelevant evidence that reached ranking: false pass.
    audit.judge("ref_5", 100.0, 400.0, relevant=False)

    metrics = audit.metrics(total_ms=60_000.0)
    ok &= check("status is audited", metrics["status"] == "audited")
    ok &= check("correct pass counted", metrics["correct_pass"] == 1,
                f"{metrics['correct_pass']}")
    ok &= check("false suppression counted", metrics["false_suppression"] == 2,
                f"{metrics['false_suppression']}")
    ok &= check("correct suppression counted",
                metrics["correct_suppression"] == 1, f"{metrics['correct_suppression']}")
    ok &= check("false pass counted", metrics["false_pass"] == 1,
                f"{metrics['false_pass']}")
    ok &= check("the false-suppression rate is the headline",
                abs(metrics["false_suppression_rate"] - 2 / 3) < 1e-4,
                f"{metrics['false_suppression_rate']}")
    ok &= check("failures are enumerable for rendering",
                len(audit.failures()) == 3, f"{len(audit.failures())}")
    ok &= check("each failure names the state that caused it",
                all(f.state for f in audit.failures()),
                str([(f.reference_id, f.state) for f in audit.failures()]))

    print("\na perfect gate reports zero, and has earned it")
    clean = GateLog()
    clean.route("motion", "seat_activity", PASS, 0.0, 1000.0, "clear activity")
    clean.route("motion", "compression_noise", SUPPRESS_FROM_AUTO_RANKING,
                2000.0, 3000.0, "isolated single-block noise")
    clean_audit = GateAudit(clean)
    clean_audit.judge("r1", 200.0, 800.0, relevant=True)
    clean_audit.judge("r2", 2200.0, 2800.0, relevant=False)
    metrics = clean_audit.metrics()
    ok &= check("no false suppression", metrics["false_suppression"] == 0)
    ok &= check("no false pass", metrics["false_pass"] == 0)
    ok &= check("status distinguishes it from unaudited",
                metrics["status"] == "audited",
                "zero errors measured is not the same as never measured")

    print("\nrecords serialise with everything needed to re-derive them")
    row = log.records[0].to_row()
    for field in ("gate", "condition", "state", "start_pts_ms", "end_pts_ms",
                  "reason", "inputs", "thresholds", "retained", "rankable",
                  "penalty", "duration_ms"):
        ok &= check(f"row carries {field}", field in row)
    ok &= check("inputs and thresholds are both present",
                row["inputs"] and row["thresholds"],
                "PRD 8: condition, PTS range, inputs/scores, thresholds/config, "
                "state, reason, artifacts")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
