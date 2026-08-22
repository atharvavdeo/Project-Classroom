"""Stage 12: post-hoc evaluation against the reference manifest (PRD 15).

    python tools/evaluate_run.py --run artifacts/runs/<run_id> \
        --reference refs/cam12_phones.jsonl

The first and only place in the pipeline permitted to read reference data.
Before doing so it runs `assert_isolated` over every pre-evaluation stage of the
run and **refuses to evaluate** if any of them carries reference annotations --
because an evaluation of a pipeline that already saw the answers measures
nothing, and the artifacts would not say so.

With no `--reference`, the isolation check still runs and the evaluation is
written as explicitly incomplete. PRD 3: "Run inference but mark object
evaluation incomplete; make no quality claim."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, evaluate as ev, gate_audit as ga  # noqa: E402
from pipeline import reference_manifest as rm  # noqa: E402
from pipeline.gates import GateLog, GateRecord  # noqa: E402
from pipeline.object_detection import TARGET_CLASSES  # noqa: E402
from pipeline.run_manifest import BLOCKED, COMPLETE, RunManifest, RunStatus  # noqa: E402
from tools.run_pose_ab import read_jsonl  # noqa: E402


def build_view(paths, config_id: str, duration_ms: float) -> ev.RunView:
    """Assemble one configuration's output from the run folder.

    Reads artifacts only. The evaluator itself never touches the run directory,
    so reference data can never flow backwards into it.
    """
    candidates = read_jsonl(paths.dir / "05_motion_roi/candidate_manifest.jsonl")
    detections = read_jsonl(paths.dir / f"09_object/{config_id}/merged_detections.jsonl")
    crops = read_jsonl(paths.dir / "09_object/object_crop_manifest.jsonl")
    groups = read_jsonl(paths.dir / "10_evidence/temporal_confirmation.jsonl")
    ranked = read_jsonl(paths.dir / "10_evidence/ranked_events.jsonl")

    status_by_crop = {}
    for group in groups:
        if group["config_id"] != config_id:
            continue
        for crop_id in group.get("crop_ids", []):
            status_by_crop[crop_id] = group["status"]

    rows, abstentions = [], []
    for detection in detections:
        row = {**detection,
               "status": status_by_crop.get(detection["crop_id"], "unknown")}
        (abstentions if detection.get("abstained") else rows).append(row)

    failed = [(o["start_pts_ms"], o["end_pts_ms"], o["condition"])
              for o in read_jsonl(paths.dir / "02_health/observations.jsonl")
              if o.get("condition") in ("decode_gap", "blackout", "whiteout",
                                        "frozen_video")]
    suppressed = [(r["start_pts_ms"], r["end_pts_ms"], r["condition"])
                  for r in read_jsonl(paths.dir / "05_motion_roi/gate_log.jsonl")
                  if r.get("state") in ("suppress_from_auto_ranking", "uncertain")]

    return ev.RunView(
        config_id=config_id,
        candidate_spans=[(c["start_pts_ms"], c["end_pts_ms"]) for c in candidates],
        failed_spans=failed,
        suppressed_spans=suppressed,
        detections=rows,
        abstentions=abstentions,
        crops=[{"pts_ms": c["pts_ms"], "crop_id": c["crop_id"],
                "native_w": c["native_w"], "native_h": c["native_h"]}
               for c in crops],
        ranks={r["event_id"]: r["rank"] for r in ranked},
        duration_ms=duration_ms)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--taxonomy-approved", action="store_true")
    ap.add_argument("--pts-tolerance-ms", type=float, default=500.0)
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    manifest = RunManifest.read(paths)
    status = RunStatus(paths, manifest.run_id)

    # ------------------------------------------------- 1. isolation first --
    violations = rm.assert_isolated(paths.dir)
    paths.write_json("12_evaluation/isolation_check.json", {
        "stages_checked": list(rm.ISOLATED_STAGES),
        "violations": violations,
        "isolated": not violations,
        "note": "PRD 15 makes the reference manifest readable only at stage 12. "
                "An evaluation of a pipeline that already saw the answers "
                "measures nothing.",
    })
    if violations:
        print(f"REFUSING TO EVALUATE: {len(violations)} reference leak(s) found "
              f"in pre-evaluation artifacts:", file=sys.stderr)
        for violation in violations[:20]:
            print(f"  {violation}", file=sys.stderr)
        status.finish("12_evaluation", BLOCKED,
                      reason=f"{len(violations)} reference leak(s) upstream")
        status.write()
        return 2
    print(f"isolation verified across {len(rm.ISOLATED_STAGES)} stages")

    # -------------------------------------------------- 2. the references --
    duration_ms = 0.0
    recorded = json.loads((paths.dir / "RUN_STATUS.json").read_text(encoding="utf-8"))
    for stage in recorded.get("stages", []):
        if stage.get("stage") == "05_motion_roi" and stage.get("total_ms"):
            duration_ms = stage["total_ms"]

    if not args.reference:
        paths.write_json("12_evaluation/metrics.json", {
            "complete": False,
            "reason": "no reference manifest was supplied. Per PRD 3 the run "
                      "proceeds and object evaluation is marked incomplete; no "
                      "quality claim of any kind may be made from it.",
            "isolation_verified": True,
            "duration_ms": duration_ms,
        })
        status.finish("12_evaluation", BLOCKED,
                      reason="no reference manifest; evaluation incomplete "
                             "and no quality claim may be made")
        status.write()
        print("\nno --reference supplied; evaluation marked incomplete (PRD 3)")
        return 0

    status.begin("12_evaluation")
    source_sha = manifest.sources[0].sha256 if manifest.sources else ""
    references = rm.load(args.reference, source_sha256=source_sha,
                         approved_classes=TARGET_CLASSES if args.taxonomy_approved
                         else ())
    paths.write_json("12_evaluation/reference_summary.json", references.summary())
    for error in references.errors:
        paths.append_jsonl("12_evaluation/validation_errors.json", error.to_row())

    # A copy of exactly what was evaluated against, per PRD 13.
    for row in references.rows:
        paths.append_jsonl("12_evaluation/reference_manifest_copy.jsonl",
                           row.to_row())

    print(f"\nreferences: {len(references.rows)} accepted, "
          f"{len(references.errors)} rejected")
    if references.errors:
        print("  evaluation is INCOMPLETE; rejected rows were recorded, not "
              "skipped (PRD 14)")

    # ------------------------------------------------ 3. per configuration --
    cfg = ev.MatchConfig(pts_tolerance_ms=args.pts_tolerance_ms)
    metrics_by_config = {}
    object_dir = paths.dir / "09_object"
    configs = sorted(d.name for d in object_dir.iterdir()
                     if d.is_dir() and (d / "merged_detections.jsonl").is_file()) \
        if object_dir.is_dir() else []

    for config_id in configs:
        view = build_view(paths, config_id, duration_ms)
        outcomes = ev.evaluate(references, view, cfg)
        for outcome in outcomes:
            paths.append_jsonl("12_evaluation/matches.jsonl", outcome.to_row())
        entry = ev.metrics(outcomes, references, view, cfg)
        metrics_by_config[config_id] = entry
        paths.write_json(f"12_evaluation/metrics_{config_id}.json", entry)

        for outcome in outcomes:
            if outcome.outcome == ev.FN:
                paths.append_jsonl("12_evaluation/misses/misses.jsonl",
                                   outcome.to_row())
            elif outcome.outcome == ev.FP:
                paths.append_jsonl(
                    "12_evaluation/false_positives/false_positives.jsonl",
                    outcome.to_row())

        overall = entry["overall"]
        print(f"  {config_id:16} TP {overall['TP']:3}  FP {overall['FP']:3}  "
              f"FN {overall['FN']:3}  not-routed {overall['not_routed_to_candidate']:3}  "
              f"insufficient {overall['insufficient_visual_evidence']:3}")

    comparison = ev.compare(metrics_by_config,
                            taxonomy_approved=args.taxonomy_approved)
    paths.write_json("12_evaluation/metrics.json", {
        **comparison,
        "isolation_verified": True,
        "reference_summary": references.summary(),
    })

    # ---------------------------------------------------- 4. the gate audit --
    log = GateLog()
    for row in read_jsonl(paths.dir / "05_motion_roi/gate_log.jsonl"):
        known = {"gate", "condition", "state", "start_pts_ms", "end_pts_ms",
                 "reason", "inputs", "thresholds", "config_id", "artifacts",
                 "seat_id"}
        log.add(GateRecord(**{k: v for k, v in row.items() if k in known}))

    disturbances = [(o["start_pts_ms"], o["end_pts_ms"], o["condition"])
                    for o in read_jsonl(paths.dir / "02_health/observations.jsonl")]
    population = ga.sample_intervals(
        duration_ms, disturbances,
        [r.source_pts_ms for r in references.rows])
    for row in references.rows:
        population.add(ga.REFERENCE, row.reference_id,
                       row.source_pts_ms - 250.0, row.source_pts_ms + 250.0, True)

    audit = ga.audit(log, population, duration_ms)
    paths.write_json("12_evaluation/gate_audit.json", audit)
    print(f"\ngate audit over {population.counts()}")
    if audit.get("shortfall", {}).get("reason"):
        print(f"  NOTE: {audit['shortfall']['reason']}")
    for source, entry in audit.get("by_source", {}).items():
        print(f"  {source:22} false suppression "
              f"{entry['false_suppression']}/{entry['relevant']}  "
              f"false pass {entry['false_pass']}")

    print(f"\nrecommendation: {comparison['recommendation']}")
    status.finish("12_evaluation",
                  COMPLETE if references.complete else BLOCKED,
                  reason=None if references.complete else
                  f"{len(references.errors)} reference row(s) failed validation; "
                  f"metrics are incomplete",
                  references=len(references.rows),
                  configurations=len(metrics_by_config))
    status.write()
    print(f"\nwritten to {paths.dir / '12_evaluation'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
