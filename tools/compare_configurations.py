"""Render the cross-configuration comparison from stage 12 (PRD 15).

    python tools/compare_configurations.py --run artifacts/runs/<run_id>

Writes `13_reports/configuration_comparison.md`: every configuration, its
availability, its per-class metrics with phone and chit reported separately, the
stage-loss waterfall, and the gate audit split by population.

Two things this will not print.

**A winner the numbers do not support.** `pipeline.evaluate.compare` decides
that, and it declines when the taxonomy is unapproved, when the reference
manifest has rejected rows, or when the margin between the top two is inside
what the corpus can resolve.

**A ratio where there was no question.** An undefined precision prints as `—`,
never as 0.000. A precision of 0.000 says the detector was wrong; `—` says it
was never asked, and on a corpus this size that distinction covers most of the
table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts  # noqa: E402
from pipeline.evaluate import LOSS_STAGES, OUTCOMES  # noqa: E402
from pipeline.run_manifest import RunManifest  # noqa: E402
from tools.render_pose_comparison import table  # noqa: E402


def fmt(value) -> str:
    """A number, or an em dash for undefined. Never a zero standing in."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    manifest = RunManifest.read(paths)
    metrics_path = paths.dir / "12_evaluation/metrics.json"
    if not metrics_path.is_file():
        print("no 12_evaluation/metrics.json; run tools/evaluate_run.py first.",
              file=sys.stderr)
        return 2
    report = json.loads(metrics_path.read_text(encoding="utf-8"))

    lines = [f"# Configuration comparison — run `{manifest.run_id}`", ""]

    isolation = paths.dir / "12_evaluation/isolation_check.json"
    if isolation.is_file():
        check = json.loads(isolation.read_text(encoding="utf-8"))
        lines += [f"**Reference isolation: "
                  f"{'verified' if check['isolated'] else 'VIOLATED'}** across "
                  f"{len(check['stages_checked'])} pre-evaluation stages.", "",
                  check["note"], ""]

    if not report.get("configurations"):
        lines += ["## Evaluation incomplete", "",
                  report.get("reason", "No configuration produced evaluable "
                                       "output."), "",
                  "No precision, recall, F1 or false-positives-per-hour figure "
                  "is available, and none may be inferred from the detection "
                  "counts elsewhere in this run.", ""]
        out = paths.dir / "13_reports/configuration_comparison.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"written {out} (evaluation incomplete)")
        return 0

    summary = report.get("reference_summary", {})
    lines += ["## References", "",
              table([["Accepted rows", str(summary.get("accepted_rows", 0))],
                     ["Rejected rows", str(summary.get("rejected_rows", 0))],
                     ["Complete", str(summary.get("complete"))],
                     ["Localised rows", str(summary.get("localised_rows", 0))]],
                    ["Measure", "Value"]), "",
              summary.get("note", ""), ""]

    lines += ["## Overall, per configuration", "",
              "An undefined ratio prints as `—`. A precision of 0.000 means the "
              "detector was wrong; `—` means it was never asked.", ""]
    header = ["Configuration", "TP", "FP", "FN", "Not routed", "Insufficient",
              "Quality unavail.", "Precision", "Recall", "F1", "FP/hour"]
    body = []
    for config_id, entry in report["configurations"].items():
        overall = entry["overall"]
        body.append([config_id, str(overall["TP"]), str(overall["FP"]),
                     str(overall["FN"]),
                     str(overall["not_routed_to_candidate"]),
                     str(overall["insufficient_visual_evidence"]),
                     str(overall["quality_unavailable"]),
                     fmt(overall["precision"]), fmt(overall["recall"]),
                     fmt(overall["f1"]), fmt(overall["fp_per_hour"])])
    lines += [table(body, header), ""]

    lines += ["## Phone and chit, separately (PRD 15)", ""]
    for cls in ("phone", "secondary_paper_chit"):
        rows = []
        for config_id, entry in report["configurations"].items():
            per = entry["per_class"].get(cls)
            if not per:
                continue
            rows.append([config_id, str(per["reference_count"]), str(per["TP"]),
                         str(per["FP"]), str(per["FN"]),
                         fmt(per["precision"]), fmt(per["recall"]),
                         fmt(per["candidate_routing_recall"]),
                         fmt(per["recall_conditional_on_routing"]),
                         fmt(per["full_pipeline_recall"])])
        if rows:
            lines += [f"### `{cls}`", "",
                      table(rows, ["Configuration", "Refs", "TP", "FP", "FN",
                                   "Precision", "Recall", "Routing recall",
                                   "Recall | routed", "Full-pipeline recall"]),
                      ""]
    lines += ["**Routing recall** is the share of references motion routed to a "
              "candidate at all. **Recall | routed** is what the rest of the "
              "pipeline then found among those. Quoting only one of these is "
              "how a pipeline gets optimised in the wrong direction: a recall "
              "figure that blames the detector for the motion gate leads "
              "directly to loosening the gate until everything is a candidate.",
              ""]

    lines += ["## Stage-loss waterfall", "",
              "The earliest stage that could have carried a reference and did "
              "not. A reference lost at `motion_roi` was never shown to a "
              "detector, so it is not a detector error.", ""]
    rows = []
    for config_id, entry in report["configurations"].items():
        waterfall = entry["stage_loss_waterfall"]
        rows.append([config_id] + [str(waterfall.get(s, 0)) for s in LOSS_STAGES])
    lines += [table(rows, ["Configuration"] + list(LOSS_STAGES)), ""]

    lines += ["## Outcome breakdown by visibility", ""]
    for config_id, entry in report["configurations"].items():
        by_visibility = entry.get("by_visibility") or {}
        if not by_visibility:
            continue
        rows = [[visibility] + [str(counts.get(o, 0)) for o in OUTCOMES]
                for visibility, counts in sorted(by_visibility.items())]
        lines += [f"### `{config_id}`", "",
                  table(rows, ["Visibility"] + list(OUTCOMES)), ""]

    audit_path = paths.dir / "12_evaluation/gate_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        lines += ["## Gate audit", "", audit.get("note", ""), ""]
        if audit.get("shortfall", {}).get("reason"):
            lines += [f"> **{audit['shortfall']['reason']}**", ""]
        rows = []
        for source, entry in (audit.get("by_source") or {}).items():
            rows.append([source, str(entry["intervals"]), str(entry["relevant"]),
                         str(entry["correct_pass"]),
                         str(entry["false_suppression"]),
                         str(entry["false_pass"]),
                         str(entry["correct_suppression"]),
                         fmt(entry["false_suppression_rate"])])
        if rows:
            lines += [table(rows, ["Population", "Intervals", "Relevant",
                                   "Correct pass", "False suppression",
                                   "False pass", "Correct suppression",
                                   "False-supp. rate"]), ""]
        else:
            lines += [audit.get("overall", {}).get("detail", ""), ""]

    lines += ["## Configuration availability", "",
              table([[m["config_id"], m["state"], str(m.get("reason") or "")[:70]]
                     for m in manifest.to_dict().get("models", [])],
                    ["Configuration", "State", "Reason"]), "",
              "A configuration that could not run stays in this table as "
              "unavailable and is never substituted (PRD 5).", ""]

    lines += ["## Recommendation", "", report["recommendation"], "",
              report.get("note", ""), ""]

    out = paths.dir / "13_reports/configuration_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
