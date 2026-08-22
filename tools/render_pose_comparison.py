"""Render the pose A/B comparison as a reviewable Markdown table (PRD 9).

    python tools/render_pose_comparison.py --run artifacts/runs/<run_id>

Writes `13_reports/pose_comparison.md`. Three properties are deliberate:

**Unavailable configurations are rows, not omissions.** PRD 5 requires a failed
configuration to stay in the table. A comparison that lists only what ran reads
as a complete comparison of a smaller field, which is a different and more
flattering claim than the truth.

**No winner is named.** PRD 5 forbids calling any configuration best until every
runnable mandatory configuration has a matched report, and selection belongs to
stage 12 against references. The report says so in place of a recommendation.

**Manifest equivalence is stated first.** If the configurations did not process
identical rows, every number below is incomparable, and that has to be the first
thing a reader sees rather than a footnote under the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts  # noqa: E402


def table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows
              else len(header[i]) for i in range(len(header))]
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    comparison_path = paths.dir / "08_pose/comparison.json"
    if not comparison_path.is_file():
        print("no 08_pose/comparison.json; run tools/run_pose_ab.py first.",
              file=sys.stderr)
        return 2
    report = json.loads(comparison_path.read_text(encoding="utf-8"))

    lines = ["# Pose configuration comparison", ""]
    equivalence = report["manifest_equivalence"]
    lines += [
        f"**Manifest equivalence: {equivalence}** — "
        f"{report['manifest_rows']} rows, digest `{report['manifest_digest'][:16]}`",
        "",
        report["equivalence_note"],
        "",
    ]
    if equivalence != "verified":
        lines += ["> The table below is retained for diagnosis only. Comparing "
                  "these numbers would attribute an input difference to a "
                  "model, which PRD 4 exists to prevent.", ""]

    lines += ["## Rows processed", ""]
    lines.append(table(
        [[k, str(v)] for k, v in sorted(report["rows_by_seat_state"].items())],
        ["Seat state", "Rows"]))
    lines += ["", table(
        [[k, str(v)] for k, v in sorted(report["rows_by_health_state"].items())],
        ["Health state", "Rows"]), ""]

    lines += ["## Configurations", ""]
    header = ["Configuration", "State", "Rows", "Joints visible",
              "Wrist visible", "Wrist occluded", "Wrist unresolvable",
              "Swaps flagged", "Rows/s"]
    body = []
    for entry in report["configurations"]:
        if entry.get("state") != "available":
            body.append([entry["config_id"], entry["state"], "—", "—", "—",
                         "—", "—", "—", "—"])
            continue
        wrist = entry["wrist_observability"]
        body.append([
            entry["config_id"], entry["state"], str(entry["rows_seen"]),
            f"{entry['joint_visible_fraction'] * 100:.1f}%",
            str(wrist.get("visible", 0)),
            str(wrist.get("occluded", 0)),
            str(wrist.get("not_resolvable_at_source_scale", 0)),
            str(entry["suspected_joint_swaps"]),
            str(entry["rows_per_second"] or "—"),
        ])
    lines += [table(body, header), ""]

    if report["unavailable"]:
        lines += ["### Unavailable configurations", "",
                  "Retained in the comparison per PRD 5; never substituted.", ""]
        lines.append(table(
            [[e["config_id"], e["state"], e["reason"]]
             for e in report["unavailable"]],
            ["Configuration", "State", "Reason"]))
        lines.append("")

    lines += ["## Selection", "", report["selection"], "",
              "### What the wrist columns mean", "",
              "- **visible** — the estimator returned a wrist above the "
              "confidence floor on a person large enough to resolve one.",
              "- **occluded** — a wrist the estimator could not see, most often "
              "because the desk is between it and the camera. This is the "
              "interesting case, not a gap.",
              "- **unresolvable** — the person box is below the source-scale "
              "floor. This is a statement about the camera, not the person, and "
              "no amount of model improvement changes it.", ""]

    out = paths.dir / "13_reports/pose_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
