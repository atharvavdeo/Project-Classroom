"""Render the detector/SAHI comparison as Markdown (PRD 10, PRD 15).

    python tools/render_detector_comparison.py --run artifacts/runs/<run_id>

Writes `13_reports/detector_comparison.md`. Three things it will not do:

**It will not name a winner.** PRD 3 blocks detector selection until the
taxonomy is approved, and PRD 15 places selection at stage 12 against the
reference manifest. Until then the table is a record of what each configuration
produced, not a ranking.

**It will not merge phone and chit.** PRD 15: "Phone and chit results must be
separate." They fail differently -- a phone is a rigid bright rectangle, a chit
is a soft low-contrast one -- and one averaged number hides whichever is worse.

**It will not hide the abstentions.** The share of crops the source could not
support is a property of the camera, and it belongs next to the detection counts
rather than in a footnote, because it bounds what any detector here could
possibly have found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts  # noqa: E402
from tools.render_pose_comparison import table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    path = paths.dir / "09_object/comparison.json"
    if not path.is_file():
        print("no 09_object/comparison.json; run tools/run_detector_ab.py first.",
              file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))

    crop_summary = {}
    crop_path = paths.dir / "09_object/crop_manifest_summary.json"
    if crop_path.is_file():
        crop_summary = json.loads(crop_path.read_text(encoding="utf-8"))

    confirmation = {}
    confirm_path = paths.dir / "10_evidence/temporal_confirmation_summary.json"
    if confirm_path.is_file():
        confirmation = json.loads(confirm_path.read_text(encoding="utf-8"))

    lines = ["# Detector and slicing comparison", ""]
    lines += [
        f"**Crop equivalence: {report['crop_equivalence']}** — "
        f"{report['crop_rows']} crops, digest `{report['crop_digest'][:16]}`",
        "", report["equivalence_note"], "",
    ]
    if report["crop_equivalence"] != "verified":
        lines += ["> The table below is retained for diagnosis only. Comparing "
                  "these numbers would attribute an input difference to a "
                  "detector, which PRD 4 exists to prevent.", ""]

    if crop_summary:
        lines += ["## The crops every configuration received", "",
                  table([[k, str(v)] for k, v in
                         sorted(crop_summary["by_crop_config"].items())],
                        ["Crop policy", "Crops"]), ""]
        if crop_summary.get("fallback_reasons"):
            lines += ["Why crops fell back from `pose_hand_context`:", "",
                      table([[k, str(v)] for k, v in
                             sorted(crop_summary["fallback_reasons"].items())],
                            ["Reason", "Crops"]), ""]
        lines += [
            f"- **{crop_summary['insufficient_native_resolution']} of "
            f"{crop_summary['crops']} crops "
            f"({crop_summary['insufficient_fraction'] * 100:.1f}%)** are below "
            f"the native-resolution floor. No detector can answer for these, "
            f"and that is a statement about the camera.",
            f"- Median magnification "
            f"**{crop_summary['median_magnification']}x**, max "
            f"**{crop_summary['max_magnification']}x**. "
            f"{crop_summary['note']}",
            "",
        ]

    lines += ["## Configurations", ""]
    header = ["Configuration", "State", "Crops", "Abstained", "Slices",
              "Merged", "Phone", "Chit", "Hard negatives", "Crops/s"]
    body = []
    for entry in report["configurations"]:
        if entry.get("state") != "available":
            body.append([entry["config_id"], entry["state"]] + ["—"] * 8)
            continue
        body.append([
            entry["config_id"], entry["state"], str(entry["crops_seen"]),
            f"{entry['abstention_rate'] * 100:.1f}%",
            str(entry["slices_run"] or "—"),
            str(entry["merged_detections"]),
            str(entry["phone"]), str(entry["secondary_paper_chit"]),
            str(entry["hard_negative_detections"]),
            str(entry["crops_per_second"] or "—"),
        ])
    lines += [table(body, header), "",
              "Phone and chit are reported separately throughout, per PRD 15. "
              "**Hard negatives** counts calls on mouse, keyboard, calculator, "
              "answer sheet, monitor bezel and stationery — the objects that "
              "look like a phone at 30x20 px. PRD 10 treats a phone called on "
              "one of these as a failure cluster to fix, not a false positive "
              "to average away.", ""]

    if report["unavailable"]:
        lines += ["### Unavailable configurations", "",
                  "Retained per PRD 5; never substituted by an available one.", "",
                  table([[e["config_id"], e["state"], e["reason"]]
                         for e in report["unavailable"]],
                        ["Configuration", "State", "Reason"]), ""]

    if confirmation.get("by_status"):
        lines += ["## Temporal confirmation", "",
                  table([[k, str(v)] for k, v in
                         sorted(confirmation["by_status"].items())],
                        ["Status", "Groups"]), "",
                  f"**{confirmation['retained_singletons']} singletons retained, "
                  f"{confirmation['deleted']} deleted.** {confirmation['note']}",
                  ""]

    lines += ["## Selection", "", report["selection"], "",
              f"Taxonomy approved: **{report['taxonomy_approved']}**", ""]

    out = paths.dir / "13_reports/detector_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
