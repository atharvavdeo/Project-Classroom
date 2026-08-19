"""Aggregate every per-video run into one cross-corpus report.

    python tools/summarise_corpus.py --tag corpus

Reads the run folders only. It computes nothing new and re-runs nothing: every
number here is copied from a run's own artifacts, so a figure in this report can
always be traced to the run that produced it.

Deliberately reports per video rather than pooling. Pooling would average a
reviewed calibration together with five draft ones, a 131-second clip with a
two-hour recording, and a 640x480 camera with five 720p ones -- and the average
would describe none of them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.render_pose_comparison import table  # noqa: E402


def load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="corpus")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/Corpus_Evaluation.md")
    args = ap.parse_args()

    runs = sorted((ROOT / "artifacts/runs").glob(f"{args.tag}_*"))
    if not runs:
        print(f"no runs matching {args.tag}_*", file=sys.stderr)
        return 2

    lines = ["# Corpus evaluation — videos 01 to 06", "",
             "Every number below is copied from a run folder under "
             "`artifacts/runs/`. Nothing here is recomputed, averaged across "
             "videos, or estimated.", ""]

    # ------------------------------------------------------------ coverage --
    coverage, pose_rows, detector_rows, gate_rows = [], [], [], []
    for run in runs:
        prefix = run.name.split("_")[-1]
        manifest = load(run / "RUN_MANIFEST.json") or {}
        status = load(run / "RUN_STATUS.json") or {}
        stages = {s["stage"]: s for s in status.get("stages", [])}
        motion = stages.get("05_motion_roi", {})
        seg = stages.get("06_segmentation", {})
        cap = load(run / "07_tracking/event_cap.json")

        source = (manifest.get("sources") or [{}])[0]
        duration_s = (motion.get("total_ms") or 0) / 1000.0
        coverage.append([
            prefix,
            f"{duration_s:.0f}",
            str(motion.get("counts", {}).get("windows", "—")),
            str(seg.get("counts", {}).get("events", "—")),
            str(seg.get("counts", {}).get("normal_priority", "—")),
            str(cap["events_processed"]) if cap else "all",
            (source.get("sha256") or "")[:10],
        ])

        # --------------------------------------------------------- pose --
        pose = load(run / "08_pose/comparison.json")
        if pose:
            for entry in pose["configurations"]:
                if entry.get("state") != "available":
                    pose_rows.append([prefix, entry["config_id"], entry["state"],
                                      "—", "—", "—", "—", "—"])
                    continue
                wrist = entry["wrist_observability"]
                pose_rows.append([
                    prefix, entry["config_id"], entry["state"],
                    f"{entry['rows_seen']}",
                    f"{entry['joint_visible_fraction'] * 100:.1f}%",
                    str(wrist.get("visible", 0)),
                    str(wrist.get("occluded", 0)),
                    str(entry["rows_per_second"] or "—")])

        # ----------------------------------------------------- detector --
        det = load(run / "09_object/comparison.json")
        if det:
            for entry in det["configurations"]:
                if entry.get("state") != "available":
                    detector_rows.append([prefix, entry["config_id"],
                                          entry["state"]] + ["—"] * 5)
                    continue
                detector_rows.append([
                    prefix, entry["config_id"], entry["state"],
                    str(entry["merged_detections"]), str(entry["phone"]),
                    str(entry["secondary_paper_chit"]),
                    str(entry["hard_negative_detections"]),
                    str(entry["crops_per_second"] or "—")])

        # --------------------------------------------------------- gates --
        gates = load(run / "05_motion_roi/gate_log.jsonl")
        gate_summary = load(run / "12_evaluation/gate_audit.json")
        gate_rows.append([
            prefix,
            str(count_lines(run / "05_motion_roi/gate_log.jsonl")),
            str(count_lines(run / "02_health/observations.jsonl")),
            str(count_lines(run / "10_evidence/temporal_confirmation.jsonl")),
            (gate_summary or {}).get("overall", {}).get("status", "—"),
        ])

    lines += ["## Complete-duration coverage", "",
              table(coverage, ["Video", "Duration s", "Zone windows", "Events",
                               "Normal priority", "Events modelled",
                               "Source sha256"]),
              "",
              "**Events modelled** is `all` unless a cap was applied, in which "
              "case the capped events were sampled evenly across the recording "
              "rather than taken from its first minutes. Motion always covered "
              "the complete decodable duration.", ""]

    lines += ["## Pose comparison, per video", "",
              "Every configuration in a row received byte-identical manifest "
              "rows; the digest is recorded in each run's "
              "`08_pose/comparison.json`.", "",
              table(pose_rows, ["Video", "Configuration", "State", "Rows",
                                "Joints visible", "Wrist visible",
                                "Wrist occluded", "Rows/s"]), "",
              "> **AlphaPose covers every row by construction.** It is top-down "
              "and is handed the manifest row's person box, so it cannot fail "
              "to return a person. RTMPose and RTMO run natively on the full "
              "frame and are matched back by IoU, so they can decline. A higher "
              "row count for AlphaPose is a property of the harness, not "
              "evidence that it finds more people.", ""]

    lines += ["## Detector and slicing comparison, per video", "",
              table(detector_rows, ["Video", "Configuration", "State", "Merged",
                                    "Phone", "Chit", "Hard negatives",
                                    "Crops/s"]), "",
              "**Hard negatives** counts calls on mouse, keyboard, calculator, "
              "answer sheet, monitor bezel and stationery. Chit is structurally "
              "zero everywhere: the stock COCO head has no "
              "`secondary_paper_chit` class, so no chit claim of any kind is "
              "supportable from this corpus.", ""]

    lines += ["## Gates and health", "",
              table(gate_rows, ["Video", "Gate records", "Health observations",
                                "Confirmed object groups", "Gate audit"]), "",
              "Gate audit reads `unaudited` wherever no reference manifest "
              "existed, which is every video in this corpus. That is not a "
              "zero-error result.", ""]

    # -------------------------------------------------------- limitations --
    lines += ["## What this corpus cannot support", "",
              "- **No accuracy of any kind.** No reference manifest exists, so "
              "there is no precision, recall, F1 or false-positives-per-hour "
              "figure anywhere in this report, and none may be inferred from "
              "the detection counts above.",
              "- **No seat-attributed claim for videos 01, 02, 04, 05 or 06.** "
              "Their calibration profiles are DRAFT, proposed automatically "
              "from person detections and never reviewed, so attribution "
              "returns `unattributed` by design (PRD 3). Only video 03 has a "
              "reviewed profile.",
              "- **No chit claim.** See above.",
              "- **No 30-60 person claim.** The validated corpus is 1-17 "
              "people.",
              "- **No GPU timing.** Every figure here was measured on CPU; the "
              "machine has a 4 GB RTX 3050 and a CPU-only torch build, and each "
              "run's manifest records the device per configuration.", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {args.out} from {len(runs)} run(s)")
    for row in coverage:
        print(f"  {row[0]}  {row[1]:>6}s  {row[3]:>5} events  "
              f"{row[4]:>4} normal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
