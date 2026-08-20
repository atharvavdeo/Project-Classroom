"""Every object detection instance ever produced, as one auditable table.

    python tools/export_detections.py --out artifacts/detections_all

Until now a detection existed only as a row inside one run's per-config
`merged_detections.jsonl`. That is the right place to *write* it, and the wrong
place to *read* it: answering "show me every phone we have ever called" meant
opening twenty files and knowing which of them came from a superseded taxonomy.

This walks every run folder, joins each detection to the crop manifest row that
produced it and to the run manifest that names the source video, and writes one
CSV and one JSONL carrying the full provenance chain:

    source video sha -> pts -> crop_id -> crop config -> detector config -> box

Coordinates in `merged_detections.jsonl` are already **source-frame** pixels
(`object_detection.detect` requires runners to project their own slice offsets),
so `area_px` here is a real footprint in the recording, not a model-input area.
That is what makes a size comparison between two detections meaningful.

Nothing is filtered, re-scored or re-classified. Abstentions are exported with
their reason, because an abstention is a result.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ("run_id", "video", "video_sha256", "config_id", "pts_ms", "cls",
          "confidence", "x1", "y1", "x2", "y2", "area_px", "width_px",
          "height_px", "aspect", "abstained", "abstain_reason", "usable",
          "seat_state", "track_id", "event_id", "crop_id", "crop_config",
          "fallback_reason", "crop_native_px", "magnification", "slice_index",
          "merged_from", "calibration_version", "pose_source")


def read_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def source_of(run_dir: Path) -> tuple[str, str]:
    manifest = run_dir / "RUN_MANIFEST.json"
    if not manifest.is_file():
        return "", ""
    try:
        sources = json.loads(manifest.read_text(encoding="utf-8")).get("sources", [])
    except json.JSONDecodeError:
        return "", ""
    if not sources:
        return "", ""
    first = sources[0]
    return Path(first.get("path", "")).name, first.get("sha256", "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "artifacts/runs")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/detections_all")
    ap.add_argument("--only-runs", nargs="*", default=None,
                    help="restrict to these run ids")
    args = ap.parse_args()

    out_csv = args.out.with_suffix(".csv")
    out_jsonl = args.out.with_suffix(".jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    per_run, per_class, per_config = {}, {}, {}
    total = 0

    with out_csv.open("w", encoding="utf-8", newline="") as csv_handle, \
            out_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=FIELDS)
        writer.writeheader()

        for run_dir in sorted(d for d in args.runs.iterdir() if d.is_dir()):
            if args.only_runs and run_dir.name not in args.only_runs:
                continue
            object_dir = run_dir / "09_object"
            if not object_dir.is_dir():
                continue
            video, sha = source_of(run_dir)

            # The crop manifest is the join key: it carries which fallback
            # produced the crop and at what magnification, which the detection
            # row deliberately does not duplicate.
            crops = {row["crop_id"]: row for row in
                     read_jsonl(object_dir / "object_crop_manifest.jsonl")}

            for config_dir in sorted(d for d in object_dir.iterdir() if d.is_dir()):
                rows = list(read_jsonl(config_dir / "merged_detections.jsonl"))
                if not rows:
                    continue
                for row in rows:
                    crop = crops.get(row.get("crop_id", ""), {})
                    width = float(row.get("x2", 0)) - float(row.get("x1", 0))
                    height = float(row.get("y2", 0)) - float(row.get("y1", 0))
                    record = {
                        "run_id": run_dir.name,
                        "video": video,
                        "video_sha256": sha,
                        "config_id": row.get("config_id", config_dir.name),
                        "pts_ms": row.get("pts_ms"),
                        "cls": row.get("cls"),
                        "confidence": row.get("confidence"),
                        "x1": round(float(row.get("x1", 0)), 2),
                        "y1": round(float(row.get("y1", 0)), 2),
                        "x2": round(float(row.get("x2", 0)), 2),
                        "y2": round(float(row.get("y2", 0)), 2),
                        "area_px": row.get("area_px"),
                        "width_px": round(width, 2),
                        "height_px": round(height, 2),
                        "aspect": round(width / height, 3) if height > 0 else None,
                        "abstained": row.get("abstained"),
                        "abstain_reason": row.get("abstain_reason", ""),
                        "usable": row.get("usable"),
                        "seat_state": row.get("seat_state"),
                        "track_id": row.get("track_id"),
                        "event_id": row.get("event_id"),
                        "crop_id": row.get("crop_id"),
                        "crop_config": crop.get("crop_config", ""),
                        "fallback_reason": crop.get("fallback_reason", ""),
                        "crop_native_px": crop.get("native_px"),
                        "magnification": crop.get("magnification"),
                        "slice_index": row.get("slice_index"),
                        "merged_from": row.get("merged_from"),
                        "calibration_version": crop.get("calibration_version", ""),
                        "pose_source": crop.get("pose_source", ""),
                    }
                    writer.writerow(record)
                    jsonl_handle.write(json.dumps(record) + "\n")
                    total += 1
                    key = (run_dir.name, video)
                    per_run[key] = per_run.get(key, 0) + 1
                    per_class[record["cls"]] = per_class.get(record["cls"], 0) + 1
                    per_config[record["config_id"]] = \
                        per_config.get(record["config_id"], 0) + 1

    print(f"{total} detection instances exported")
    print(f"  {out_csv}")
    print(f"  {out_jsonl}\n")

    print(f"{'run':16} {'source video':52} {'rows':>7}")
    print("-" * 78)
    for (run_id, video), count in sorted(per_run.items()):
        print(f"{run_id:16} {(video or '(unrecorded)')[:52]:52} {count:7}")

    print(f"\n{'class':34} {'rows':>8}")
    print("-" * 44)
    for cls, count in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"{cls:34} {count:8}")

    print(f"\n{'detector configuration':24} {'rows':>8}")
    print("-" * 34)
    for config, count in sorted(per_config.items(), key=lambda kv: -kv[1]):
        print(f"{config:24} {count:8}")

    print("\nBoxes are source-frame pixels, so area_px is a real footprint in "
          "the recording. Rows from superseded taxonomies are exported as they "
          "were written; run_id and video identify which corpus each belongs to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
