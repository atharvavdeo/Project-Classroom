"""One readable report across the 1446 run's videos.

    python tools/summarise_1446.py

Pulls together, per video: what was ingested, which preprocessing transform did
what, whether the camera moved, how many people were tracked and posed, the
head and neck behaviour, every object class detected, and what a reviewer is
being asked to open.

Reports per video and **never pools across them**. The recordings differ in
resolution, frame rate and camera geometry -- video 04 is 640x480 at 8 fps
while the others are 720p at 25 -- so a pooled average would describe no
camera that exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def block(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=ROOT / "artifacts/runs/1446")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    videos = sorted(d for d in args.run.iterdir()
                    if d.is_dir() and (d / "RUN_MANIFEST.json").is_file())
    if not videos:
        print(f"no completed videos under {args.run}", file=sys.stderr)
        return 2

    report = {}
    for run_dir in videos:
        label = run_dir.name
        manifest = read_json(run_dir / "RUN_MANIFEST.json") or {}
        source = (manifest.get("sources") or [{}])[0]
        entry = {"source": Path(source.get("path", "")).name,
                 "sha256": source.get("sha256", "")[:16]}

        block(f"{label}   {entry['source'][:60]}")

        # ---------------------------------------------------- preprocessing --
        pp = read_json(run_dir / "01_ingest/preprocessing/comparison.json")
        if pp:
            print("\npreprocessing (measured, never applied to evaluation)")
            print(f"  {'transform':28}{'d sharpness':>13}{'d blocking':>12}"
                  f"{'mean|diff|':>12}{'sec':>7}")
            entry["preprocessing"] = {}
            for name, cfgblock in pp["configurations"].items():
                delta = cfgblock["delta"]
                print(f"  {name:28}{delta['sharpness']:+13.2f}"
                      f"{delta['blocking_ratio']:+12.4f}"
                      f"{cfgblock['mean_abs_difference']:12.3f}"
                      f"{cfgblock['seconds']:7.1f}")
                entry["preprocessing"][name] = {
                    "d_sharpness": delta["sharpness"],
                    "d_blocking": delta["blocking_ratio"]}

        # -------------------------------------------------------- alignment --
        al = read_json(run_dir / "03_alignment/comparison.json")
        if al:
            print(f"\ncamera motion  (static background support "
                  f"{al['static_support_coverage']:.1%}, "
                  f"{len(al['epoch_breaks'])} epoch break(s))")
            for name, cfgblock in al["configurations"].items():
                print(f"  {name:20}{cfgblock['passed_gate_and_compensated']:3}"
                      f"/{cfgblock['estimates']:<3} compensated   "
                      f"median residual {cfgblock['median_residual']:8.4f}")
            entry["alignment"] = al["configurations"]

        # ------------------------------------------------------------- pose --
        pose = read_json(run_dir / "08_pose/comparison.json")
        if pose:
            print(f"\npose A/B on {pose['manifest_rows']} identical rows "
                  f"(equivalence {pose['manifest_equivalence']})")
            print(f"  {'configuration':28}{'returned':>10}{'coverage':>10}"
                  f"{'joints vis':>12}{'wrist vis':>11}{'rows/s':>8}")
            entry["pose"] = {}
            for cfgblock in pose["configurations"]:
                wrist = cfgblock.get("wrist_observability", {})
                print(f"  {cfgblock['config_id']:28}"
                      f"{cfgblock.get('persons_returned', 0):10}"
                      f"{cfgblock.get('row_coverage', 0):10.1%}"
                      f"{cfgblock.get('joint_visible_fraction', 0):12.3f}"
                      f"{wrist.get('visible', 0):11}"
                      f"{cfgblock.get('rows_per_second', 0):8.2f}")
                entry["pose"][cfgblock["config_id"]] = {
                    "returned": cfgblock.get("persons_returned"),
                    "coverage": cfgblock.get("row_coverage"),
                    "joint_visible_fraction":
                        cfgblock.get("joint_visible_fraction")}

        # ------------------------------------------------- head behaviour --
        head = read_json(run_dir / "08_pose/head/summary.json")
        if head:
            print(f"\nhead and neck  ({head['observations']} observations, "
                  f"{head['resolvable_fraction']:.1%} resolvable, "
                  f"{head['seats']} tracked subject(s))")
            print(f"  events: {head['events_by_kind']}")
            ranked = sorted(head["per_seat"].items(),
                            key=lambda kv: -(kv[1]["turn_events"]
                                             + kv[1]["look_down_events"]))
            shown = [(k, v) for k, v in ranked
                     if v["turn_events"] or v["look_down_events"]][:12]
            if shown:
                print(f"  {'subject':26}{'obs':>5}{'resolv':>8}{'headpx':>8}"
                      f"{'downs':>7}{'longest':>9}{'turns':>7}{'nbr':>5}"
                      f"{'repeat':>8}")
                for key, s in shown:
                    nb = s.get("neighbour_bias") or {}
                    print(f"  {key[-26:]:26}{s['observations']:5}"
                          f"{s['resolvable_fraction']:8.1%}"
                          f"{s['median_head_scale_px']:8.1f}"
                          f"{s['look_down_events']:7}"
                          f"{s['longest_look_down_ms'] / 1000:8.1f}s"
                          f"{s['turn_events']:7}"
                          f"{nb.get('toward_occupied_neighbour', 0):5}"
                          f"{nb.get('repeated_same_direction', 0):8}")
            else:
                print("  no sustained head event passed the duration floor")
            entry["head"] = {
                "observations": head["observations"],
                "resolvable_fraction": head["resolvable_fraction"],
                "events_by_kind": head["events_by_kind"],
                "subjects": head["seats"]}

        # ----------------------------------------------------------- objects --
        obj = read_json(run_dir / "09_object/comparison.json")
        if obj:
            print(f"\nobjects on {obj['crop_rows']} identical crops "
                  f"(equivalence {obj['crop_equivalence']})")
            print(f"  {'configuration':20}{'merged':>8}{'phone':>7}{'chit':>6}"
                  f"{'stationery':>12}{'hard neg':>10}{'crops/s':>9}")
            entry["objects"] = {}
            for cfgblock in obj["configurations"]:
                if cfgblock.get("state") != "available":
                    print(f"  {cfgblock['config_id']:20}  "
                          f"{cfgblock.get('state')}: "
                          f"{str(cfgblock.get('reason', ''))[:44]}")
                    continue
                by = cfgblock.get("by_class", {})
                print(f"  {cfgblock['config_id']:20}"
                      f"{cfgblock.get('merged_detections', 0):8}"
                      f"{by.get('phone', 0):7}"
                      f"{by.get('secondary_paper_chit', 0):6}"
                      f"{by.get('stationery', 0):12}"
                      f"{cfgblock.get('hard_negative_detections', 0):10}"
                      f"{cfgblock.get('crops_per_second', 0):9.2f}")
                entry["objects"][cfgblock["config_id"]] = by

        # ------------------------------------------------------ dispositions --
        ranked_rows = read_jsonl(run_dir / "10_evidence/ranked_events.jsonl")
        if ranked_rows:
            counts: dict[str, int] = {}
            for row in ranked_rows:
                key = row.get("disposition", "unknown")
                counts[key] = counts.get(key, 0) + 1
            queue = sum(v for k, v in counts.items()
                        if k in ("high_potential_corroborated",
                                 "flagged_for_review"))
            print(f"\nreview dispositions over {len(ranked_rows)} event(s)")
            for key, value in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"  {key:34}{value:5}")
            print(f"  {'-> reviewer queue':34}{queue:5}  "
                  f"({queue / len(ranked_rows):.0%} of events)")
            entry["dispositions"] = counts
            entry["review_queue"] = queue

        # --------------------------------------------------------- artifacts --
        images = sum(1 for p in run_dir.rglob("*")
                     if p.suffix.lower() in (".jpg", ".png"))
        files = sum(1 for p in run_dir.rglob("*") if p.is_file())
        print(f"\nartifacts: {files} files, {images} images")
        entry["artifact_files"] = files
        entry["artifact_images"] = images
        report[label] = entry

    out = args.out or (args.run / "SUMMARY_1446.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n\nwritten {out}")
    print("\nReported per video and never pooled: the recordings differ in "
          "resolution, frame rate and camera geometry, so a pooled average "
          "would describe no camera that exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
