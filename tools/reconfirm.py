"""Re-run temporal confirmation over detections already on disk.

    python tools/reconfirm.py --run artifacts/runs/1446/01_phone VIDEO

Grouping, recurrence and handheld plausibility are all post-passes over
`09_object/<config>/merged_detections.jsonl`. When those rules change there is
no need to run the detector again -- the expensive part, 10 to 28 minutes per
video on this corpus, produced rows that have not changed.

Rewrites `10_evidence/temporal_confirmation.jsonl` and its summary in place.
Ranking is not touched here; re-run `render_evidence_packet.py` afterwards so
the dispositions follow the new statuses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, temporal_confirmation as tc  # noqa: E402
from pipeline.object_detection import ObjectDetection  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pose-source", default="alphapose_crowd_baseline")
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)

    import av
    with av.open(str(args.video)) as container:
        frame_w = float(container.streams.video[0].codec_context.width)
    cfg = tc.ConfirmConfig(frame_width_px=frame_w)
    print(f"{args.run.name}: {frame_w:.0f} px frame, recurrence grid "
          f"{cfg.recurrence_grid_fraction * frame_w:.1f} px")

    # Wrists from the same pose observations the crops were built from.
    wrists: dict = {}
    for row in read_jsonl(
            paths.dir / f"08_pose/{args.pose_source}/observations.jsonl"):
        scale = row.get("torso_scale_px")
        if not scale:
            continue
        frame = wrists.setdefault(round(float(row["pts_ms"]), 1), [])
        for joint in row.get("joints", []):
            if joint.get("name") in ("left_wrist", "right_wrist") \
                    and joint.get("observability") == "visible":
                frame.append((float(joint["x"]), float(joint["y"]),
                              float(scale)))
    print(f"  wrists on {len(wrists)} frame(s)")

    config_dirs = sorted(
        d for d in (paths.dir / "09_object").iterdir()
        if d.is_dir() and (d / "merged_detections.jsonl").is_file())
    if not config_dirs:
        print("no merged detections to re-confirm", file=sys.stderr)
        return 2

    target = paths.dir / "10_evidence/temporal_confirmation.jsonl"
    if target.exists():
        target.unlink()

    all_groups = []
    for config_dir in config_dirs:
        rows = read_jsonl(config_dir / "merged_detections.jsonl")
        detections = [ObjectDetection(**{k: v for k, v in row.items()
                                         if k in ObjectDetection.__dataclass_fields__})
                      for row in rows]
        groups = tc.confirm(detections, cfg)
        groups = tc.flag_recurrent(groups, cfg)
        groups = tc.flag_handheld(groups, wrists, cfg)
        recurrent = sum(1 for g in groups
                        if g.status == tc.RECURRENT_SCENE_OBJECT)
        phones = [g for g in groups if g.cls == "phone"]
        supported = sum(1 for g in phones if g.status == tc.SUPPORTED)
        print(f"  {config_dir.name:16} {len(rows):6} detections -> "
              f"{len(groups):6} groups, {recurrent:6} recurrent, "
              f"phone supported {supported}")
        all_groups += groups
        for group in groups:
            paths.append_jsonl("10_evidence/temporal_confirmation.jsonl",
                               group.to_row())

    summary = tc.summarise(all_groups)
    paths.write_json("10_evidence/temporal_confirmation_summary.json", summary)
    print(f"\n  {summary['groups']} groups: {summary['by_status']}")
    print(f"written {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
