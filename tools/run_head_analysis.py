"""Head and neck behaviour analysis over a completed pose stage.

    python tools/run_head_analysis.py --run artifacts/runs/1446/01 \
        --calibration calib/cam01_e1_v1.json VIDEO

Consumes `08_pose/<config>/observations.jsonl`, which already holds the five
facial keypoints AlphaPose returns, and writes `08_pose/head/`:

    per_observation.jsonl   yaw, pitch, roll and facing for every observation
    excursions.jsonl        sustained turns and downward dwells, with duration
    summary.json            per-seat metrics and neighbour bias
    overlays/               the actual frames, with the head vector drawn

Runs no model. Every number comes from keypoints already on disk, so this can
be re-run with different thresholds without touching a GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, head_pose as hp  # noqa: E402
from pipeline.calibration import CalibrationProfile  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seat_centres(profile: CalibrationProfile) -> dict:
    out = {}
    for seat in profile.seats:
        zones = getattr(seat, "zones", {}) or {}
        polygon = (zones.get("desk_work_surface") or zones.get("upper_body")
                   or getattr(seat, "polygon", None))
        if not polygon:
            continue
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        out[seat.seat_id] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return out


def decode_at(video: Path, wanted_ms: list[float]) -> dict:
    import av

    wanted = sorted(set(round(t, 1) for t in wanted_ms))
    found, index = {}, 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None or index >= len(wanted):
                continue
            pts_ms = float(frame.pts * time_base) * 1000.0
            while index < len(wanted) and pts_ms >= wanted[index] - 1.0:
                found[wanted[index]] = frame.to_ndarray(format="bgr24")
                index += 1
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--config", default="alphapose_crowd_baseline")
    ap.add_argument("--overlays", type=int, default=24,
                    help="how many frames to draw, taken from the strongest "
                         "excursions so the pictures match the findings")
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    rows = read_jsonl(paths.dir / f"08_pose/{args.config}/observations.jsonl")
    if not rows:
        print(f"no observations for {args.config} in {args.run}",
              file=sys.stderr)
        return 2

    manifest = {r["pose_row_id"]: r for r in
                read_jsonl(paths.dir / "08_pose/pose_manifest.jsonl")}

    cfg = hp.HeadPoseConfig()
    poses = []
    for row in rows:
        joints = {j["name"]: j for j in row.get("joints", [])}
        entry = manifest.get(row["pose_row_id"], {})
        poses.append(hp.estimate(
            joints, pose_row_id=row["pose_row_id"], config_id=args.config,
            pts_ms=float(row["pts_ms"]), seat_state=row.get("seat_state", ""),
            track_id=entry.get("track_id"), cfg=cfg))

    events = hp.excursions(poses, cfg)
    centres = {}
    if args.calibration and args.calibration.is_file():
        centres = seat_centres(CalibrationProfile.read(args.calibration))
    bias = hp.neighbour_bias(events, centres, cfg) if centres else {}
    report = hp.summarise(poses, events, bias, cfg)

    out_dir = paths.stage("08_pose") / "head"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_observation.jsonl").open("w", encoding="utf-8") as h:
        for pose in poses:
            h.write(json.dumps(pose.to_row()) + "\n")
    with (out_dir / "excursions.jsonl").open("w", encoding="utf-8") as h:
        for event in events:
            h.write(json.dumps(event.to_row()) + "\n")
    paths.write_json(f"08_pose/head/summary.json", report)

    print(f"{len(poses)} observations from {args.config}; "
          f"{report['resolvable_fraction']:.1%} resolvable")
    print(f"events: {report['events_by_kind']}")
    print(f"\n{'seat':14}{'obs':>5}{'resolv':>8}{'headpx':>8}"
          f"{'yaw med':>9}{'pitch med':>11}{'down ev':>9}{'longest':>9}"
          f"{'turn ev':>9}{'nbr':>6}")
    print("-" * 96)
    for seat, block in report["per_seat"].items():
        yaw = block["yaw"]["median"] if block["yaw"] else 0.0
        pitch = block["pitch"]["median"] if block["pitch"] else 0.0
        nb = block.get("neighbour_bias") or {}
        print(f"{seat[:14]:14}{block['observations']:5}"
              f"{block['resolvable_fraction']:8.1%}"
              f"{block['median_head_scale_px']:8.1f}"
              f"{yaw:+9.3f}{pitch:+11.3f}"
              f"{block['look_down_events']:9}"
              f"{block['longest_look_down_ms'] / 1000:8.1f}s"
              f"{block['turn_events']:9}"
              f"{nb.get('toward_occupied_neighbour', 0):6}")

    # Overlays, taken from the strongest events so the images correspond to the
    # numbers rather than to an arbitrary sample.
    if args.overlays > 0 and events:
        import cv2
        import numpy as np

        ranked = sorted(events, key=lambda e: -e.duration_ms)[:args.overlays]
        wanted = []
        by_pts = {}
        for event in ranked:
            mid = (event.start_pts_ms + event.end_pts_ms) / 2.0
            nearest = min((p for p in poses if p.seat_state == event.seat_state),
                          key=lambda p: abs(p.pts_ms - mid), default=None)
            if nearest:
                wanted.append(nearest.pts_ms)
                by_pts.setdefault(round(nearest.pts_ms, 1), []).append(
                    (event, nearest))

        frames = decode_at(args.video, wanted)
        overlay_dir = out_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        # Clear first. Overlays are drawn from the current strongest
        # excursions, so a file left from a previous pass points at an event
        # that no longer exists -- and the findings report lists whatever is in
        # this directory. A stale frame beside a live count is worse than no
        # frame at all.
        for stale in overlay_dir.glob("*_head.jpg"):
            stale.unlink()
        joints_by_row = {r["pose_row_id"]: r for r in rows}
        drawn = 0

        for pts_ms, items in sorted(by_pts.items()):
            frame = frames.get(pts_ms)
            if frame is None:
                continue
            canvas = frame.copy()
            for event, pose in items:
                row = joints_by_row.get(pose.pose_row_id, {})
                jm = {j["name"]: j for j in row.get("joints", [])}
                nose = jm.get("nose")
                if not nose:
                    continue
                nx, ny = int(nose["x"]), int(nose["y"])
                for name, colour in (("left_eye", (255, 200, 0)),
                                     ("right_eye", (255, 200, 0)),
                                     ("left_ear", (0, 200, 255)),
                                     ("right_ear", (0, 200, 255)),
                                     ("nose", (0, 0, 255))):
                    j = jm.get(name)
                    if j and j["observability"] == "visible":
                        cv2.circle(canvas, (int(j["x"]), int(j["y"])), 2,
                                   colour, -1)
                # Head vector: direction from yaw, length from magnitude.
                if pose.yaw is not None:
                    length = 26.0
                    dx = -pose.yaw * length
                    dy = (pose.pitch or 0.0) * length * 2.0
                    cv2.arrowedLine(canvas, (nx, ny),
                                    (int(nx + dx), int(ny + dy)),
                                    (0, 255, 0), 2, tipLength=0.3)
                label = (f"{event.seat_state} {event.kind} "
                         f"{event.duration_ms / 1000:.1f}s "
                         f"yaw={pose.yaw if pose.yaw is not None else 0:+.2f} "
                         f"pitch={pose.pitch if pose.pitch is not None else 0:+.2f}")
                cv2.putText(canvas, label, (max(nx - 90, 4), max(ny - 16, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            cv2.imwrite(str(overlay_dir
                            / f"{artifacts.pts_prefix(pts_ms)}_head.jpg"),
                        canvas)
            drawn += 1
        print(f"\n{drawn} overlay frame(s) in {overlay_dir}")

    print(f"written {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
