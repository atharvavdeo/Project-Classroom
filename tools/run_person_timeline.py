"""Continuous per-person timeline over the WHOLE recording.

    python tools/run_person_timeline.py VIDEO --run artifacts/runs/1446/01_phone \
        --calibration calib/cam01_e1_v1.json --sample-hz 2 --annotate 40

This is `run_pose_ab.py`'s pipeline with exactly one thing changed: the sample
plan comes from `person_detection.plan_continuous()` instead of the candidate
manifest. Detection, tracking, seat association and pose are the same proven
code paths, so this stage cannot drift from them.

**Why the change matters.** Measured on run 1446, the candidate-gated plan
sampled 20.2% / 39.0% / 37.5% of the three recordings, so **60-80% of every
video was never decoded for person detection**. Video 01 was only looked at
between 53.6s and 102.9s. A candidate sitting still while using a phone under
the desk raises no motion candidate and was therefore never observed at all --
and no later stage can recover them, because the frames were never read.

It also made per-person metrics impossible. "This person turns right more than
they usually do" needs the ordinary frames far more than the exceptional ones,
and the gate discarded exactly those.

Writes `14_person_timeline/`:

    samples.jsonl   every person in every sampled frame, with head orientation
    profiles.json   one profile per track over the whole recording
    summary.json    cross-person view, with the coverage caveat stated
    annotated/      frames with EVERY person boxed, numbered, and given a gaze
                    arrow -- not only the ones inside an event
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import (artifacts, head_pose as hp, person_detection as pd,  # noqa: E402
                      person_timeline as pt, pose as ps, seat_association as sa,
                      tracking)
from pipeline.calibration import CalibrationProfile  # noqa: E402
from pipeline.run_manifest import COMPLETE, RunManifest, RunStatus  # noqa: E402

FACING_COLOUR = {
    hp.FRONTAL: (0, 200, 0),
    hp.TURNED_LEFT: (0, 190, 255),
    hp.TURNED_RIGHT: (255, 170, 0),
    hp.DOWNWARD: (0, 235, 235),
    hp.NOT_RESOLVABLE: (140, 140, 140),
}


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
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--person-model", type=Path,
                    default=ROOT / "models/dfine/onnx/model.onnx")
    ap.add_argument("--sample-hz", type=float, default=2.0)
    ap.add_argument("--pose-device", default="cuda")
    ap.add_argument("--annotate", type=int, default=40,
                    help="sampled frames to render with every person boxed and "
                         "given a gaze arrow. 0 none, -1 all.")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap the grid for a quick pass; 0 = no cap")
    args = ap.parse_args()

    import av

    sys.path.insert(0, str(ROOT / "tools"))
    from run_pose_ab import SequentialFrameReader  # noqa: E402

    paths = artifacts.open_run(args.run.parent, args.run.name)
    manifest_doc = RunManifest.read(paths)
    status = RunStatus(paths, manifest_doc.run_id)

    with av.open(str(args.video)) as container:
        duration_ms = float(container.duration or 0) / 1000.0
        height = container.streams.video[0].codec_context.height

    sha = ""
    manifest_path = paths.dir / "RUN_MANIFEST.json"
    if manifest_path.is_file():
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        sha = str((doc.get("sources") or [{}])[0].get("sha256", ""))

    cfg = pd.SampleConfig(sample_hz=args.sample_hz)
    rows, notes = pd.plan_continuous(
        duration_ms, source_video=args.video.name,
        source_video_sha256=sha, cfg=cfg)
    if args.max_rows and len(rows) > args.max_rows:
        step = len(rows) / args.max_rows
        rows = [rows[int(i * step)] for i in range(args.max_rows)]
        notes[0]["capped_to"] = len(rows)

    print(f"{args.video.name}: {duration_ms / 1000:.1f}s at {args.sample_hz} Hz")
    print(f"  {len(rows)} sampled frames covering the WHOLE recording "
          f"(the candidate-gated plan covered 20-39%)")

    out_dir = paths.dir / "14_person_timeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_plan_notes.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8")

    # --------------------------------------------- detection + tracking --
    reader = SequentialFrameReader(args.video)
    runner = pd.onnx_runner(args.person_model, frame_reader=reader)
    detection = pd.detect(rows, runner, "person_detection_continuous", cfg)
    print(f"  person detection: {detection.state} "
          f"({detection.returned} boxes over {detection.rows_seen} rows)")
    if not detection.boxes:
        print("no person boxes; nothing to profile", file=sys.stderr)
        return 2

    result = tracking.track(list(detection.boxes), tracking.PRIMARY)
    print(f"  tracking: {len(result.tracks)} track(s) over "
          f"{len(result.boxes)} box(es)")

    profile_doc = None
    associations = []
    if args.calibration and args.calibration.is_file():
        profile_doc = CalibrationProfile.read(args.calibration)
        associations = sa.associate(result, profile_doc)

    # --------------------------------------------------------- pose --
    rows_by_id = {r.row_id: r for r in rows}
    pose_rows = ps.freeze(associations, rows_by_id) if associations else []
    if not pose_rows:
        # No approved calibration, so seat association returns nothing. Build
        # the manifest straight from the tracked boxes instead: the point of
        # this stage is that every person is measured, and a draft calibration
        # must not silence the whole thing.
        pose_rows = []
        for box in result.boxes:
            row = rows_by_id.get(box.row_id)
            if row is None:
                continue
            pose_rows.append(ps.PoseRow(
                pose_row_id=f"{box.row_id}|{box.track_id}",
                row_id=box.row_id, event_id=row.event_id,
                source_video_sha256=row.source_video_sha256,
                pts_ms=row.pts_ms, frame_index=None,
                box=(box.x1, box.y1, box.x2, box.y2),
                box_origin="detected", track_id=box.track_id,
                seat_state="unattributed"))
        print(f"  no seat associations (draft calibration); built "
              f"{len(pose_rows)} pose rows from tracked boxes directly")

    pose_reader = SequentialFrameReader(args.video)
    pose_runner = ps.alphapose_runner(
        "alphapose_crowd_baseline",
        str(ROOT / "models/alphapose/256x192_res152_lr1e-3_1x-duc.yaml"),
        str(ROOT / "models/alphapose/fast_421_res152_256x192.pth"),
        frame_reader=pose_reader, device=args.pose_device)
    pose_result = ps.run(pose_rows, pose_runner, "alphapose_crowd_baseline",
                         profile_doc)
    print(f"  pose: {pose_result.state} "
          f"({len(pose_result.observations)} observations)")

    # ----------------------------------------- head orientation + samples --
    by_row = {r.pose_row_id: r for r in pose_rows}
    samples: list[pt.PersonSample] = []
    for observation in pose_result.observations:
        row = observation.to_row()
        joints = {j["name"]: j for j in row.get("joints", [])}
        source = by_row.get(row.get("pose_row_id"))
        track_id = getattr(source, "track_id", None)
        head = hp.estimate(joints, pose_row_id=row.get("pose_row_id", ""),
                           config_id="alphapose_crowd_baseline",
                           pts_ms=float(row["pts_ms"]),
                           seat_state=row.get("seat_state", "unattributed"),
                           track_id=track_id)

        def wrist(name):
            j = joints.get(name)
            return ((float(j["x"]), float(j["y"]))
                    if j and j.get("observability") == "visible" else None)

        samples.append(pt.PersonSample(
            track_id=track_id if track_id is not None else -1,
            pts_ms=float(row["pts_ms"]),
            box=tuple(getattr(source, "box", (0.0, 0.0, 0.0, 0.0))),
            seat_state=row.get("seat_state", "unattributed"),
            torso_scale_px=row.get("torso_scale_px"),
            yaw=head.yaw, pitch=head.pitch, roll=head.roll,
            facing=head.facing, head_scale_px=head.head_scale_px,
            left_wrist=wrist("left_wrist"), right_wrist=wrist("right_wrist"),
            wrist_below_desk=bool(row.get("wrist_in_lap_context"))))

    timelines = pt.build_timelines(samples)
    tcfg = pt.TimelineConfig()
    profiles = [pt.profile(t, duration_ms, cfg.interval_ms(), tcfg)
                for t in timelines.values()]
    profiles.sort(key=lambda p: -p.samples)
    report = pt.summarise(profiles, duration_ms, len(rows), tcfg)

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_row()) + "\n")
    (out_dir / "profiles.json").write_text(
        json.dumps([p.to_row() for p in profiles], indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{report['people']} people profiled across the whole recording "
          f"({report['people_with_a_yaw_baseline']} with a yaw baseline)")
    print(f"{'track':>6}{'samp':>6}{'span s':>9}{'cover':>8}{'baseYaw':>9}"
          f"{'away':>7}{'down':>6}{'abs':>5}  facing")
    print("-" * 88)
    for p in profiles[:20]:
        top = sorted(p.facing_counts.items(), key=lambda kv: -kv[1])[:2]
        base = f"{p.baseline_yaw:+.3f}" if p.baseline_yaw is not None else "--"
        print(f"{p.track_id:6}{p.samples:6}{p.observed_span_ms / 1000:9.1f}"
              f"{p.coverage_of_recording:8.1%}{base:>9}"
              f"{p.fraction_away_from_baseline:7.1%}"
              f"{p.samples_looking_down:6}{len(p.absences):5}  "
              f"{', '.join(f'{k[:9]}:{v}' for k, v in top)}")

    # ----------------------------------------------------- annotation --
    if args.annotate != 0 and samples:
        import cv2

        annotated = out_dir / "annotated"
        annotated.mkdir(parents=True, exist_ok=True)
        for stale in annotated.glob("*.jpg"):
            stale.unlink()

        by_pts: dict[float, list] = {}
        for sample in samples:
            by_pts.setdefault(round(sample.pts_ms, 1), []).append(sample)
        times = sorted(by_pts)
        if args.annotate > 0 and len(times) > args.annotate:
            step = len(times) / args.annotate
            times = [times[int(i * step)] for i in range(args.annotate)]

        # One forward decode pass for exactly the timestamps being drawn.
        # SequentialFrameReader serves PoseRows, not bare timestamps, and seek
        # is unreliable on this corpus (Phase 1 measured it returning to the
        # head of the file on two of six files), so decode straight through.
        want = sorted(times)
        frames, index = {}, 0
        with av.open(str(args.video)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            for decoded in container.decode(stream):
                if decoded.pts is None or index >= len(want):
                    continue
                here_ms = float(decoded.pts * time_base) * 1000.0
                while index < len(want) and here_ms >= want[index] - 1.0:
                    frames[want[index]] = decoded.to_ndarray(format="bgr24")
                    index += 1

        drawn = 0
        for pts_ms in times:
            here = by_pts[pts_ms]
            frame = frames.get(pts_ms)
            if frame is None:
                continue
            canvas = frame.copy()
            for sample in here:
                x1, y1, x2, y2 = (int(v) for v in sample.box)
                colour = FACING_COLOUR.get(sample.facing, (140, 140, 140))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
                label = f"#{sample.track_id} {sample.facing[:11]}"
                if sample.yaw is not None:
                    label += f" y{sample.yaw:+.2f}"
                cv2.putText(canvas, label, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1)
                if sample.yaw is not None:
                    hx, hy = (x1 + x2) // 2, y1 + max((y2 - y1) // 8, 6)
                    length = max((x2 - x1) * 0.45, 18.0)
                    cv2.arrowedLine(
                        canvas, (hx, hy),
                        (int(hx - sample.yaw * length),
                         int(hy + (sample.pitch or 0.0) * length * 1.5)),
                        colour, 2, tipLength=0.35)
            cv2.putText(canvas, f"{pts_ms / 1000:7.2f}s   {len(here)} people",
                        (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.imwrite(str(annotated / f"{artifacts.pts_prefix(pts_ms)}.jpg"),
                        canvas)
            drawn += 1
        print(f"\n{drawn} annotated frame(s) in {annotated}")

    status.finish("14_person_timeline", COMPLETE,
                  reason=f"continuous {args.sample_hz} Hz over the whole "
                         f"recording; {report['people']} people profiled",
                  sampled_rows=len(rows), people=report["people"])
    status.write()
    print(f"written {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
