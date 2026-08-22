"""Render the whole recording as an annotated video.

    python tools/render_annotated_video.py VIDEO \
        --run artifacts/runs/1446/04_talking --fps 8

Reads `14_person_timeline/samples.jsonl` -- one row per person per sampled
frame, carrying the box, track id, head yaw/pitch, facing state and every
resolved keypoint -- and writes `14_person_timeline/annotated_video.mp4`.

Every person in every frame is drawn: skeleton, box, track id, facing state and
a gaze vector. Not only the people inside a motion event, and not only the
frames something happened in. A reviewer scrubbing this video should be able to
see, at any instant, who the system was watching and which way it thought they
were looking -- including the long stretches where the answer is "everyone was
facing their own screen".

Where a sample rate below the source frame rate leaves gaps, the most recent
sample for each track is held on screen and drawn dimmed, so the difference
between "measured here" and "carried from an earlier measurement" stays visible
rather than being smoothed away.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, head_pose as hp  # noqa: E402

# COCO skeleton over the joints AlphaPose returns here.
BONES = (
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
)

# Only these can corroborate an event, so only these are drawn prominently.
TARGET_CLASSES = ("phone", "secondary_paper_chit",
                  "book_or_notebook_like_object",
                  # From `chit_detector`. Named for what was observed rather
                  # than for what it would mean if illicit -- the model cannot
                  # tell a chit from an answer sheet, so the label must not.
                  "paper_like_object")

FACING_COLOUR = {
    hp.FRONTAL: (90, 220, 90),
    hp.TURNED_LEFT: (60, 190, 255),
    hp.TURNED_RIGHT: (255, 170, 60),
    hp.DOWNWARD: (80, 235, 235),
    hp.NOT_RESOLVABLE: (150, 150, 150),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dim(colour, factor=0.45):
    return tuple(int(c * factor) for c in colour)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=0.0,
                    help="output frame rate; 0 uses the source rate")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--hold-ms", type=float, default=400.0,
                    help="how long a sample stays on screen after its "
                         "timestamp, drawn dimmed, before it is dropped")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--samples", default="samples.jsonl",
                    help="which samples file in 14_person_timeline to draw; "
                         "`attach_chit_detections.py` writes an enriched copy "
                         "beside the original rather than overwriting it")
    args = ap.parse_args()

    import av
    import cv2
    import numpy as np

    paths = artifacts.open_run(args.run.parent, args.run.name)
    samples = read_jsonl(paths.dir / "14_person_timeline" / args.samples)
    if not samples:
        print("no samples.jsonl; run tools/run_person_timeline.py first",
              file=sys.stderr)
        return 2

    profiles = {}
    profile_path = paths.dir / "14_person_timeline/profiles.json"
    if profile_path.is_file():
        for row in json.loads(profile_path.read_text(encoding="utf-8")):
            profiles[row["track_id"]] = row

    by_pts: dict[float, list] = {}
    for sample in samples:
        by_pts.setdefault(round(float(sample["pts_ms"]), 1), []).append(sample)
    stamps = sorted(by_pts)
    print(f"{len(samples)} person-samples over {len(stamps)} measured frames")

    out_path = args.out or (paths.dir / "14_person_timeline/annotated_video.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        # Run 1501 silently produced `annotated_video_1.mp4` beside an older
        # file, and the stale one was the one people opened. Refuse instead.
        print(f"{out_path} already exists; pass --out or remove it",
              file=sys.stderr)
        return 2

    with av.open(str(args.video)) as probe:
        stream = probe.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        source_fps = float(stream.average_rate or 25.0)
    fps = args.fps or source_fps

    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        print(f"could not open a writer for {out_path}", file=sys.stderr)
        return 2

    index = 0
    live: dict[int, tuple] = {}          # track_id -> (pts_ms, sample)
    written = 0
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts_ms = float(frame.pts * time_base) * 1000.0
            if args.max_seconds and pts_ms > args.max_seconds * 1000.0:
                break

            # Adopt every measurement at or before this frame.
            while index < len(stamps) and stamps[index] <= pts_ms + 1.0:
                for sample in by_pts[stamps[index]]:
                    live[sample["track_id"]] = (stamps[index], sample)
                index += 1
            # Drop measurements that have gone stale.
            for track_id in [t for t, (when, _s) in live.items()
                             if pts_ms - when > args.hold_ms]:
                live.pop(track_id, None)

            canvas = frame.to_ndarray(format="bgr24")
            for track_id, (when, sample) in sorted(live.items()):
                fresh = abs(pts_ms - when) < 1.0
                base = FACING_COLOUR.get(sample.get("facing", ""),
                                         (150, 150, 150))
                colour = base if fresh else dim(base)
                thickness = 2 if fresh else 1

                box = sample.get("box") or [0, 0, 0, 0]
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

                joints = sample.get("joints") or {}
                for a, b in BONES:
                    if a in joints and b in joints:
                        cv2.line(canvas,
                                 (int(joints[a][0]), int(joints[a][1])),
                                 (int(joints[b][0]), int(joints[b][1])),
                                 colour, thickness)
                for name, value in joints.items():
                    cv2.circle(canvas, (int(value[0]), int(value[1])),
                               2 if fresh else 1, colour, -1)

                label = f"#{track_id} {sample.get('facing', '')[:11]}"
                if sample.get("yaw") is not None:
                    label += f" y{sample['yaw']:+.2f}"
                cv2.putText(canvas, label, (x1, max(y1 - 4, 11)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)

                # Gaze vector from the head, in the direction yaw reports.
                if sample.get("yaw") is not None and "nose" in joints:
                    nx, ny = int(joints["nose"][0]), int(joints["nose"][1])
                    length = max((x2 - x1) * 0.5, 16.0)
                    cv2.arrowedLine(
                        canvas, (nx, ny),
                        (int(nx - sample["yaw"] * length),
                         int(ny + (sample.get("pitch") or 0.0) * length * 1.5)),
                        colour, thickness, tipLength=0.3)

                # Objects the detector put near this person's hands. Target
                # classes are drawn in red because they are the reportable
                # ones; everything else is dimmed, so a reviewer can see what
                # the detector is reacting to without the room's furniture
                # competing with the finding.
                for item in (sample.get("near_hand_objects") or []):
                    if not isinstance(item, dict) or not item.get("box"):
                        continue
                    ox1, oy1, ox2, oy2 = (int(v) for v in item["box"])
                    target = item["cls"] in TARGET_CLASSES
                    ocolour = (60, 60, 235) if target else (110, 110, 110)
                    cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), ocolour,
                                  2 if target else 1)
                    if target:
                        cv2.putText(
                            canvas,
                            f"{item['cls']} {item.get('confidence', 0):.2f}",
                            (ox1, max(oy1 - 3, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, ocolour, 1)

                # A person who has departed far from their own resting posture
                # for much of the recording is worth the reviewer's eye.
                profile = profiles.get(track_id)
                if profile and profile.get("fraction_away_from_baseline", 0) > 0.4 \
                        and fresh:
                    cv2.putText(
                        canvas,
                        f"away {profile['fraction_away_from_baseline']:.0%}",
                        (x1, min(y2 + 13, height - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 165, 255), 1)

            banner = (f"{pts_ms / 1000:7.2f}s   {len(live)} tracked   "
                      f"green frontal / blue turned / yellow down / grey "
                      f"unresolved   RED = target object near hands")
            cv2.rectangle(canvas, (0, height - 22), (width, height), (0, 0, 0), -1)
            cv2.putText(canvas, banner, (6, height - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            writer.write(canvas)
            written += 1
            if written % 200 == 0:
                print(f"  {written} frames written ({pts_ms / 1000:.1f}s)",
                      flush=True)

    writer.release()
    size = out_path.stat().st_size / 1048576 if out_path.is_file() else 0.0
    print(f"\n{written} frames at {fps:.1f} fps -> {out_path}  ({size:.1f} MB)")
    print("Dimmed skeletons are the most recent measurement being held on "
          "screen between samples, not a fresh one. A person drawn brightly "
          "was measured in that exact frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
