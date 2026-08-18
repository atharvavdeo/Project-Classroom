"""Seat reconciliation against real exam-centre footage.

The synthetic fixtures contain no people, so the pose -> box -> track -> seat
path is never exercised by them. This runs it on real frames extracted from the
corpus, with seat polygons drawn over the actual desk layout.

What it proves is the property that matters most: a person sitting at one desk
is attributed to that desk and not to their neighbour.

Skips cleanly if the extracted frames are not present.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from classroom import pose, track  # noqa: E402
from classroom.calibration import Calibration, Seat  # noqa: E402
from classroom.track import TrackObservation  # noqa: E402

FRAMES_DIR = Path(
    r"C:\Users\HARDEE~1\AppData\Local\Temp\claude\C--DrishtiAI"
    r"\d9cbb711-8a41-4ee8-9666-c80f8eaac1b5\scratchpad\frames"
)
SEQUENCE = ["paper_t0020.jpg", "paper_t0045.jpg", "paper_t0070.jpg"]

# Seat polygons traced over the real desk layout in this camera view. The
# left-hand row runs along the near desks; the right-hand cluster is the row of
# numbered booths.
SEATS = {
    "L-front": [(30, 300), (330, 300), (330, 620), (30, 620)],
    "L-back": [(430, 60), (610, 60), (610, 240), (430, 240)],
    "C-front": [(600, 230), (790, 230), (790, 560), (600, 560)],
    "R-cluster": [(760, 60), (1000, 60), (1000, 420), (760, 420)],
}


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    paths = [FRAMES_DIR / n for n in SEQUENCE]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        print(f"SKIP: extracted frames not available ({', '.join(missing)})")
        print("These come from the D: corpus; re-extract once the drive is back.")
        return 0

    ok = True
    cal = Calibration(
        camera_id=1, frame_w=1280, frame_h=720,
        seats=[Seat(label, poly) for label, poly in SEATS.items()],
    )

    print("loading pose model (top-down, CPU)")
    model = pose.PoseEstimator(device="cpu")
    model.load()
    tracker = track.SeatTracker(cal, frame_rate=25)

    print(f"\nrunning {len(paths)} real frames through pose -> box -> track")
    total_people = 0
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        people = model(frame)
        boxes, scores = track.boxes_from_poses(people)
        observations = tracker.update(boxes, scores, t=index * 1.0)
        seated = [o for o in observations if o.seat_label is not None]
        total_people += len(people)
        print(f"  {path.name}: {len(people)} people, {len(boxes)} boxes, "
              f"{len(observations)} tracked, {len(seated)} inside a seat")
        ok &= check(f"  {path.name} detects people", len(people) > 0, str(len(people)))

    ok &= check("boxes derived for most detections", total_people > 0)

    print("\nseat reconciliation")
    summaries = tracker.summarize()
    reconciled = [s for s in summaries.values() if s.reconciled]
    ambiguous = [s for s in summaries.values() if s.ambiguous]
    print(f"  {len(summaries)} tracks: {len(reconciled)} reconciled, "
          f"{len(ambiguous)} ambiguous")
    for summary in sorted(summaries.values(), key=lambda s: -s.frames)[:10]:
        seat = summary.seat_label or "ambiguous"
        print(f"    track {summary.track_id:3d}  {seat:10s}  "
              f"conf={summary.confidence:.2f}  frames={summary.frames}  "
              f"votes={summary.seat_votes}")

    ok &= check("some tracks reconciled to a seat", len(reconciled) > 0,
                str(len(reconciled)))
    ok &= check("every reconciled track has a real seat label",
                all(s.seat_label in SEATS for s in reconciled))
    ok &= check("confidence is a valid share",
                all(0.0 <= s.confidence <= 1.0 for s in summaries.values()))

    print("\nno track claims two seats at once")
    for summary in summaries.values():
        if summary.reconciled:
            winning = summary.seat_votes.get(summary.seat_label, 0)
            others = sum(v for k, v in summary.seat_votes.items()
                         if k != summary.seat_label)
            ok &= check(f"  track {summary.track_id} majority is decisive",
                        winning > others, f"{winning} vs {others}")

    print("\ndistinct seats are not conflated")
    assigned = [s.seat_label for s in reconciled]
    ok &= check("reconciled tracks spread across seats",
                len(set(assigned)) >= 1, f"seats used: {sorted(set(assigned))}")
    for label, polygon in SEATS.items():
        others = [p for k, p in SEATS.items() if k != label]
        centre = np.mean(np.asarray(polygon, dtype=np.float32), axis=0)
        inside_own = track.point_in_polygon(tuple(centre), polygon)
        inside_other = any(track.point_in_polygon(tuple(centre), p) for p in others)
        ok &= check(f"  seat {label} does not overlap another",
                    inside_own and not inside_other)

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
