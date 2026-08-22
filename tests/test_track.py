"""Tracking gate: seat reconciliation must survive tracker identity switches.

The failure this guards against is the worst one the system can produce --
evidence from one candidate attributed to their neighbour. Reconciliation is
therefore tested against exactly that: a track that drifts across a seat
boundary, and a track that never settles anywhere.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from classroom import calibration as cal_mod, db, track  # noqa: E402
from classroom.calibration import Calibration, Seat  # noqa: E402
from classroom.track import TrackObservation  # noqa: E402

W, H = 1280, 720
SEATS = {"11": (100, 300, 400, 650), "12": (500, 300, 800, 650),
         "13": (900, 300, 1200, 650)}


def rect(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def build_cal() -> Calibration:
    return Calibration(
        camera_id=1, frame_w=W, frame_h=H,
        seats=[Seat(label, rect(box)) for label, box in SEATS.items()],
    )


def box_in(label: str, dy: float = 0.0) -> tuple[float, float, float, float]:
    """A person-shaped box whose anchor lands inside the given seat."""
    x1, y1, x2, y2 = SEATS[label]
    cx = (x1 + x2) / 2
    return (cx - 60, y1 + 20 + dy, cx + 60, y2 - 30 + dy)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    ok = True
    cal = build_cal()

    print("anchor placement")
    ok &= check("anchor sits low in the box, at the chair",
                track.anchor_point((0, 0, 100, 100))[1] > 80,
                f"y={track.anchor_point((0, 0, 100, 100))[1]:.0f}")
    for label in SEATS:
        ok &= check(f"box in seat {label} resolves to {label}",
                    track.seat_for_box(box_in(label), cal) == label)
    ok &= check("box on empty floor resolves to no seat",
                track.seat_for_box((420, 100, 480, 200), cal) is None)

    print("\npolygon containment")
    poly = rect((0, 0, 100, 100))
    ok &= check("inside", track.point_in_polygon((50, 50), poly))
    ok &= check("outside", not track.point_in_polygon((150, 50), poly))
    ok &= check("far outside", not track.point_in_polygon((-10, -10), poly))

    print("\nreconciliation: a stable track gets its seat")
    stable = [TrackObservation(1, i * 0.2, box_in("12")) for i in range(10)]
    summaries = track.reconcile(stable, cal)
    ok &= check("seat assigned", summaries[1].seat_label == "12")
    ok &= check("full confidence", abs(summaries[1].confidence - 1.0) < 1e-9)
    ok &= check("not ambiguous", not summaries[1].ambiguous)
    ok &= check("counted as reconciled", summaries[1].reconciled)

    print("\nreconciliation: a majority wins over transient drift")
    # 8 frames in seat 12, 2 frames drifting into 13.
    drifting = ([TrackObservation(2, i * 0.2, box_in("12")) for i in range(8)]
                + [TrackObservation(2, 1.6 + i * 0.2, box_in("13")) for i in range(2)])
    summaries = track.reconcile(drifting, cal)
    ok &= check("majority seat wins", summaries[2].seat_label == "12")
    ok &= check("confidence reflects the drift",
                abs(summaries[2].confidence - 0.8) < 1e-9,
                f"{summaries[2].confidence:.2f}")
    ok &= check("both seats recorded in the vote",
                summaries[2].seat_votes == {"12": 8, "13": 2},
                str(summaries[2].seat_votes))

    print("\nreconciliation: a split track is ambiguous, never guessed")
    split = ([TrackObservation(3, i * 0.2, box_in("11")) for i in range(5)]
             + [TrackObservation(3, 1.0 + i * 0.2, box_in("12")) for i in range(5)])
    summaries = track.reconcile(split, cal)
    ok &= check("no seat assigned on a 50/50 split",
                summaries[3].seat_label is None)
    ok &= check("flagged ambiguous", summaries[3].ambiguous)
    ok &= check("not counted as reconciled", not summaries[3].reconciled)
    ok &= check("the evidence for the split is retained",
                summaries[3].seat_votes == {"11": 5, "12": 5},
                str(summaries[3].seat_votes))

    print("\nreconciliation: a track never in any seat")
    floating = [TrackObservation(4, i * 0.2, (420, 100, 480, 200)) for i in range(6)]
    summaries = track.reconcile(floating, cal)
    ok &= check("no seat", summaries[4].seat_label is None)
    ok &= check("ambiguous", summaries[4].ambiguous)
    ok &= check("frames still counted", summaries[4].frames == 6)

    print("\nreconciliation: independent tracks do not interfere")
    mixed = ([TrackObservation(5, i * 0.2, box_in("11")) for i in range(6)]
             + [TrackObservation(6, i * 0.2, box_in("13")) for i in range(6)])
    summaries = track.reconcile(mixed, cal)
    ok &= check("two tracks summarised", len(summaries) == 2)
    ok &= check("track 5 -> seat 11", summaries[5].seat_label == "11")
    ok &= check("track 6 -> seat 13", summaries[6].seat_label == "13")

    print("\nboxes derived from pose keypoints")
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = (600.0, 340.0, 0.9)
    kp[6] = (700.0, 345.0, 0.9)
    kp[11] = (610.0, 560.0, 0.8)
    kp[12] = (690.0, 565.0, 0.8)
    boxes, scores = track.boxes_from_poses([kp])
    ok &= check("one box derived", len(boxes) == 1)
    ok &= check("box bounds the visible keypoints",
                boxes[0][0] == 600.0 and boxes[0][3] == 565.0, str(boxes[0]))
    ok &= check("derived box lands in seat 12",
                track.seat_for_box(tuple(boxes[0]), cal) == "12")

    sparse = np.zeros((17, 3), dtype=np.float32)
    sparse[0] = (100.0, 100.0, 0.9)
    ok &= check("a single keypoint yields no box",
                len(track.boxes_from_poses([sparse])[0]) == 0)
    ok &= check("no people yields empty arrays",
                len(track.boxes_from_poses([])[0]) == 0)

    print("\npersistence")
    with tempfile.TemporaryDirectory() as tmp:
        conn = db.connect(Path(tmp) / "t.db")
        session_id = db.insert(conn, "session", name="fixture")
        camera_id = db.insert(conn, "camera", session_id=session_id, label="cam")
        cal.camera_id = camera_id
        cal_id = cal_mod.save(conn, cal)
        seat_ids = cal_mod.seat_ids(conn, cal_id)
        video_id = db.insert(
            conn, "source_video", camera_id=camera_id, path="x", sha256="z" * 64,
            size_bytes=1, container="mp4", codec="h264", pix_fmt="yuv420p",
            width=W, height=H, avg_fps=25.0, duration_s=10.0, frame_count=250,
            is_vfr=0, has_mv=1,
        )
        summaries = track.reconcile(stable + split, cal)
        written = track.persist(conn, video_id, summaries, seat_ids)
        ok &= check("tracks written", written == 2, str(written))

        rows = db.all_rows(conn, "SELECT * FROM track WHERE source_video_id = ?",
                           (video_id,))
        by_tracker = {r["tracker_id"]: r for r in rows}
        ok &= check("reconciled track keeps its seat",
                    by_tracker[1]["seat_id"] == seat_ids["12"])
        ok &= check("ambiguous track stores no seat",
                    by_tracker[3]["seat_id"] is None)
        ok &= check("ambiguous flag persisted", by_tracker[3]["ambiguous"] == 1)
        again = track.persist(conn, video_id, summaries, seat_ids)
        ok &= check("rewrite is idempotent",
                    len(db.all_rows(conn, "SELECT * FROM track WHERE source_video_id = ?",
                                    (video_id,))) == again)
        conn.close()

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
