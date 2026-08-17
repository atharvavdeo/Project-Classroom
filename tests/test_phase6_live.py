"""Phase 6 orchestration, executed for real.

Runs evidence.run() over the demo database with a live pose model and tracker.
Everything else in the suite tests components; this proves the orchestration
actually decodes, crops, infers, reconciles and persists without a detector
present -- which is the configuration the project is in until D-FINE weights
arrive.

Uses synthetic footage, so zero people is the expected and correct outcome. What
is being verified is that the pipeline completes, records the absence honestly,
and leaves scoring able to tell "no detector ran" from "nothing was found".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import calibration as cal_mod, db, evidence, pose, scoring, track  # noqa: E402
from classroom.config import Config  # noqa: E402

DB = Path(__file__).resolve().parents[1] / "data" / "classroom.db"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    if not DB.is_file():
        print(f"demo database missing: {DB}\nrun: python tests/build_demo.py")
        return 1

    ok = True
    cfg = Config(db_path=DB)
    conn = db.connect(DB)

    video = db.one(conn, "SELECT * FROM source_video ORDER BY id LIMIT 1")
    cal = cal_mod.latest_for_camera(conn, video["camera_id"])
    events = db.all_rows(
        conn, "SELECT COUNT(*) c FROM event WHERE source_video_id = ?", (video["id"],)
    )[0]["c"]
    print(f"video {video['id']}: {video['duration_s']:.0f}s, {events} events, "
          f"{len(cal.seats)} seats")

    print("\nloading pose model (top-down, CPU)")
    t0 = time.perf_counter()
    pose_model = pose.PoseEstimator(device="cpu")
    pose_model.load()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    tracker = track.SeatTracker(cal, frame_rate=int(video["avg_fps"]))

    print("\nrunning phase 6 over candidate events")
    t0 = time.perf_counter()
    results = evidence.run(
        conn, video["id"], cal, cfg.detect,
        detector=None, pose_model=pose_model, tracker=tracker,
    )
    elapsed = time.perf_counter() - t0
    print(f"  {len(results)} events in {elapsed:.1f}s")

    ok &= check("every event processed", len(results) == events, f"{len(results)}")
    ok &= check("pose stream reported present",
                all("pose" in r.streams_present for r in results))
    ok &= check("object stream reported absent, not zero",
                all(r.object_evidence is None for r in results))
    ok &= check("pitch evidence computed",
                all(r.pitch_evidence is not None for r in results))

    print("\ntrack reconciliation")
    summaries = tracker.summarize()
    seat_ids = cal_mod.seat_ids(
        conn,
        db.one(conn, "SELECT id FROM calibration WHERE camera_id = ? "
                     "ORDER BY version DESC LIMIT 1", (video["camera_id"],))["id"],
    )
    written = track.persist(conn, video["id"], summaries, seat_ids)
    print(f"  {len(summaries)} tracks, {written} persisted, "
          f"{sum(1 for s in summaries.values() if s.ambiguous)} ambiguous")
    ok &= check("track table writable", written == len(summaries))

    print("\nrescoring with model evidence in place")
    scoring.score_video(conn, video["id"], cfg.score)
    rows = db.all_rows(
        conn,
        """
        SELECT e.id, s.seat_label, e.score, e.evidence_quality, e.score_factors
        FROM event e JOIN seat s ON s.id = e.seat_id
        WHERE e.source_video_id = ? ORDER BY e.score DESC
        """,
        (video["id"],),
    )
    for r in rows:
        print(f"    event {r['id']} seat {r['seat_label']}: score={r['score']:.2f} "
              f"quality={r['evidence_quality']}")
    ok &= check("all events scored", all(r["score"] is not None for r in rows))
    ok &= check("evidence quality stays limited without a detector",
                all(r["evidence_quality"] == "limited" for r in rows))

    print("\nidempotency")
    before = db.all_rows(
        conn, "SELECT COUNT(*) c FROM pose_observation")[0]["c"]
    evidence.run(conn, video["id"], cal, cfg.detect,
                 detector=None, pose_model=pose_model, tracker=tracker, verbose=False)
    after = db.all_rows(conn, "SELECT COUNT(*) c FROM pose_observation")[0]["c"]
    ok &= check("rerun does not duplicate observations", before == after,
                f"{before} -> {after}")

    conn.close()
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
