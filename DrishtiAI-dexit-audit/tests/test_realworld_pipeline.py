"""The whole pipeline, on real people, with known ground truth.

Synthesises a long recording from the real corpus frames (PRD 17.1), then runs
every phase over it: ingest, calibration, motion scan, segmentation, scoring,
clips, pose evidence and seat reconciliation. Recall is measured against the
insertion timestamps.

The pixels, the people, the camera geometry and the compression are all real.
Only the timeline is constructed -- which is the point, because the corpus has
no negative time to measure against.

Also calibrates the per-camera head-pitch baseline from real observations,
which cannot be derived from synthetic fixtures.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from classroom import (  # noqa: E402
    calibration as cal_mod, clips, db, evidence, harness, ingest, pipeline,
    pose, scoring, track,
)
from classroom.calibration import Calibration, MaskRegion, Seat  # noqa: E402
from classroom.config import BaselineConfig, Config  # noqa: E402
from classroom.harness import PlantedEvent  # noqa: E402

FRAMES_DIR = Path(
    r"C:\Users\HARDEE~1\AppData\Local\Temp\claude\C--DrishtiAI"
    r"\d9cbb711-8a41-4ee8-9666-c80f8eaac1b5\scratchpad\frames"
)
BASE = "paper_t0020.jpg"
ALTERNATES = ["paper_t0045.jpg", "paper_t0070.jpg"]
OUT = Path("data/realworld")
DURATION = 150.0

# Traced over the real desk layout in this camera view.
SEATS = {
    "L-front": (30, 300, 330, 620),
    "C-front": (600, 230, 790, 560),
    "R-cluster": (800, 60, 1010, 420),
}
# The burned-in date/time, top-left in every file in this corpus.
OVERLAY = (8, 40, 240, 76)

PLANTED = [
    PlantedEvent("L-front", SEATS["L-front"], 22.0, 28.0, 0),
    PlantedEvent("C-front", SEATS["C-front"], 48.0, 53.0, 1),
    PlantedEvent("R-cluster", SEATS["R-cluster"], 76.0, 82.0, 0),
    PlantedEvent("L-front", SEATS["L-front"], 104.0, 111.0, 1),
    PlantedEvent("C-front", SEATS["C-front"], 128.0, 133.0, 0),
]


def rect(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    paths = [FRAMES_DIR / BASE] + [FRAMES_DIR / a for a in ALTERNATES]
    if any(not p.is_file() for p in paths):
        print("SKIP: real corpus frames not available")
        return 0

    ok = True
    OUT.mkdir(parents=True, exist_ok=True)
    base = cv2.imread(str(paths[0]))
    alternates = [cv2.imread(str(p)) for p in paths[1:]]
    print(f"source frames {base.shape[1]}x{base.shape[0]}, "
          f"{len(alternates)} alternates")

    print(f"\nsynthesising a {DURATION:.0f}s recording with {len(PLANTED)} planted events")
    t0 = time.perf_counter()
    video = harness.build(OUT / "synth_camera.mp4", base, alternates, PLANTED,
                          DURATION, fps=25)
    size_mb = video.stat().st_size / 1e6
    print(f"  {video} ({size_mb:.1f} MB, {time.perf_counter() - t0:.0f}s)")

    db_path = OUT / "realworld.db"
    if db_path.exists():
        db_path.unlink()
    cfg = Config(db_path=db_path, media_root=OUT / "media",
                 baseline=BaselineConfig(learn_s=DURATION, min_windows=60))
    conn = db.connect(cfg.db_path)

    print("\nphase 1: ingest")
    profile = ingest.probe(video)
    print(f"  {profile.codec} {profile.width}x{profile.height} "
          f"{profile.avg_fps:.1f}fps {profile.duration_s:.1f}s "
          f"{profile.bit_rate_mbps:.2f}Mbps {profile.mv_per_frame:.0f}mv/frame")
    session_id = db.insert(conn, "session", name="realworld harness")
    video_id = ingest.register(conn, profile, session_id, "Camera-AHMD3", "room-a")
    ok &= check("720p, matching the corpus", (profile.width, profile.height) == (1280, 720))
    ok &= check("motion vectors present", profile.has_mv)
    ok &= check("bitrate representative of the corpus",
                1.0 < profile.bit_rate_mbps < 4.0, f"{profile.bit_rate_mbps:.2f} Mbps")

    print("\nphase 2: calibration with real seat geometry")
    cal = Calibration(
        camera_id=db.one(conn, "SELECT camera_id FROM source_video WHERE id = ?",
                         (video_id,))["camera_id"],
        frame_w=1280, frame_h=720,
        seats=[Seat(label, rect(box), zones={"desk": rect(box)},
                    desk_line_y=box[3] - 90)
               for label, box in SEATS.items()],
        masks=[MaskRegion("overlay", rect(OVERLAY), "burned-in date/time")],
    )
    cal_id = cal_mod.save(conn, cal)
    masks = cal_mod.rasterize(cal)
    print(f"  {len(cal.seats)} seats, {int(masks.excluded.sum())} blocks excluded")
    ok &= check("overlay region excluded", int(masks.excluded.sum()) > 0)

    print("\nphase 3+4: motion scan and segmentation")
    result = pipeline.run(conn, video_id, cal_id, cfg, verbose=False)
    print(f"  {result.windows} windows in {result.seconds:.1f}s "
          f"({DURATION / result.seconds:.0f}x realtime)")
    print(f"  {result.events} events, "
          f"{result.candidate_fraction * 100:.1f}% of the recording flagged")

    rows = db.all_rows(
        conn,
        """
        SELECT s.seat_label, e.t_start, e.t_end, e.peak_deviation
        FROM event e JOIN seat s ON s.id = e.seat_id
        WHERE e.source_video_id = ? ORDER BY e.t_start
        """,
        (video_id,),
    )
    print(f"\n  detected {len(rows)} events:")
    for r in rows:
        print(f"    {r['seat_label']:10s} {r['t_start']:7.2f} - {r['t_end']:7.2f}  "
              f"peak z={r['peak_deviation']:.2f}")

    print("\n  recall against insertion timestamps")
    detected = [(r["seat_label"], r["t_start"], r["t_end"]) for r in rows]
    ranks = [r["peak_deviation"] for r in rows]
    metrics = harness.score_recall(detected, harness.ground_truth(PLANTED),
                                   ranks=ranks, duration_s=DURATION)
    for key, value in metrics.items():
        print(f"    {key:20s} {value:.3f}" if isinstance(value, float)
              else f"    {key:20s} {value}")

    ok &= check("all planted events recalled",
                metrics["recall"] == 1.0, f"{metrics['matched']}/{metrics['planted']}")
    # The system ranks rather than eliminates, so what matters is not whether a
    # marginal detection exists but whether it outranks a real one.
    ok &= check("top-K contains only real events",
                metrics["precision_at_k"] == 1.0,
                f"precision@{metrics['k']} = {metrics['precision_at_k']:.2f}")
    ok &= check("any false event ranks below every real one",
                max((ranks[i] for i, d in enumerate(detected)
                     if not any(d[0] == t["seat_label"]
                                and d[1] < t["t_end"] + 2 and d[2] > t["t_start"] - 2
                                for t in harness.ground_truth(PLANTED))),
                    default=0.0)
                < min((ranks[i] for i, d in enumerate(detected)
                       if any(d[0] == t["seat_label"]
                              and d[1] < t["t_end"] + 2 and d[2] > t["t_start"] - 2
                              for t in harness.ground_truth(PLANTED))),
                      default=0.0))
    ok &= check("temporal IoU is usable", metrics["mean_temporal_iou"] > 0.4,
                f"{metrics['mean_temporal_iou']:.2f}")
    ok &= check("no fragmentation: one detection per real event",
                abs(metrics["fragmentation"] - 1.0) < 1e-9,
                f"{metrics['fragmentation']:.2f}")
    ok &= check("cascade gates most of the recording",
                result.candidate_fraction < 0.5,
                f"{result.candidate_fraction * 100:.1f}%")

    print("\nscoring and evidence clips")
    scoring.score_video(conn, video_id, cfg.score)
    made = clips.materialise(conn, video_id, cfg.media_root, cal, cfg.segment,
                             verbose=False)
    ok &= check("clips written for every event", made == len(rows), str(made))

    print("\nphase 6: pose evidence and seat reconciliation on real people")
    pose_model = pose.PoseEstimator(device="cpu")
    pose_model.load()
    tracker = track.SeatTracker(cal, frame_rate=25)
    t0 = time.perf_counter()
    results = evidence.run(conn, video_id, cal, cfg.detect,
                           detector=None, pose_model=pose_model, tracker=tracker,
                           verbose=False)
    total_poses = sum(len(r.poses) for r in results)
    print(f"  {len(results)} events, {total_poses} pose observations "
          f"in {time.perf_counter() - t0:.0f}s")
    ok &= check("pose observations captured on real people", total_poses > 0,
                str(total_poses))

    summaries = tracker.summarize()
    reconciled = [s for s in summaries.values() if s.reconciled]
    seat_ids = cal_mod.seat_ids(conn, cal_id)
    track.persist(conn, video_id, summaries, seat_ids)
    print(f"  {len(summaries)} tracks, {len(reconciled)} reconciled to seats")
    ok &= check("tracks reconciled to real seats", len(reconciled) > 0,
                str(len(reconciled)))

    print("\ncalibrating the per-camera head-pitch baseline (task #11)")
    observations = [o for r in results for o in r.poses]
    baseline = pose.calibrate_pitch_baseline(observations)
    resolved = [o.head_pitch for o in observations if o.head_pitch is not None]
    if baseline is None:
        print(f"  only {len(resolved)} resolvable observations; "
              f"need 20 for a trustworthy baseline")
        ok &= check("enough observations to calibrate", False, str(len(resolved)))
    else:
        print(f"  {len(resolved)} resolvable pitches: "
              f"min {min(resolved):+.2f}  median {np.median(resolved):+.2f}  "
              f"max {max(resolved):+.2f}")
        print(f"  calibrated threshold (90th percentile): {baseline:+.3f}")
        print(f"  library default was {pose.DEFAULT_PITCH_BASELINE:+.3f}")
        ok &= check("calibrated baseline derived from real footage",
                    isinstance(baseline, float))
        ok &= check("calibration differs from the default, as predicted",
                    abs(baseline - pose.DEFAULT_PITCH_BASELINE) > 0.05,
                    f"{baseline:+.3f} vs {pose.DEFAULT_PITCH_BASELINE:+.3f}")
        with_default = pose.pitch_evidence(observations)
        with_calibrated = pose.pitch_evidence(observations, baseline=baseline)
        print(f"  pitch evidence: default={with_default:.2f} "
              f"calibrated={with_calibrated:.2f}")
        ok &= check("calibrated evidence is a valid fraction",
                    0.0 <= with_calibrated <= 1.0)

    print("\nreview surface")
    scored = db.all_rows(
        conn,
        """
        SELECT e.id, s.seat_label, e.score, e.evidence_quality, e.explanation
        FROM event e JOIN seat s ON s.id = e.seat_id
        WHERE e.source_video_id = ? ORDER BY e.score DESC
        """,
        (video_id,),
    )
    for r in scored:
        print(f"    [{r['score']:5.2f}] {r['seat_label']:10s} "
              f"{r['evidence_quality']:10s} {r['explanation']}")
    ok &= check("every event carries an observational explanation",
                all(r["explanation"] for r in scored))
    ok &= check("no explanation states a verdict",
                not any(w in (r["explanation"] or "").lower()
                        for r in scored
                        for w in ("cheat", "guilty", "malpractice", "misconduct")))

    conn.close()
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
