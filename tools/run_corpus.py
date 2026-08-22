"""Run every phase over the real corpus.

Ingests each source file, applies a per-camera calibration, runs the motion
scan, segmentation, scoring and clip extraction, then reports throughput and
the candidate fraction actually reaching the expensive stages.

Seat geometry is supplied per camera; files without one get a single
whole-frame region so the motion layer still runs and reports honestly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import calibration as cal_mod, clips, db, ingest, pipeline, scoring
from classroom.calibration import Calibration, MaskRegion, Seat
from classroom.config import BaselineConfig, Config

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/DrishtiAI")
OUT = Path("data/corpus")

# Seat polygons traced from the frames of each camera view. The burned-in
# date/time sits top-left in every file in this corpus.
OVERLAY_720 = (8, 30, 520, 92)
OVERLAY_480 = (6, 20, 300, 60)

LAYOUTS = {
    # "paper" / Seat 12 view: near desks left, booth row right.
    "Seat No. 12": {
        "seats": {"L-front": (30, 300, 330, 620),
                  "C-front": (600, 230, 790, 560),
                  "R-cluster": (800, 60, 1010, 420)},
        "overlay": OVERLAY_720,
    },
    # Camera 12: single candidate at seat 61, desk row right.
    "03.CCTV": {
        "seats": {"S-61": (60, 280, 380, 700),
                  "R-desk": (700, 120, 1180, 480)},
        "overlay": OVERLAY_720,
    },
    # Camera 04: booth row with numbered placards.
    "01.Candidate": {
        "seats": {"B-18": (60, 380, 380, 700),
                  "B-20": (380, 300, 640, 660),
                  "B-mid": (700, 300, 1080, 700)},
        "overlay": OVERLAY_720,
    },
    "02.Candidate": {
        "seats": {"B-18": (60, 380, 380, 700),
                  "B-20": (380, 300, 640, 660),
                  "B-mid": (700, 300, 1080, 700)},
        "overlay": OVERLAY_720,
    },
    "05.Crowd": {
        "seats": {"Left": (20, 180, 420, 700),
                  "Centre": (460, 160, 840, 700),
                  "Right": (880, 140, 1260, 700)},
        "overlay": OVERLAY_720,
    },
    "04.CCTV": {
        "seats": {"Whole": (10, 60, 630, 470)},
        "overlay": OVERLAY_480,
    },
}


def rect(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def layout_for(name: str):
    for key, layout in LAYOUTS.items():
        if name.startswith(key):
            return layout
    return None


def main() -> int:
    files = sorted(list(CORPUS.glob("*.mkv")) + list(CORPUS.glob("*.mp4")))
    if not files:
        print(f"no source files under {CORPUS}")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    db_path = OUT / "corpus.db"
    if db_path.exists():
        db_path.unlink()
    cfg = Config(db_path=db_path, media_root=OUT / "media",
                 baseline=BaselineConfig(learn_s=1e9, min_windows=40))
    conn = db.connect(cfg.db_path)
    session_id = db.insert(conn, "session", name="real corpus", exam_date="2026-08-18")

    grand_events = 0
    grand_seconds = 0.0
    grand_scan = 0.0

    for path in files:
        layout = layout_for(path.name)
        if layout is None:
            print(f"\n{path.name}: no layout defined, skipped")
            continue

        print(f"\n{'=' * 78}\n{path.name}\n{'=' * 78}")
        profile = ingest.probe(path)
        video_id = ingest.register(conn, profile, session_id, path.stem[:24], "corpus")
        print(f"  {profile.codec} {profile.width}x{profile.height} "
              f"{profile.avg_fps:.1f}fps {profile.duration_s:.1f}s "
              f"{'VFR' if profile.is_vfr else 'CFR'}")

        cal = Calibration(
            camera_id=db.one(conn, "SELECT camera_id FROM source_video WHERE id = ?",
                             (video_id,))["camera_id"],
            frame_w=profile.width, frame_h=profile.height,
            seats=[Seat(label, rect(box), zones={"desk": rect(box)},
                        desk_line_y=box[3] - max((box[3] - box[1]) // 4, 20))
                   for label, box in layout["seats"].items()],
            masks=[MaskRegion("overlay", rect(layout["overlay"]), "burned-in date/time")],
        )
        cal_id = cal_mod.save(conn, cal)

        started = time.perf_counter()
        result = pipeline.run(conn, video_id, cal_id, cfg, verbose=False)
        elapsed = time.perf_counter() - started
        scoring.score_video(conn, video_id, cfg.score)
        made = clips.materialise(conn, video_id, cfg.media_root, cal, cfg.segment,
                                 limit=12, verbose=False)

        grand_events += result.events
        grand_seconds += profile.duration_s
        grand_scan += elapsed

        print(f"  {len(cal.seats)} seats, {result.windows} windows, "
              f"{elapsed:.1f}s ({profile.duration_s / elapsed:.0f}x realtime)")
        print(f"  {result.events} events, "
              f"{result.candidate_fraction * 100:.1f}% flagged, {made} clips")

        rows = db.all_rows(
            conn,
            """
            SELECT s.seat_label, e.t_start, e.t_end, e.peak_deviation, e.score
            FROM event e JOIN seat s ON s.id = e.seat_id
            WHERE e.source_video_id = ? ORDER BY e.score DESC LIMIT 6
            """,
            (video_id,),
        )
        for r in rows:
            print(f"    [{r['score']:5.2f}] {r['seat_label']:10s} "
                  f"{r['t_start']:7.2f}-{r['t_end']:7.2f}  z={r['peak_deviation']:6.2f}")

    print(f"\n{'=' * 78}")
    print(f"corpus: {grand_seconds:.0f}s of footage scanned in {grand_scan:.0f}s "
          f"({grand_seconds / grand_scan:.0f}x realtime)")
    print(f"{grand_events} events across {len(files)} files")
    print(f"extrapolated: 5 h of footage -> {5 * 3600 / (grand_seconds / grand_scan) / 60:.1f} min")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
