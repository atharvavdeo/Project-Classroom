"""Build a populated demo database for the console.

Produces a 90-second recording with planted events, runs the full pipeline,
scores the events and materialises clips and thumbnails.

    python tests/build_demo.py
    uvicorn classroom.api:app
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av  # noqa: E402
import numpy as np  # noqa: E402

from classroom import calibration as cal_mod, clips, db, ingest, pipeline, scoring  # noqa: E402
from classroom.calibration import Calibration, MaskRegion, Seat  # noqa: E402
from classroom.config import BaselineConfig, Config  # noqa: E402

W, H, FPS, SECONDS = 640, 480, 25, 90.0
OVERLAY = (10, 10, 210, 60)
SEATS = {"12": (40, 190, 260, 330), "13": (360, 190, 580, 330)}
PLANTED = [("12", 14.0, 19.0), ("13", 33.0, 36.5), ("12", 52.0, 58.0), ("13", 71.0, 74.0)]

ROOT = Path(__file__).resolve().parents[1] / "data"


def rect(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_recording(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)
    ramp = np.linspace(40, 190, W, dtype=np.float32)
    bg = np.repeat(ramp[None, :], H, axis=0) + np.linspace(0, 30, H, dtype=np.float32)[:, None]
    base = np.dstack([np.clip(bg, 0, 255).astype(np.uint8)] * 3)
    for box in SEATS.values():
        x1, y1, x2, y2 = box
        base[y1:y2, x1:x2] = (200, 200, 195)
        base[y1 + 10 : y1 + 40, x1 + 10 : x2 - 10] = (60, 60, 65)

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.gop_size = FPS
        for i in range(int(SECONDS * FPS)):
            t = i / FPS
            img = base.copy()
            ox1, oy1, ox2, oy2 = OVERLAY
            img[oy1:oy2, ox1:ox2] = rng.integers(0, 255, (oy2 - oy1, ox2 - ox1, 3), np.uint8)
            for seat, start, end in PLANTED:
                if start <= t < end:
                    x1, y1, x2, y2 = SEATS[seat]
                    span = (t - start) / max(end - start, 1e-6)
                    bw, bh = 46, 34
                    by = int(y1 + 45 + span * (y2 - y1 - bh - 50))
                    img[by : by + bh, x1 + 30 : x1 + 30 + bw] = (30, 30, 35)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    db_path = ROOT / "classroom.db"
    if db_path.exists():
        db_path.unlink()

    print(f"building {SECONDS:.0f}s recording with {len(PLANTED)} planted events")
    video = make_recording(ROOT / "raw" / "demo_camera12.mp4")

    cfg = Config(db_path=db_path, media_root=ROOT / "media",
                 baseline=BaselineConfig(learn_s=SECONDS, min_windows=40))
    conn = db.connect(cfg.db_path)

    print("phase 1: ingest")
    profile = ingest.probe(video)
    session_id = db.insert(conn, "session", name="CBT centre demo", exam_date="2026-08-18")
    video_id = ingest.register(conn, profile, session_id, "Camera12", "room-a")

    print("phase 2: calibration")
    cal = Calibration(
        camera_id=db.one(conn, "SELECT camera_id FROM source_video WHERE id = ?",
                         (video_id,))["camera_id"],
        frame_w=W, frame_h=H,
        seats=[Seat(label, rect(box), zones={"desk": rect(box)}, desk_line_y=box[3] - 40)
               for label, box in SEATS.items()],
        masks=[MaskRegion("overlay", rect(OVERLAY), "burned-in timestamp")],
    )
    cal_id = cal_mod.save(conn, cal)

    print("phase 3+4: scan and segment")
    result = pipeline.run(conn, video_id, cal_id, cfg)
    print(f"  {result.events} events, {result.windows} windows in {result.seconds:.1f}s")

    print("scoring")
    scored = scoring.score_video(conn, video_id, cfg.score)
    print(f"  {scored} events scored")

    print("clips and thumbnails")
    made = clips.materialise(conn, video_id, cfg.media_root, cal, cfg.segment)
    print(f"  {made} clips written")

    rows = db.all_rows(
        conn,
        """
        SELECT e.id, s.seat_label, e.t_start, e.t_end, e.score, e.explanation
        FROM event e JOIN seat s ON s.id = e.seat_id
        WHERE e.source_video_id = ? ORDER BY e.score DESC
        """,
        (video_id,),
    )
    print(f"\n{'id':>3}  {'seat':5}  {'start':>7}  {'dur':>6}  {'score':>6}  explanation")
    for r in rows:
        print(f"{r['id']:>3}  {r['seat_label']:5}  {r['t_start']:>7.2f}  "
              f"{r['t_end'] - r['t_start']:>6.2f}  {r['score']:>6.2f}  {r['explanation']}")

    conn.close()
    print(f"\ndatabase: {db_path}\nstart the console:  uvicorn classroom.api:app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
