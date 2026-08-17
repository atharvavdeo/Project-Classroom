"""Integration gate: ingest -> calibration -> motion -> segmentation.

Builds a 60-second recording with three planted events at known timestamps and
checks the pipeline recovers them. This is the harness described in PRD 17.1 --
the mechanism that lets event recall be measured before continuous footage from
the real centre is available.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av  # noqa: E402
import numpy as np  # noqa: E402

from classroom import calibration as cal_mod, db, ingest, pipeline  # noqa: E402
from classroom.calibration import Calibration, MaskRegion, Seat  # noqa: E402
from classroom.config import BaselineConfig, Config  # noqa: E402

W, H, FPS = 640, 480, 25
SECONDS = 60.0
OVERLAY = (10, 10, 210, 60)
SEATS = {
    "A": (40, 190, 260, 330),
    "B": (360, 190, 580, 330),
}
# (seat, start, end)
PLANTED = [("A", 12.0, 16.0), ("B", 28.0, 31.0), ("A", 44.0, 49.0)]


def rect(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_recording(path: Path) -> Path:
    rng = np.random.default_rng(11)
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
            img[oy1:oy2, ox1:ox2] = rng.integers(
                0, 255, (oy2 - oy1, ox2 - ox1, 3), dtype=np.uint8
            )
            for seat, start, end in PLANTED:
                if start <= t < end:
                    x1, y1, x2, y2 = SEATS[seat]
                    span = (t - start) / max(end - start, 1e-6)
                    bw, bh = 46, 34
                    bx = x1 + 30
                    by = int(y1 + 45 + span * (y2 - y1 - bh - 50))
                    img[by : by + bh, bx : bx + bw] = (30, 30, 35)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def build_calibration() -> Calibration:
    return Calibration(
        camera_id=1, frame_w=W, frame_h=H,
        seats=[
            Seat(label, rect(box), zones={"desk": rect(box)}, desk_line_y=box[3] - 40)
            for label, box in SEATS.items()
        ],
        masks=[MaskRegion("overlay", rect(OVERLAY), "burned-in timestamp")],
    )


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print(f"building a {SECONDS:.0f}s recording with {len(PLANTED)} planted events")
        video = make_recording(tmp / "recording.mp4")

        cfg = Config(
            db_path=tmp / "e2e.db",
            baseline=BaselineConfig(learn_s=SECONDS, min_windows=40),
        )
        conn = db.connect(cfg.db_path)

        print("\nphase 1: ingest")
        profile = ingest.probe(video)
        session_id = db.insert(conn, "session", name="e2e")
        video_id = ingest.register(conn, profile, session_id, "cam-e2e", "room-1")
        ok &= check("video registered", video_id > 0)
        ok &= check("duration correct", abs(profile.duration_s - SECONDS) < 0.1,
                    f"{profile.duration_s:.2f}s")
        ok &= check("motion vectors available", profile.has_mv,
                    f"{profile.mv_per_frame:.0f}/frame")

        print("\nphase 2: calibration")
        cal = build_calibration()
        cal.camera_id = db.one(
            conn, "SELECT camera_id FROM source_video WHERE id = ?", (video_id,)
        )["camera_id"]
        cal_id = cal_mod.save(conn, cal)
        ok &= check("calibration saved", cal_id > 0)

        print("\nphase 3+4: scan and segment")
        result = pipeline.run(conn, video_id, cal_id, cfg, verbose=True)
        ok &= check("job completed", result.events >= 0)
        print(f"    {result.windows} windows in {result.seconds:.1f}s "
              f"({SECONDS / result.seconds:.1f}x realtime)")

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
            print(f"    seat {r['seat_label']}  {r['t_start']:6.2f} - {r['t_end']:6.2f}"
                  f"   peak z={r['peak_deviation']:.2f}")

        print("\n  recall against planted ground truth:")
        matched = 0
        for seat, start, end in PLANTED:
            hit = next(
                (r for r in rows
                 if r["seat_label"] == seat
                 and r["t_start"] < end + 1.5 and r["t_end"] > start - 1.5),
                None,
            )
            matched += hit is not None
            detail = (f"detected {hit['t_start']:.2f}-{hit['t_end']:.2f}"
                      if hit else "MISSED")
            ok &= check(f"seat {seat} event at {start:.0f}-{end:.0f}s", hit is not None,
                        detail)
        ok &= check("event recall is 100%", matched == len(PLANTED),
                    f"{matched}/{len(PLANTED)}")

        print("\n  precision: no events outside the planted spans")
        spurious = [
            r for r in rows
            if not any(
                r["seat_label"] == seat and r["t_start"] < end + 1.5 and r["t_end"] > start - 1.5
                for seat, start, end in PLANTED
            )
        ]
        ok &= check("no false events", len(spurious) == 0,
                    f"{len(spurious)} spurious")
        false_per_hour = len(spurious) / (SECONDS / 3600.0)
        print(f"    false events/hour: {false_per_hour:.1f}")

        print("\n  fragmentation")
        ok &= check("one event per planted action", len(rows) == len(PLANTED),
                    f"{len(rows)} events for {len(PLANTED)} actions")

        print("\n  candidate fraction reaching expensive stages")
        planted_fraction = sum(e - s for _, s, e in PLANTED) / SECONDS
        print(f"    {result.candidate_fraction * 100:.2f}% "
              f"(planted activity is {planted_fraction * 100:.2f}%)")
        ok &= check("cascade gates most of the recording",
                    result.candidate_fraction < 0.35,
                    f"{result.candidate_fraction * 100:.2f}%")

        print("\n  idempotency")
        again = pipeline.run(conn, video_id, cal_id, cfg, verbose=False)
        ok &= check("rerun yields identical event count",
                    again.events == result.events, f"{again.events} == {result.events}")

        print("\n  calibration/resolution mismatch is rejected")
        bad = build_calibration()
        bad.camera_id = cal.camera_id
        bad.frame_w, bad.frame_h = 1280, 720
        bad_id = cal_mod.save(conn, bad)
        try:
            pipeline.run(conn, video_id, bad_id, cfg, verbose=False)
            ok &= check("mismatched calibration rejected", False)
        except ValueError:
            ok &= check("mismatched calibration rejected", True)

        conn.close()

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
