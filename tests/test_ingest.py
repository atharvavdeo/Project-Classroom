"""Phase 1 validation gate.

Checks the three properties that, if wrong, silently corrupt every downstream
timestamp: frame count, PTS-derived duration, and motion-vector availability.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import db, ingest  # noqa: E402
from tests.fixtures import make_video  # noqa: E402

FPS = 25
SECONDS = 4.0
EXPECTED_FRAMES = int(FPS * SECONDS)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return condition


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        print("probe: h264 constant frame rate")
        h264 = make_video(tmp / "cfr.mp4", seconds=SECONDS, fps=FPS, codec="libx264")
        p = ingest.probe(h264)
        ok &= check("frame count exact", p.frame_count == EXPECTED_FRAMES,
                    f"{p.frame_count} == {EXPECTED_FRAMES}")
        ok &= check("duration from PTS", abs(p.duration_s - SECONDS) < 0.05,
                    f"{p.duration_s:.3f}s")
        ok &= check("fps derived", abs(p.avg_fps - FPS) < 0.5, f"{p.avg_fps:.2f}")
        ok &= check("not flagged VFR", p.is_vfr is False)
        ok &= check("motion vectors present", p.has_mv, f"{p.mv_per_frame:.0f}/frame")
        ok &= check("codec identified", p.codec == "h264", p.codec)
        ok &= check("sha256 computed", len(p.sha256) == 64)

        print("\nprobe: mpeg4, matching the three MPEG-4 files in the corpus")
        mp4v = make_video(tmp / "m4.mp4", seconds=SECONDS, fps=FPS, codec="mpeg4")
        p4 = ingest.probe(mp4v)
        ok &= check("frame count exact", p4.frame_count == EXPECTED_FRAMES,
                    f"{p4.frame_count}")
        ok &= check("motion vectors present", p4.has_mv, f"{p4.mv_per_frame:.0f}/frame")

        print("\nprobe: static scene still decodes and reports")
        still = make_video(tmp / "still.mp4", seconds=2.0, fps=FPS, moving_box=False)
        ps = ingest.probe(still)
        ok &= check("frame count exact", ps.frame_count == 50, f"{ps.frame_count}")

        print("\nprobe: rejects a non-video file")
        bogus = tmp / "notvideo.mp4"
        bogus.write_bytes(b"not a container")
        try:
            ingest.probe(bogus)
            ok &= check("raises on garbage input", False)
        except Exception:
            ok &= check("raises on garbage input", True)

        print("\nregister: idempotent on content hash")
        conn = db.connect(tmp / "test.db")
        session_id = db.insert(conn, "session", name="fixture")
        first = ingest.register(conn, p, session_id, "cam-a", "room-1")
        again = ingest.register(conn, p, session_id, "cam-a", "room-1")
        ok &= check("same video returns same id", first == again, f"{first} == {again}")
        other = ingest.register(conn, p4, session_id, "cam-a", "room-1")
        ok &= check("different video gets new id", other != first)
        rows = db.all_rows(conn, "SELECT COUNT(*) c FROM camera")
        ok &= check("camera not duplicated", rows[0]["c"] == 1, f"{rows[0]['c']}")
        stored = db.one(conn, "SELECT * FROM source_video WHERE id = ?", (first,))
        ok &= check("vfr stored as integer", stored["is_vfr"] in (0, 1))
        ok &= check("frame count round-trips",
                    stored["frame_count"] == EXPECTED_FRAMES)
        conn.close()

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
