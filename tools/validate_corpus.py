"""Re-measure the real corpus and confirm the PRD 5.1 table.

Runs ingest over every source file, prints the measured table, and compares it
against the values recorded in the PRD so drift is visible rather than assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import db, ingest  # noqa: E402
from classroom.config import Config  # noqa: E402

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/DrishtiAI")

# The table as recorded in PRD 5.1: codec, width, height, fps, duration, vfr.
EXPECTED = {
    "01": ("h264", 1280, 720, 25.0, 131, False),
    "02": ("h264", 1280, 720, 25.0, 212, False),
    "03": ("mpeg4", 1280, 720, 15.0, 282, True),
    "04": ("h264", 640, 480, 8.0, 143, False),
    "05": ("mpeg4", 1280, 720, 25.0, 241, True),
    "Se": ("mpeg4", 1280, 720, 21.9, 88, True),
}


def main() -> int:
    files = sorted(list(CORPUS.glob("*.mkv")) + list(CORPUS.glob("*.mp4")))
    if not files:
        print(f"no source files under {CORPUS}")
        return 1

    cfg = Config(db_path=Path("data/corpus.db"))
    conn = db.connect(cfg.db_path)
    session_id = db.insert(conn, "session", name="real corpus")

    print(f"{'file':34s} {'codec':6s} {'res':10s} {'fps':>6s} {'dur':>7s} "
          f"{'Mbps':>6s} {'vfr':>4s} {'mv/f':>6s} {'frames':>7s}")
    print("-" * 96)

    ok = True
    total = 0.0
    for path in files:
        profile = ingest.probe(path)
        total += profile.duration_s
        ingest.register(conn, profile, session_id, path.stem[:24], "corpus")
        print(f"{path.name[:34]:34s} {profile.codec:6s} "
              f"{profile.width}x{profile.height:<5} {profile.avg_fps:>6.1f} "
              f"{profile.duration_s:>7.1f} {profile.bit_rate_mbps:>6.2f} "
              f"{'yes' if profile.is_vfr else 'no':>4s} "
              f"{profile.mv_per_frame:>6.0f} {profile.frame_count:>7d}")

        key = path.name[:2]
        if key in EXPECTED:
            codec, w, h, fps, dur, vfr = EXPECTED[key]
            for label, got, want in (
                ("codec", profile.codec, codec),
                ("width", profile.width, w),
                ("height", profile.height, h),
                ("vfr", profile.is_vfr, vfr),
            ):
                if got != want:
                    print(f"    MISMATCH {label}: measured {got}, PRD says {want}")
                    ok = False
            if abs(profile.avg_fps - fps) > 0.3:
                print(f"    MISMATCH fps: measured {profile.avg_fps:.2f}, PRD says {fps}")
                ok = False
            if abs(profile.duration_s - dur) > 1.5:
                print(f"    MISMATCH duration: measured {profile.duration_s:.1f}, "
                      f"PRD says {dur}")
                ok = False
        if not profile.has_mv:
            print("    FAIL: no codec motion vectors")
            ok = False

    print("-" * 96)
    print(f"{len(files)} files, {total:.1f} s total ({total/60:.1f} min)")
    print(f"\nPRD 5.1 table: {'CONFIRMED' if ok else 'DRIFT DETECTED'}")
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
