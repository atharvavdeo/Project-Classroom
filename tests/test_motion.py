"""Phase 3 validation gate.

The scene fixture puts a per-frame-changing overlay in the corner and confines
real activity to one of two seats. That isolates the three properties the scan
has to get right:

  1. a masked region contributes nothing, no matter how violently it changes
  2. an active seat's deviation rises during the event and not outside it
  3. an idle seat stays flat throughout
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from classroom import calibration as cal_mod, db, motion  # noqa: E402
from classroom.calibration import Calibration, MaskRegion, Seat  # noqa: E402
from classroom.config import BaselineConfig, MotionConfig  # noqa: E402
from tests.fixtures import SCENE, make_scene_video  # noqa: E402

EVENT = (3.0, 5.0)
SECONDS = 8.0


def rect(box) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def build_calibration(masked: bool) -> Calibration:
    w, h = SCENE["size"]
    return Calibration(
        camera_id=1,
        frame_w=w,
        frame_h=h,
        seats=[
            Seat("A", rect(SCENE["seat_a"]), zones={"desk": rect(SCENE["seat_a"])},
                 desk_line_y=SCENE["seat_a"][3] - 40),
            Seat("B", rect(SCENE["seat_b"]), zones={"desk": rect(SCENE["seat_b"])}),
        ],
        masks=[MaskRegion("overlay", rect(SCENE["overlay"]), "timestamp")] if masked else [],
    )


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def run_scan(video: Path, cal: Calibration):
    masks = cal_mod.rasterize(cal)
    mcfg = MotionConfig()
    bcfg = BaselineConfig(learn_s=SECONDS, min_windows=4)
    windows = list(motion.scan(video, masks, mcfg))
    baselines = motion.learn_baselines(windows, bcfg)
    return windows, motion.apply_baselines(windows, baselines, mcfg, bcfg), baselines


def during(scored, key):
    return np.array([s.deviation[key] for s in scored
                     if EVENT[0] <= s.window.t_start < EVENT[1]])


def outside(scored, key):
    return np.array([s.deviation[key] for s in scored
                     if not (EVENT[0] - 0.5 <= s.window.t_start < EVENT[1] + 0.5)])


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video = make_scene_video(tmp / "scene.mp4", seconds=SECONDS, event_window=EVENT)

        print("scan: window structure")
        cal = build_calibration(masked=True)
        windows, scored, baselines = run_scan(video, cal)
        expected = int(SECONDS / MotionConfig().window_s)
        ok &= check("window count near expected",
                    abs(len(windows) - expected) <= 2, f"{len(windows)} vs ~{expected}")
        ok &= check("windows are time ordered",
                    all(b.t_start >= a.t_start for a, b in zip(windows, windows[1:])))
        ok &= check("every window carries both seats",
                    all({("A", "desk"), ("B", "desk")} <= set(w.zones) for w in windows))
        ok &= check("frames counted per window",
                    all(w.frames > 0 for w in windows))

        print("\nFR-08: the overlay is excluded")
        masks_on = cal_mod.rasterize(cal)
        masks_off = cal_mod.rasterize(build_calibration(masked=False))
        ok &= check("overlay blocks excluded when masked",
                    bool(masks_on.excluded.sum() > 0), f"{int(masks_on.excluded.sum())}")
        ok &= check("overlay blocks not excluded when unmasked",
                    int(masks_off.excluded.sum()) == 0)
        # The overlay sits outside both seat polygons, so seat statistics must be
        # identical whether or not it is masked. If binning leaked, they diverge.
        _, scored_off, _ = run_scan(video, build_calibration(masked=False))
        n = min(len(scored), len(scored_off))
        delta = max(
            abs(scored[i].window.zones[("B", "desk")].residual
                - scored_off[i].window.zones[("B", "desk")].residual)
            for i in range(n)
        )
        ok &= check("idle seat unaffected by overlay masking", delta < 1e-9,
                    f"max delta {delta:.2e}")

        print("\ndetection: active seat rises during the event")
        a_in, a_out = during(scored, ("A", "desk")), outside(scored, ("A", "desk"))
        ok &= check("seat A peaks inside the event",
                    a_in.max() > a_out.max(), f"in {a_in.max():.2f} vs out {a_out.max():.2f}")
        ok &= check("seat A clears the start threshold", a_in.max() > 4.0,
                    f"{a_in.max():.2f}")
        peak_window = max(scored, key=lambda s: s.deviation[("A", "desk")]).window
        ok &= check("peak lands within the event",
                    EVENT[0] - 0.6 <= peak_window.t_start < EVENT[1] + 0.6,
                    f"t={peak_window.t_start:.2f}")

        print("\ndetection: idle seat stays flat")
        b_all = np.array([s.deviation[("B", "desk")] for s in scored])
        ok &= check("seat B never clears the start threshold", b_all.max() < 4.0,
                    f"max {b_all.max():.2f}")
        ok &= check("seat B area ratio near zero",
                    max(w.zones[("B", "desk")].area_ratio for w in windows) < 0.15,
                    f"{max(w.zones[('B', 'desk')].area_ratio for w in windows):.3f}")

        print("\nglobal motion and exposure")
        ok &= check("static camera not flagged as moved",
                    not any(s.window.camera_moved for s in scored))
        ok &= check("no exposure step on constant lighting",
                    not any(s.window.exposure_step for s in scored))
        ok &= check("luma measured", all(w.luma_mean > 0 for w in windows))

        print("\nbelow-desk feature")
        below = max(w.zones[("A", "desk")].below_desk for w in windows)
        ok &= check("seat A registers below-desk motion", below > 0.0, f"{below:.2f}")
        ok &= check("seat B has no desk line, reports zero",
                    all(w.zones[("B", "desk")].below_desk == 0.0 for w in windows))

        print("\nbaselines")
        ok &= check("baseline learned for both seats",
                    {("A", "desk"), ("B", "desk")} <= set(baselines))
        ok &= check("baseline sample count recorded",
                    all(b.sample_windows > 0 for b in baselines.values()))

        print("\npersistence")
        conn = db.connect(tmp / "motion.db")
        session_id = db.insert(conn, "session", name="fixture")
        camera_id = db.insert(conn, "camera", session_id=session_id, label="cam")
        cal.camera_id = camera_id
        cal_id = cal_mod.save(conn, cal)
        seat_ids = cal_mod.seat_ids(conn, cal_id)
        video_id = db.insert(
            conn, "source_video", camera_id=camera_id, path=str(video),
            sha256="x" * 64, size_bytes=1, container="mp4", codec="h264",
            pix_fmt="yuv420p", width=SCENE["size"][0], height=SCENE["size"][1],
            avg_fps=25.0, duration_s=SECONDS, frame_count=200, is_vfr=0, has_mv=1,
        )
        # rasterize emits both desk and torso for every seat, falling back to
        # the footprint where a zone was not drawn: 2 seats x 2 zones.
        zone_keys = len(scored[0].window.zones)
        rows = motion.persist(conn, video_id, seat_ids, scored)
        ok &= check("motion rows written", rows == len(scored) * zone_keys,
                    f"{rows} = {len(scored)} windows x {zone_keys} zones")
        motion.persist_baselines(conn, cal_id, seat_ids, baselines)

        stored = db.all_rows(
            conn,
            "SELECT COUNT(*) c FROM motion_window WHERE source_video_id = ?",
            (video_id,),
        )
        ok &= check("rows readable back", stored[0]["c"] == rows)
        again = motion.persist(conn, video_id, seat_ids, scored)
        recount = db.all_rows(
            conn,
            "SELECT COUNT(*) c FROM motion_window WHERE source_video_id = ?",
            (video_id,),
        )
        ok &= check("rescan is idempotent", recount[0]["c"] == again, f"{recount[0]['c']}")
        conn.close()

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
