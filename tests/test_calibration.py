"""Phase 2 validation gate.

The load-bearing assertion is FR-08: blocks inside a screen zone or a masked
region must be excluded before any statistic is computed. If that leaks, every
occupied seat produces continuous events and the whole gate is worthless.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from classroom import calibration as cal_mod, db  # noqa: E402
from classroom.calibration import Calibration, MaskRegion, Seat  # noqa: E402

W, H = 1280, 720


def rect(x1: int, y1: int, x2: int, y2: int) -> list[tuple[float, float]]:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def build() -> Calibration:
    """Two seats laid out like the profiled CBT footage."""
    return Calibration(
        camera_id=1,
        frame_w=W,
        frame_h=H,
        seats=[
            Seat(
                label="61",
                polygon=rect(100, 200, 400, 700),
                zones={
                    "screen": rect(120, 220, 380, 400),
                    "desk": rect(100, 400, 400, 560),
                    "torso": rect(140, 380, 360, 700),
                },
                desk_line_y=560,
            ),
            Seat(
                label="62",
                polygon=rect(500, 200, 800, 700),
                zones={
                    "screen": rect(520, 220, 780, 400),
                    "desk": rect(500, 400, 800, 560),
                    "torso": rect(540, 380, 760, 700),
                },
                desk_line_y=560,
            ),
        ],
        masks=[
            # The burned-in timestamp, top-left in every profiled file.
            MaskRegion("overlay", rect(20, 20, 520, 90), "date/time overlay"),
            # Interior window reflecting the adjacent room.
            MaskRegion("reflection", rect(900, 100, 1260, 320), "glass partition"),
        ],
    )


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return condition


def main() -> int:
    ok = True
    cal = build()

    print("rasterise: exclusions")
    bm = cal_mod.rasterize(cal)
    ok &= check("grid matches 16px blocks", (bm.grid_h, bm.grid_w) == (H // 16, W // 16),
                f"{bm.grid_h}x{bm.grid_w}")

    # Overlay region centre -> block coords.
    ok &= check("overlay masked", bool(bm.excluded[55 // 16, 270 // 16]))
    ok &= check("reflection masked", bool(bm.excluded[200 // 16, 1080 // 16]))
    ok &= check("seat 61 screen masked", bool(bm.excluded[300 // 16, 250 // 16]))
    ok &= check("seat 62 screen masked", bool(bm.excluded[300 // 16, 650 // 16]))
    ok &= check("open floor not masked", not bool(bm.excluded[650 // 16, 950 // 16]))

    print("\nrasterise: zones never overlap exclusions")
    leaked = 0
    for key, mask in bm.zones.items():
        overlap = int((mask & bm.excluded).sum())
        if overlap:
            leaked += overlap
            print(f"    leak in {key}: {overlap} blocks")
    ok &= check("no zone includes an excluded block", leaked == 0, f"{leaked} leaked")

    print("\nrasterise: zones are populated and distinct")
    d61 = bm.zone_block_count("61", "desk")
    t61 = bm.zone_block_count("61", "torso")
    ok &= check("seat 61 desk has blocks", d61 > 0, f"{d61}")
    ok &= check("seat 61 torso has blocks", t61 > 0, f"{t61}")
    ok &= check("seat 61 and 62 desks disjoint",
                not bool((bm.zones[("61", "desk")] & bm.zones[("62", "desk")]).any()))
    ok &= check("desk line row computed", bm.desk_line_row["61"] == 560 // 16,
                str(bm.desk_line_row["61"]))

    print("\nrasterise: seat missing zones falls back to footprint")
    bare = Calibration(camera_id=1, frame_w=W, frame_h=H,
                       seats=[Seat(label="99", polygon=rect(100, 100, 300, 300))])
    bmb = cal_mod.rasterize(bare)
    ok &= check("bare seat still produces desk blocks",
                bmb.zone_block_count("99", "desk") > 0)
    ok &= check("bare seat desk line is None", bmb.desk_line_row["99"] is None)

    print("\nvalidation: rejects malformed input")
    for label, bad in [
        ("duplicate seat labels", Calibration(1, W, H, seats=[
            Seat("a", rect(0, 0, 10, 10)), Seat("a", rect(20, 20, 30, 30))])),
        ("no seats", Calibration(1, W, H, seats=[])),
        ("degenerate polygon", Calibration(1, W, H, seats=[Seat("a", [(0, 0), (1, 1)])])),
        ("unknown zone", Calibration(1, W, H, seats=[
            Seat("a", rect(0, 0, 10, 10), zones={"lap": rect(0, 0, 5, 5)})])),
        ("unknown mask kind", Calibration(1, W, H, seats=[Seat("a", rect(0, 0, 10, 10))],
                                          masks=[MaskRegion("blur", rect(0, 0, 5, 5))])),
    ]:
        try:
            bad.validate()
            ok &= check(f"rejects {label}", False)
        except ValueError:
            ok &= check(f"rejects {label}", True)

    print("\npersistence: round-trips through SQLite")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        conn = db.connect(tmp / "cal.db")
        session_id = db.insert(conn, "session", name="fixture")
        camera_id = db.insert(conn, "camera", session_id=session_id, label="cam-a")
        cal.camera_id = camera_id

        cal_id = cal_mod.save(conn, cal)
        loaded = cal_mod.load(conn, cal_id)
        ok &= check("seat count preserved", len(loaded.seats) == 2)
        ok &= check("zones preserved", set(loaded.seats[0].zones) == {"screen", "desk", "torso"})
        ok &= check("masks preserved", len(loaded.masks) == 2)
        ok &= check("desk line preserved", loaded.seats[0].desk_line_y == 560)
        ok &= check("version starts at 1", loaded.version == 1, str(loaded.version))

        second = cal_mod.save(conn, build() if True else cal)
        ok &= check("second save increments version",
                    cal_mod.load(conn, second).version == 2)
        ok &= check("latest_for_camera returns newest",
                    cal_mod.latest_for_camera(conn, camera_id).version == 2)

        ids = cal_mod.seat_ids(conn, cal_id)
        ok &= check("seat id map complete", set(ids) == {"61", "62"})

        # Masks identical after a round trip through rasterise.
        rt = cal_mod.rasterize(loaded)
        ok &= check("excluded mask identical after reload",
                    bool(np.array_equal(rt.excluded, bm.excluded)))
        conn.close()

    print("\npersistence: JSON round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cal.json"
        cal.write_json(p)
        back = Calibration.read_json(p)
        ok &= check("json preserves seats", len(back.seats) == 2)
        ok &= check("json preserves masks", len(back.masks) == 2)
        ok &= check("json rasterises identically",
                    bool(np.array_equal(cal_mod.rasterize(back).excluded, bm.excluded)))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
