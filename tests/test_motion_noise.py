"""Phase 3 gate: motion adaptation, noise classification and ROI (PRD 6.2, 8).

Runs the real scanner over real corpus footage through the new adapter, and
checks the properties that only appear at that scale: that a static region stays
quiet, that compression noise is classified and de-prioritised rather than
deleted, that a persistent region is called environmental, and that ROI output
always carries seat context alongside its subregions.

The headline check is the one that reproduces a measured defect from scratch.
`03.CCTV Mobile Usage.mkv` contains one occupied seat and one unoccupied desk
row, and the unoccupied row previously produced the twelve highest-scoring
events in the recording. Every event this pipeline produces must land on the
occupied seat.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pipeline import baseline as bl, gates, motion, roi, segmentation as seg  # noqa: E402
from pipeline.calibration import (  # noqa: E402
    DESK_WORK_SURFACE, ENVIRONMENT, LAP_BAG, OVERLAY, REVIEW_CONDITIONS,
    SEAT_CONTEXT, STATIC_BACKGROUND, UPPER_BODY, CalibrationProfile,
    MaskRegion, Seat,
)
from pipeline.health import COMPRESSION_NOISE, ENVIRONMENTAL_MOTION  # noqa: E402

VIDEO = Path("C:/DrishtiAI/03.CCTV Mobile Usage.mkv")

# The occupied seat and the unoccupied desk row, from the reviewed profile.
OCCUPIED = "S-61"
EMPTY = "R-desk"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def seat(seat_id, x1, y1, x2, y2) -> Seat:
    mid = y1 + (y2 - y1) * 0.55
    return Seat(seat_id, {
        SEAT_CONTEXT: rect(x1, y1, x2, y2),
        UPPER_BODY: rect(x1, y1, x2, mid),
        DESK_WORK_SURFACE: rect(x1, mid, x2, y2),
        LAP_BAG: rect(x1, y2 - (y2 - y1) * 0.2, x2, y2),
    }, desk_line_y=mid)


def profile() -> CalibrationProfile:
    p = CalibrationProfile(camera_id="cam12", epoch_id="e1", version=1,
                           frame_w=1280, frame_h=720, operator="gate")
    p.seats = [seat(OCCUPIED, 60, 280, 380, 700), seat(EMPTY, 700, 120, 1180, 480)]
    p.masks = [
        MaskRegion(OVERLAY, rect(8, 30, 520, 92), "burned-in date and time"),
        MaskRegion(ENVIRONMENT, rect(655, 0, 965, 232), "interior serving window"),
        MaskRegion(STATIC_BACKGROUND, rect(0, 240, 660, 700), "floor and near wall"),
    ]
    p.review_images = {c: f"{c}.jpg" for c in REVIEW_CONDITIONS}
    p.submit()
    p.approve("gate-reviewer", "2026-08-18T00:00:00Z")
    return p


def synthetic(seat_id: str, zone: str, area, blocks, residual=0.5, start=0.0):
    return [bl.ZoneWindow(seat_id, zone, start + i * 500.0, start + (i + 1) * 500.0,
                          residual, a, b)
            for i, (a, b) in enumerate(zip(area, blocks))]


def main() -> int:
    ok = True
    cfg = bl.BaselineConfig()

    print("zone mapping into the measured scanner")
    p = profile()
    converted = motion.to_scanner_calibration(p)
    ok &= check("both seats survive conversion", len(converted.seats) == 2)
    ok &= check("desk and torso zones are mapped",
                all({"desk", "torso"} <= set(s.zones) for s in converted.seats),
                "a renamed zone must fail loudly, not score nothing")
    ok &= check("the overlay mask is carried across",
                any(m.kind == "overlay" for m in converted.masks))
    ok &= check("static background is not passed as an exclusion",
                not any(m.kind == "static_background" for m in converted.masks),
                "it marks where alignment may look, not where scoring may not")

    print("\ncompression noise: isolated, low area, non-persistent")
    # One block moving in an otherwise still zone, with no persistent neighbour.
    noisy = synthetic("N", DESK_WORK_SURFACE,
                      area=[0.0, 0.0, 0.002, 0.0, 0.0, 0.003, 0.0, 0.0],
                      blocks=[0, 0, 1, 0, 0, 1, 0, 0])
    observations = motion.classify_noise(noisy, cfg, total_ms=4000.0)
    noise = [o for o in observations if o.condition == COMPRESSION_NOISE]
    ok &= check("isolated single-block activity is classified", len(noise) >= 1,
                f"{len(noise)}")
    ok &= check("it is retained and de-prioritised, not deleted",
                noise and noise[0].action == "retain_diagnostic_deprioritise",
                noise[0].action if noise else "")
    ok &= check("it does not exclude the span from baseline learning",
                noise and not noise[0].excludes_from_baseline,
                "noise is a nuisance, not missing data")
    ok &= check("an event may still bridge it",
                noise and not noise[0].blocks_bridging)

    # A run of real activity at the same low area is *not* isolated noise.
    sustained = synthetic("M", DESK_WORK_SURFACE,
                          area=[0.30] * 6, blocks=[40] * 6)
    ok &= check("sustained activity is not called noise",
                not [o for o in motion.classify_noise(sustained, cfg, 3000.0)
                     if o.condition == COMPRESSION_NOISE])

    print("\nenvironmental motion: persistent, penalised, never deleted")
    fan = synthetic("Fan", DESK_WORK_SURFACE, area=[0.40] * 100, blocks=[50] * 100)
    log = gates.GateLog()
    observations = motion.classify_noise(fan, cfg, total_ms=50_000.0, log=log)
    environmental = [o for o in observations if o.condition == ENVIRONMENTAL_MOTION]
    ok &= check("a persistent zone is environmental", len(environmental) == 1,
                f"{len(environmental)}")
    if environmental:
        ok &= check("penalised in ranking, retained as an observation",
                    environmental[0].action == "retain_named_observation_penalise_ranking",
                    environmental[0].action)
        ok &= check("the persistence is recorded as evidence",
                    "persistence" in environmental[0].evidence,
                    str(environmental[0].evidence))
    penalised = [r for r in log.records if r.condition == ENVIRONMENTAL_MOTION]
    ok &= check("and a gate record carries the penalty",
                penalised and penalised[0].state == gates.DEPRIORITISE
                and 0 < penalised[0].penalty < 1.0,
                f"{penalised[0].penalty if penalised else None}")

    print("\nevery suppressed window is routed, not dropped")
    windows = synthetic("Q", DESK_WORK_SURFACE, area=[0.001] * 40, blocks=[1] * 40)
    scored = bl.score(windows, bl.learn(windows, bl.BaselineConfig(min_windows=10)),
                      bl.BaselineConfig(min_windows=10))
    log = gates.GateLog()
    motion.suppression_gates(scored, log)
    suppressed = [s for s in scored if s.suppressed]
    ok &= check("suppressed windows exist", len(suppressed) > 0, f"{len(suppressed)}")
    ok &= check("each has a gate record", len(log.records) == len(suppressed),
                f"{len(log.records)} records for {len(suppressed)} suppressions")
    ok &= check("each names the guard that fired",
                all(r.condition for r in log.records))
    ok &= check("each carries the measured inputs",
                all("area_ratio" in r.inputs for r in log.records))
    ok &= check("nothing was discarded", log.summary()["discarded_records"] == 0)

    print("\nROI carries seat context alongside subregions (PRD 8)")
    moving = np.zeros((20, 30), dtype=bool)
    moving[8:11, 12:16] = True
    zone_mask = np.ones((20, 30), dtype=np.uint8)
    result = roi.build("A", DESK_WORK_SURFACE, 1000.0, moving, zone_mask,
                       (100.0, 200.0, 580.0, 520.0))
    ok &= check("a subregion is found", len(result.subregions) == 1,
                f"{len(result.subregions)}")
    ok &= check("the broad seat context is always present",
                result.context is not None,
                "a narrow motion blob alone is insufficient (PRD 8)")
    ok &= check("regions() returns context first",
                result.regions[0].source == roi.SOURCE_SEAT_CONTEXT)
    if result.subregions:
        region = result.subregions[0]
        # 3 blocks of context on each side of a 4x3 block cluster.
        ok &= check("the subregion is grown to hold desk and hand context",
                    region.width > 4 * 16 and region.height > 3 * 16,
                    f"{region.width:.0f}x{region.height:.0f} px "
                    f"around a {4 * 16}x{3 * 16} px cluster")

    tiny = np.zeros((20, 30), dtype=bool)
    tiny[5, 5] = True
    result = roi.build("A", DESK_WORK_SURFACE, 1000.0, tiny, zone_mask,
                       (0.0, 0.0, 480.0, 320.0))
    ok &= check("a one-block region is kept, not filtered",
                len(result.subregions) == 1)
    ok &= check("and marked uncertain rather than trusted",
                result.subregions and not result.subregions[0].certain,
                str(result.subregions[0].uncertainty if result.subregions else []))
    ok &= check("naming why", roi.UNCERTAIN_SMALL in result.subregions[0].uncertainty)

    edge = np.zeros((20, 30), dtype=bool)
    edge[0:3, 0:3] = True
    result = roi.build("A", DESK_WORK_SURFACE, 1000.0, edge, zone_mask,
                       (0.0, 0.0, 480.0, 320.0))
    ok &= check("a region touching the zone edge is flagged",
                result.subregions
                and roi.UNCERTAIN_EDGE in result.subregions[0].uncertainty,
                "it may continue into a neighbouring seat")

    whole = np.ones((20, 30), dtype=bool)
    result = roi.build("A", DESK_WORK_SURFACE, 1000.0, whole, zone_mask,
                       (0.0, 0.0, 480.0, 320.0))
    ok &= check("when the whole zone moves, no sub-box is invented",
                not result.subregions and result.context is not None,
                f"{len(result.subregions)} subregions")

    print("\nend to end on real footage")
    if not VIDEO.is_file():
        print("  SKIP: corpus not available")
    else:
        scan = motion.scan(VIDEO, profile(), motion.MotionConfig())
        ok &= check("motion vectors present and recorded",
                    scan.motion_vectors_available and scan.mv_per_frame > 1000,
                    f"{scan.mv_per_frame:.0f}/frame")
        ok &= check("throughput is far above realtime",
                    scan.realtime_factor > 10.0,
                    f"{scan.realtime_factor:.0f}x")
        ok &= check("both seats produced windows",
                    {w.seat_id for w in scan.windows} == {OCCUPIED, EMPTY},
                    str({w.seat_id for w in scan.windows}))

        baselines = bl.learn(scan.windows, cfg)
        empty_base = baselines[(EMPTY, DESK_WORK_SURFACE)]
        busy_base = baselines[(OCCUPIED, DESK_WORK_SURFACE)]
        ok &= check("the unoccupied row has a far smaller baseline",
                    empty_base.median < busy_base.median / 5,
                    f"{empty_base.median:.5f} vs {busy_base.median:.5f}")

        scored = bl.score(scan.windows, baselines, cfg)
        events, _ = seg.segment(scored, seg.SegmentConfig())
        seats = {e.seat_id for e in events}

        # The measured defect, reproduced from scratch. The unoccupied desk row
        # previously produced the twelve highest-scoring events in this file.
        ok &= check("no event lands on the unoccupied desk row",
                    EMPTY not in seats, f"{seats}")
        ok &= check("events land on the occupied seat",
                    OCCUPIED in seats, f"{seats}")

        normal = [e for e in events if e.priority == seg.PRIORITY_NORMAL]
        ok &= check("some events reach normal priority", len(normal) > 0,
                    f"{len(normal)}")
        ok &= check("short candidates are retained rather than discarded",
                    len(events) > len(normal),
                    f"{len(events) - len(normal)} retained at low priority")

        report = seg.summarise(events, scan.duration_ms)
        ok &= check("review burden is a small share of the recording",
                    report["candidate_fraction"] < 0.30,
                    f"{report['candidate_fraction'] * 100:.2f}%")

        observations = motion.classify_noise(scan.windows, cfg, scan.duration_ms)
        noise = [o for o in observations if o.condition == COMPRESSION_NOISE]
        ok &= check("compression noise is classified on real footage",
                    len(noise) > 0, f"{len(noise)} observation(s)")
        # One whole-recording span would be true and useless: every event would
        # inherit it and the field would stop discriminating.
        ok &= check("and stays localised rather than covering everything",
                    all(o.duration_ms < scan.duration_ms * 0.5 for o in noise),
                    f"longest {max(o.duration_ms for o in noise):.0f} ms of "
                    f"{scan.duration_ms:.0f} ms")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
