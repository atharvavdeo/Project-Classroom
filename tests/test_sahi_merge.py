"""Phase 5 gate: candidate-crop slicing and its merge (PRD 10, PRD 19).

PRD 10 restricts slicing to the candidate crop and bans the version everyone
reaches for first:

> SAHI: only on the same **candidate crop**, never unconditional 3x3/10x10
> full-video grids

and PRD 19 bounds what it can achieve:

> slicing can aid small-object detection but cannot restore detail missing from
> the CCTV source.

So three things are tested, each of which is a way slicing goes wrong quietly:

  * **containment.** Every slice lies inside its crop. A slice that runs past
    the crop is inference on pixels the motion stage never selected, and the
    resulting detection is attributed to a candidate it did not come from.
  * **the merge.** Overlapping slices see the same object twice. Merging is
    per class, so a phone and a hand in the same place stay two objects, and the
    surviving box is one a detector actually produced rather than an average of
    several — an averaged box cannot be traced back to a slice for review.
  * **edge detections survive.** An object cut by a slice boundary is the exact
    case overlap exists to recover. Dropping boundary detections would silently
    remove the class of object slicing was added for, and the run would look
    cleaner for it.

Also asserted: slicing does not move the abstention floor. A 24 px object is
24 px whether it arrives whole or as one quarter of a crop.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.crops import NATIVE_FALLBACK, ObjectCrop  # noqa: E402
from pipeline.object_detection import (  # noqa: E402
    HAND, INSUFFICIENT, MOUSE, PHONE, DetectConfig, ObjectDetection, detect,
)
from pipeline.sahi_adapter import (  # noqa: E402
    SahiConfig, SliceLog, iou, merge, sahi_runner, slices, touches_edge,
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def make_crop(x1=100.0, y1=200.0, x2=420.0, y2=520.0, crop_id="c1") -> ObjectCrop:
    return ObjectCrop(
        crop_id=crop_id, pose_row_id="p1", event_id="EV-1",
        source_video_sha256="abc123", pts_ms=1000.0,
        crop_config=NATIVE_FALLBACK, x1=x1, y1=y1, x2=x2, y2=y2,
        seat_state="S-A", track_id=1)


def det(cls, confidence, box, slice_index=None, pts=1000.0) -> ObjectDetection:
    return ObjectDetection(
        crop_id="c1", config_id="dfine_sahi", pts_ms=pts, cls=cls,
        confidence=confidence, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
        slice_index=slice_index, seat_state="S-A", track_id=1, event_id="EV-1")


def main() -> int:
    ok = True
    cfg = SahiConfig()
    crop = make_crop()

    print("slices tile the candidate crop, and only the candidate crop")
    tiles = slices(crop, cfg)
    grid = [t for t in tiles if not t.is_full_crop]
    ok &= check("a 2x2 grid plus the full crop", len(grid) == 4 and
                len(tiles) == 5, f"{len(grid)} tiles + "
                f"{len(tiles) - len(grid)} full-crop pass")
    ok &= check("every slice lies inside the crop",
                all(t.x1 >= crop.x1 - 1e-6 and t.y1 >= crop.y1 - 1e-6
                    and t.x2 <= crop.x2 + 1e-6 and t.y2 <= crop.y2 + 1e-6
                    for t in tiles),
                "a slice outside its crop detects on pixels motion never chose")
    ok &= check("the tiles cover the crop between them",
                min(t.x1 for t in grid) == crop.x1
                and max(t.x2 for t in grid) == crop.x2
                and min(t.y1 for t in grid) == crop.y1
                and max(t.y2 for t in grid) == crop.y2)
    ok &= check("the full crop is included so large objects are not lost",
                any(t.is_full_crop for t in tiles),
                "sliced-only would lose objects bigger than one slice and "
                "confound two changes in the A/B at once")

    print("\nneighbouring slices overlap, which is what recovers cut objects")
    top_left = next(t for t in grid if t.x1 == crop.x1 and t.y1 == crop.y1)
    top_right = next(t for t in grid if t.x2 == crop.x2 and t.y1 == crop.y1)
    ok &= check("horizontal neighbours share a band",
                top_left.x2 > top_right.x1,
                f"overlap {top_left.x2 - top_right.x1:.0f} px")
    ok &= check("the overlap matches the configured fraction",
                abs((top_left.x2 - top_right.x1)
                    - crop.native_w / cfg.cols * cfg.overlap) < 1.0)

    print("\nno 3x3 or 10x10 full-video grid is reachable from this module")
    ok &= check("the default grid is 2x2", (cfg.rows, cfg.cols) == (2, 2),
                f"{cfg.rows}x{cfg.cols}")
    ok &= check("slices() takes a crop, never a frame or a video",
                "crop" in slices.__code__.co_varnames[:1],
                "the API cannot express an unconditional full-video grid")

    print("\ntiny crops are not sliced further into magnification")
    small = make_crop(0.0, 0.0, 50.0, 50.0, crop_id="c-small")
    small_tiles = [t for t in slices(small, cfg) if not t.is_full_crop]
    ok &= check("a 50 px crop yields no sub-slices", not small_tiles,
                f"{len(small_tiles)} tiles; a slice of an already-small crop "
                f"is magnification, not resolution")

    print("\nthe merge is per class, and keeps a box a detector produced")
    raw = [
        det(PHONE, 0.80, (200.0, 300.0, 232.0, 322.0), slice_index=0),
        det(PHONE, 0.72, (203.0, 302.0, 234.0, 324.0), slice_index=1),
        det(PHONE, 0.65, (201.0, 299.0, 233.0, 323.0), slice_index=2),
        det(HAND, 0.90, (198.0, 296.0, 240.0, 330.0), slice_index=0),
    ]
    merged = merge(raw, cfg)
    phones = [m for m in merged if m.cls == PHONE]
    hands = [m for m in merged if m.cls == HAND]
    ok &= check("three overlapping phone boxes become one",
                len(phones) == 1, str(len(phones)))
    ok &= check("the overlapping hand is not deleted by the phone",
                len(hands) == 1,
                "class-agnostic NMS would drop whichever scored lower")
    ok &= check("the survivor is the highest-scoring original box",
                phones[0].confidence == 0.80 and
                (phones[0].x1, phones[0].y1) == (200.0, 300.0),
                "an averaged box is a rectangle no detector produced")
    ok &= check("corroboration is counted", phones[0].merged_from == 3,
                f"merged_from={phones[0].merged_from}")

    print("\ncorroboration distinguishes a real object from a boundary artefact")
    lone = merge([det(PHONE, 0.80, (200.0, 300.0, 232.0, 322.0), slice_index=0)],
                 cfg)
    ok &= check("a box seen in one slice reports merged_from=1",
                lone[0].merged_from == 1,
                "one slice is one pass, not corroboration")

    print("\ndistant same-class boxes stay separate objects")
    apart = merge([det(PHONE, 0.8, (200.0, 300.0, 232.0, 322.0)),
                   det(PHONE, 0.7, (380.0, 480.0, 412.0, 502.0))], cfg)
    ok &= check("two phones on opposite sides of a desk are two objects",
                len(apart) == 2, str(len(apart)))

    print("\nedge detections are flagged, never dropped")
    tile = top_left
    on_edge = (tile.x2 - 2.0, tile.y1 + 40.0, tile.x2 + 10.0, tile.y1 + 70.0)
    inside = (tile.x1 + 50.0, tile.y1 + 50.0, tile.x1 + 80.0, tile.y1 + 72.0)
    ok &= check("a box abutting a slice boundary is detected as such",
                touches_edge(on_edge, tile, cfg))
    ok &= check("a box in the slice interior is not", not touches_edge(inside, tile, cfg))

    calls = {"n": 0}

    def base_runner(tile_view):
        calls["n"] += 1
        # Emit one box hugging this tile's left edge every time.
        return [(PHONE, 0.7, tile_view.x1, tile_view.y1 + 10.0,
                 tile_view.x1 + 30.0, tile_view.y1 + 32.0)]

    log = SliceLog()
    runner = sahi_runner(base_runner, cfg, on_slice=log.record)
    produced = list(runner(crop))
    ok &= check("the base runner ran once per slice", calls["n"] == len(tiles),
                f"{calls['n']} calls for {len(tiles)} slices")
    ok &= check("edge-flagged boxes still reach the merged output",
                len(produced) >= 1, f"{len(produced)} merged")
    ok &= check("every slice was logged with its raw output",
                len(log.rows) == len(tiles), str(len(log.rows)))
    ok &= check("the log counts edge-flagged detections",
                log.summary()["edge_flagged"] >= 1,
                str(log.summary()["edge_flagged"]))
    ok &= check("the log distinguishes tiles from the full-crop pass",
                log.summary()["full_crop_passes"] == 1,
                str(log.summary()["full_crop_passes"]))

    print("\nthe base runner sees exactly the contract a native runner sees")
    seen = {}

    def probe(tile_view):
        seen["native_w"] = tile_view.native_w
        seen["magnification"] = tile_view.magnification
        seen["crop_id"] = tile_view.crop_id
        return []

    list(sahi_runner(probe, cfg)(crop))
    ok &= check("a slice exposes native_w, magnification and crop_id",
                set(seen) == {"native_w", "magnification", "crop_id"},
                "identical detector code runs in both arms, so the A/B tests "
                "slicing rather than two code paths")

    print("\nslicing does not move the abstention floor (PRD 19)")
    detect_cfg = DetectConfig(abstain_below_native_px=32)
    thin = make_crop(0.0, 0.0, 24.0, 18.0, crop_id="c-thin")

    def never_called(_crop):
        raise AssertionError("the detector must not see a crop under the floor")

    result = detect([thin], never_called, "dfine_sahi", detect_cfg)
    ok &= check("a sub-floor crop abstains before the model runs",
                result.crops_abstained == 1 and result.state == "available",
                "magnifying it adds interpolation, not detail")
    ok &= check("the abstention is a row, not a missing row",
                len(result.merged) == 1 and result.merged[0].cls == INSUFFICIENT,
                "an absent row is indistinguishable from a crop nobody processed")
    ok &= check("the reason states the native footprint",
                "24x18" in result.merged[0].abstain_reason,
                result.merged[0].abstain_reason[:60])

    print("\nIoU behaves at the boundaries")
    ok &= check("identical boxes are 1.0",
                abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9)
    ok &= check("disjoint boxes are 0.0", iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0)
    ok &= check("touching-but-not-overlapping is 0.0",
                iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0)

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
