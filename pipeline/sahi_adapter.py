"""Candidate-crop slicing, measured rather than assumed (PRD 10, PRD 19).

PRD 10 permits SAHI under one restriction, and it is the whole design:

> SAHI: only on the same **candidate crop**, never unconditional 3x3/10x10
> full-video grids; explicit slice/overlap/edge/threshold/NMS configuration.

The prohibited version is the tempting one. Slicing every frame of a full
recording into a 10x10 grid multiplies inference cost by a hundred and produces
a hundred times the false positives, most of them on desk edges and monitor
bezels. Slicing a crop the motion stage already selected costs a few extra
passes over a few hundred crops.

PRD 19 is equally clear about what slicing can and cannot do: it "can aid
small-object detection but cannot restore detail missing from the CCTV source."
A slice magnifies a region into the detector's input; it does not add pixels. So
the abstention floors in `object_detection` are judged on the **native**
footprint and are unaffected by slicing -- a 24 px object is 24 px whether it
arrives alone or as one ninth of a crop.

### The merge

Detections from overlapping slices are projected back to source coordinates,
then merged by class with greedy NMS. Two properties are enforced:

  * **the raw slice output survives the merge.** PRD 10 requires both preserved,
    and a merged box with no trail back to the slices that produced it cannot be
    audited when it turns out to be a bezel seen twice.
  * **edge detections are marked, not dropped.** An object cut by a slice
    boundary is exactly what overlap exists to catch; discarding boundary
    detections would silently remove the class of object slicing was added for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .object_detection import ObjectDetection


@dataclass
class SahiConfig:
    # Slice grid over the candidate crop. 2x2 with overlap, not 3x3 or 10x10:
    # the crops reaching here are already a seat or a hand context, a few
    # hundred px on a side, and a 3x3 over that produces slices smaller than the
    # detector's own receptive field.
    rows: int = 2
    cols: int = 2

    # Fraction of a step shared between two adjacent slices. An object narrower
    # than this band cannot be cut by every boundary at once, which is the
    # property that makes edge objects recoverable.
    #
    # Each slice is padded by half of it on each interior side, so the shared
    # band between neighbours is exactly `overlap * step`. Padding by the full
    # fraction on both sides -- the obvious implementation -- gives a band twice
    # the configured number, and the config then does not mean what it says.
    overlap: float = 0.25

    # A detection touching a slice edge is flagged. It is still merged and still
    # reported; the flag travels so a reviewer can see that a box may be a
    # fragment of something the slice cut.
    edge_margin_px: float = 4.0

    # IoU above which two same-class boxes from different slices are one object.
    nms_iou: float = 0.45

    # Slices whose own native footprint falls under this are not run: a slice of
    # an already-small crop is magnification, not resolution.
    min_slice_native_px: int = 32

    # Whether the whole unsliced crop is also run and merged in. Kept on: a
    # sliced-only configuration loses objects larger than one slice, and the
    # comparison against `*_native` would then confound two changes at once.
    include_full_crop: bool = True


@dataclass
class Slice:
    """One tile of a candidate crop, in source coordinates."""

    index: int
    x1: float
    y1: float
    x2: float
    y2: float
    is_full_crop: bool = False

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_row(self) -> dict:
        return {"index": self.index, "x1": round(self.x1, 2),
                "y1": round(self.y1, 2), "x2": round(self.x2, 2),
                "y2": round(self.y2, 2), "is_full_crop": self.is_full_crop,
                "width": round(self.width, 2), "height": round(self.height, 2)}


def slices(crop, cfg: SahiConfig | None = None) -> list[Slice]:
    """Tile one candidate crop. Never a frame, never a whole video."""
    cfg = cfg or SahiConfig()
    out: list[Slice] = []

    width, height = crop.native_w, crop.native_h
    step_x = width / cfg.cols
    step_y = height / cfg.rows
    pad_x = step_x * cfg.overlap / 2.0
    pad_y = step_y * cfg.overlap / 2.0

    index = 0
    for row in range(cfg.rows):
        for col in range(cfg.cols):
            x1 = crop.x1 + col * step_x - (pad_x if col > 0 else 0.0)
            y1 = crop.y1 + row * step_y - (pad_y if row > 0 else 0.0)
            x2 = crop.x1 + (col + 1) * step_x + (pad_x if col < cfg.cols - 1 else 0.0)
            y2 = crop.y1 + (row + 1) * step_y + (pad_y if row < cfg.rows - 1 else 0.0)
            x1, y1 = max(x1, crop.x1), max(y1, crop.y1)
            x2, y2 = min(x2, crop.x2), min(y2, crop.y2)
            if min(x2 - x1, y2 - y1) < cfg.min_slice_native_px:
                continue
            out.append(Slice(index, x1, y1, x2, y2))
            index += 1

    if cfg.include_full_crop:
        out.append(Slice(index, crop.x1, crop.y1, crop.x2, crop.y2,
                         is_full_crop=True))
    return out


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def touches_edge(box: tuple[float, float, float, float], tile: Slice,
                 cfg: SahiConfig) -> bool:
    """Whether a detection reaches or crosses a slice boundary.

    Reaches *or crosses*. An earlier version compared the distance to the
    boundary symmetrically, which flagged a box ending exactly at the edge but
    missed one extending past it -- and a box extending past the edge is more
    likely to be a fragment, not less. The comparison is therefore one-sided.
    """
    margin = cfg.edge_margin_px
    return (box[0] <= tile.x1 + margin or box[1] <= tile.y1 + margin
            or box[2] >= tile.x2 - margin or box[3] >= tile.y2 - margin)


def merge(raw: list[ObjectDetection], cfg: SahiConfig | None = None
          ) -> list[ObjectDetection]:
    """Greedy per-class NMS over slice detections, in source coordinates.

    Per class, never across: a phone and a hand overlapping is two objects, and
    class-agnostic NMS would delete one of them because it scored lower.

    The surviving box records how many raw boxes merged into it. A box merged
    from four slices is corroborated by four independent passes; a box merged
    from one appeared in exactly one slice and may be a boundary artefact. The
    two are not the same evidence and the count is what distinguishes them.
    """
    cfg = cfg or SahiConfig()
    out: list[ObjectDetection] = []

    by_class: dict[str, list[ObjectDetection]] = {}
    for detection in raw:
        by_class.setdefault(detection.cls, []).append(detection)

    for cls, group in by_class.items():
        remaining = sorted(group, key=lambda d: -d.confidence)
        while remaining:
            best = remaining.pop(0)
            absorbed = [best]
            keep = []
            for other in remaining:
                if iou(best.box, other.box) >= cfg.nms_iou:
                    absorbed.append(other)
                else:
                    keep.append(other)
            remaining = keep

            # The merged box is the highest-scoring one, not an average. An
            # averaged box is a rectangle no detector ever produced, and it
            # cannot be traced back to a slice for review.
            merged = ObjectDetection(
                crop_id=best.crop_id, config_id=best.config_id,
                pts_ms=best.pts_ms, cls=cls, confidence=best.confidence,
                x1=best.x1, y1=best.y1, x2=best.x2, y2=best.y2,
                native_px=best.native_px, abstained=best.abstained,
                abstain_reason=best.abstain_reason,
                slice_index=best.slice_index, merged_from=len(absorbed),
                seat_state=best.seat_state, track_id=best.track_id,
                event_id=best.event_id)
            out.append(merged)

    out.sort(key=lambda d: (d.pts_ms, -d.confidence))
    return out


def sahi_runner(base_runner, cfg: SahiConfig | None = None,
                on_slice=None):
    """Wrap a native crop runner so it runs per slice and merges.

    `base_runner` is any callable taking an object with `x1/y1/x2/y2`,
    `native_w/native_h`, `pts_ms` and `crop_id` -- the same contract the native
    runners already satisfy -- so exactly the same detector code runs in both
    arms of the comparison. That is what makes the A/B a test of slicing rather
    than a test of two code paths.

    `on_slice(crop, tile, detections)` receives every slice's raw output, so the
    caller can persist it. PRD 10 requires every slice and its raw detections to
    be stored, not only the merged result.
    """
    cfg = cfg or SahiConfig()

    class _Tile:
        """A slice presented to the base runner as if it were a crop."""

        def __init__(self, crop, tile: Slice):
            self.crop_id = crop.crop_id
            self.pts_ms = crop.pts_ms
            self.seat_state = crop.seat_state
            self.track_id = crop.track_id
            self.event_id = crop.event_id
            self.input_size = crop.input_size
            self.x1, self.y1, self.x2, self.y2 = tile.x1, tile.y1, tile.x2, tile.y2

        @property
        def native_w(self):
            return self.x2 - self.x1

        @property
        def native_h(self):
            return self.y2 - self.y1

        @property
        def native_px(self):
            return max(self.native_w, 0.0) * max(self.native_h, 0.0)

        @property
        def magnification(self):
            longest = max(self.native_w, self.native_h)
            return self.input_size / longest if longest > 0 else 0.0

    def run(crop):
        raw: list[ObjectDetection] = []
        for tile in slices(crop, cfg):
            proposals = list(base_runner(_Tile(crop, tile)))
            tile_rows = []
            for cls, confidence, x1, y1, x2, y2 in proposals:
                detection = ObjectDetection(
                    crop_id=crop.crop_id, config_id="", pts_ms=crop.pts_ms,
                    cls=cls, confidence=round(float(confidence), 4),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    native_px=round(crop.native_px, 1),
                    slice_index=tile.index, seat_state=crop.seat_state,
                    track_id=crop.track_id, event_id=crop.event_id)
                if not tile.is_full_crop and \
                        touches_edge(detection.box, tile, cfg):
                    # Marked, never dropped. An object cut by a boundary is what
                    # the overlap exists to recover.
                    detection.abstain_reason = "touches_slice_edge"
                tile_rows.append(detection)
            if on_slice is not None:
                on_slice(crop, tile, tile_rows)
            raw.extend(tile_rows)

        merged = merge(raw, cfg)
        return [(d.cls, d.confidence, d.x1, d.y1, d.x2, d.y2) for d in merged]

    return run


@dataclass
class SliceLog:
    """Every slice run, for the artifact requirement in PRD 10."""

    rows: list[dict] = field(default_factory=list)

    def record(self, crop, tile: Slice, detections: list[ObjectDetection]) -> None:
        self.rows.append({
            "crop_id": crop.crop_id,
            "pts_ms": round(crop.pts_ms, 3),
            **tile.to_row(),
            "raw_detections": len(detections),
            "classes": sorted({d.cls for d in detections}),
            "edge_flagged": sum(1 for d in detections
                                if d.abstain_reason == "touches_slice_edge"),
        })

    def summary(self) -> dict:
        if not self.rows:
            return {"slices": 0, "note": "no slices were run"}
        full = sum(1 for r in self.rows if r["is_full_crop"])
        return {
            "slices": len(self.rows),
            "tiles": len(self.rows) - full,
            "full_crop_passes": full,
            "slices_per_crop": round(
                len(self.rows) / len({r["crop_id"] for r in self.rows}), 2),
            "raw_detections": sum(r["raw_detections"] for r in self.rows),
            "edge_flagged": sum(r["edge_flagged"] for r in self.rows),
            "note": "slicing magnifies a region into the detector input; it "
                    "does not add source pixels. Abstention floors are judged "
                    "on the native footprint and are unchanged by it (PRD 19).",
        }
