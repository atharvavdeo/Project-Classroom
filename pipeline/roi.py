"""Regions of interest, uncertainty-aware (PRD 8).

PRD 8 is specific about the shape of this output:

> ROI output is an uncertainty-aware set of active subregions **plus** broad seat
> context. Tiny object detection needs desk/hand context; a narrow motion blob
> alone is insufficient.

Both halves matter, and the second is the one that is easy to drop. A motion
blob tells you *where something moved*; it does not contain enough pixels around
it to tell you *what moved*. At the measured object scale on this corpus — a
phone at roughly 30x20 px — a crop tight to the moving blocks would be a
handful of pixels with no desk, no hand and no forearm in it, and no detector or
verifier could make anything of that.

So every ROI set carries the seat context alongside the subregions, and the
subregions are annotated with why they are uncertain rather than being filtered
to the confident ones.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# Why a subregion is uncertain. These accumulate; a region can be several at
# once, and the reasons travel with it to the crop stage and the reviewer.
UNCERTAIN_SMALL = "below_object_scale"
UNCERTAIN_EDGE = "touches_zone_edge"
UNCERTAIN_SPARSE = "sparse_blocks"
UNCERTAIN_QUALITY = "quality_affected"
UNCERTAIN_UNCOMPENSATED = "camera_motion_uncompensated"

# What produced the region.
SOURCE_BLOCKS = "moving_blocks"
SOURCE_SEAT_CONTEXT = "seat_context"


@dataclass
class RoiConfig:
    # A block is 16x16 px. A region smaller than this many blocks cannot hold a
    # 30x20 px object plus any surrounding context, so it is kept and flagged
    # rather than trusted on its own.
    min_blocks: int = 4

    # Blocks within this Chebyshev distance join the same region. One block of
    # slack bridges the gap a moving hand leaves between its fingers and wrist
    # without merging two seats' activity.
    link_distance: int = 1

    # Context added around a block cluster, in blocks, before it becomes a crop.
    # This is the "desk/hand context" PRD 8 requires; without it the crop is the
    # blob and the blob is not identifiable.
    context_blocks: int = 3

    # A region occupying more than this share of its zone is not a subregion at
    # all -- the whole seat moved, and reporting a sub-box would imply a
    # precision that is not there.
    whole_zone_fraction: float = 0.60


@dataclass
class Region:
    """One active subregion, or the broad seat context that accompanies it."""

    seat_id: str
    zone: str
    source: str
    # Source-frame pixel coordinates, so every downstream crop and overlay maps
    # back without a second conversion.
    x1: float
    y1: float
    x2: float
    y2: float
    blocks: int = 0
    area_ratio: float = 0.0
    uncertainty: list[str] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def px_area(self) -> float:
        return self.width * self.height

    @property
    def certain(self) -> bool:
        return not self.uncertainty

    def to_row(self) -> dict:
        return {**asdict(self), "px_area": round(self.px_area, 1)}


@dataclass
class RoiSet:
    """The ROI output for one seat-zone at one instant."""

    seat_id: str
    zone: str
    pts_ms: float
    subregions: list[Region] = field(default_factory=list)
    context: Region | None = None

    @property
    def regions(self) -> list[Region]:
        """Subregions plus context. The context is never optional (PRD 8)."""
        return ([self.context] if self.context else []) + self.subregions

    def to_row(self) -> dict:
        return {
            "seat_id": self.seat_id,
            "zone": self.zone,
            "pts_ms": round(self.pts_ms, 3),
            "subregions": [r.to_row() for r in self.subregions],
            "context": self.context.to_row() if self.context else None,
            "any_certain": any(r.certain for r in self.subregions),
        }


def _components(mask: np.ndarray, link: int) -> list[np.ndarray]:
    """Connected components of a boolean block grid, with `link` slack.

    Written directly rather than via cv2.connectedComponents so the linking
    distance is explicit: a moving hand leaves gaps between fingers and wrist,
    and a strict 4-connectivity split would report three regions where there is
    one hand.
    """
    if not mask.any():
        return []
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy in range(-link, link + 1):
                for dx in range(-link, link + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width \
                            and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        out.append(np.array(component))
    return out


def build(seat_id: str, zone: str, pts_ms: float, moving: np.ndarray,
          zone_mask: np.ndarray, zone_bbox: tuple[float, float, float, float],
          cfg: RoiConfig | None = None, block: int = 16,
          quality_affected: bool = False,
          camera_uncompensated: bool = False) -> RoiSet:
    """Build the ROI set for one seat-zone from its moving-block grid.

    `moving` and `zone_mask` are block grids of the same shape; `zone_bbox` is
    the seat-zone's pixel bounding box, used for the broad context region that
    always accompanies the subregions.
    """
    cfg = cfg or RoiConfig()
    inside = moving & (zone_mask > 0)
    zone_blocks = int((zone_mask > 0).sum())

    x1, y1, x2, y2 = zone_bbox
    context = Region(
        seat_id=seat_id, zone=zone, source=SOURCE_SEAT_CONTEXT,
        x1=x1, y1=y1, x2=x2, y2=y2,
        blocks=zone_blocks,
        area_ratio=round(float(inside.sum()) / zone_blocks, 5) if zone_blocks else 0.0,
    )
    if quality_affected:
        context.uncertainty.append(UNCERTAIN_QUALITY)
    if camera_uncompensated:
        context.uncertainty.append(UNCERTAIN_UNCOMPENSATED)

    roi = RoiSet(seat_id=seat_id, zone=zone, pts_ms=pts_ms, context=context)
    if not inside.any():
        return roi

    grid_h, grid_w = moving.shape
    for component in _components(inside, cfg.link_distance):
        ys, xs = component[:, 0], component[:, 1]
        count = len(component)

        # Grow by the context margin, clamped to the frame's block grid. This
        # is what makes the region a usable crop rather than a blob.
        gy1 = max(int(ys.min()) - cfg.context_blocks, 0)
        gy2 = min(int(ys.max()) + cfg.context_blocks + 1, grid_h)
        gx1 = max(int(xs.min()) - cfg.context_blocks, 0)
        gx2 = min(int(xs.max()) + cfg.context_blocks + 1, grid_w)

        region = Region(
            seat_id=seat_id, zone=zone, source=SOURCE_BLOCKS,
            x1=float(gx1 * block), y1=float(gy1 * block),
            x2=float(gx2 * block), y2=float(gy2 * block),
            blocks=count,
            area_ratio=round(count / zone_blocks, 5) if zone_blocks else 0.0,
        )

        if count < cfg.min_blocks:
            region.uncertainty.append(UNCERTAIN_SMALL)
        if count <= 2:
            region.uncertainty.append(UNCERTAIN_SPARSE)
        # A region touching the zone edge may be part of something that
        # continues into the neighbouring seat, which is exactly the case that
        # must not be silently attributed.
        if (ys.min() == 0 or xs.min() == 0
                or ys.max() == grid_h - 1 or xs.max() == grid_w - 1):
            region.uncertainty.append(UNCERTAIN_EDGE)
        if quality_affected:
            region.uncertainty.append(UNCERTAIN_QUALITY)
        if camera_uncompensated:
            region.uncertainty.append(UNCERTAIN_UNCOMPENSATED)

        # If the "subregion" is most of the zone, the whole seat moved and a
        # sub-box would imply precision that is not there. The context region
        # already covers it.
        if zone_blocks and count / zone_blocks >= cfg.whole_zone_fraction:
            continue

        roi.subregions.append(region)

    roi.subregions.sort(key=lambda r: -r.blocks)
    return roi


def heat(rois: list[RoiSet], grid_h: int, grid_w: int, block: int = 16
         ) -> np.ndarray:
    """Accumulate subregion activity into a block-resolution heatmap."""
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
    for roi in rois:
        for region in roi.subregions:
            y1 = int(region.y1 // block)
            y2 = min(int(region.y2 // block), grid_h)
            x1 = int(region.x1 // block)
            x2 = min(int(region.x2 // block), grid_w)
            heatmap[y1:y2, x1:x2] += region.blocks
    return heatmap


def summarise(rois: list[RoiSet]) -> dict:
    subregions = [r for roi in rois for r in roi.subregions]
    uncertain = [r for r in subregions if not r.certain]
    reasons: dict[str, int] = {}
    for region in uncertain:
        for reason in region.uncertainty:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "roi_sets": len(rois),
        "subregions": len(subregions),
        "certain_subregions": len(subregions) - len(uncertain),
        "uncertain_subregions": len(uncertain),
        "uncertainty_reasons": reasons,
        "sets_with_context": sum(1 for r in rois if r.context is not None),
    }
