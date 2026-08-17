"""Phase 2 - calibration, zones and masks.

A calibration profile pins the scene model from PRD 4: every seat is split into
screen, desk and torso zones, and frame-global regions that generate motion but
carry no information -- the burned-in timestamp, reflective glass, fans -- are
masked out before any statistic is computed.

Everything here rasterises down to the 16x16 block grid the codec motion-vector
field uses, because that is what the tier-1 scan actually consumes.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import db

# The motion-vector grid. H.264 may use finer partitions, so vectors are binned
# into this grid by destination position rather than assumed to align with it.
BLOCK = 16

ZONE_KINDS = ("screen", "desk", "torso")
MASK_KINDS = ("overlay", "reflection", "environment", "staff")

Point = tuple[float, float]


@dataclass
class Seat:
    label: str
    polygon: list[Point]
    zones: dict[str, list[Point]] = field(default_factory=dict)
    desk_line_y: float | None = None

    def validate(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"seat {self.label}: polygon needs at least 3 points")
        for kind in self.zones:
            if kind not in ZONE_KINDS:
                raise ValueError(f"seat {self.label}: unknown zone '{kind}'")


@dataclass
class MaskRegion:
    kind: str
    polygon: list[Point]
    note: str | None = None

    def validate(self) -> None:
        if self.kind not in MASK_KINDS:
            raise ValueError(f"unknown mask kind '{self.kind}'")
        if len(self.polygon) < 3:
            raise ValueError(f"{self.kind} mask: polygon needs at least 3 points")


@dataclass
class Calibration:
    camera_id: int
    frame_w: int
    frame_h: int
    version: int = 1
    seats: list[Seat] = field(default_factory=list)
    masks: list[MaskRegion] = field(default_factory=list)
    homography: list[list[float]] | None = None
    validated: bool = False

    def validate(self) -> None:
        if not self.seats:
            raise ValueError("calibration has no seats")
        labels = [s.label for s in self.seats]
        if len(labels) != len(set(labels)):
            raise ValueError("duplicate seat labels")
        for seat in self.seats:
            seat.validate()
        for mask in self.masks:
            mask.validate()

    # ------------------------------------------------------------ transfer --

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "version": self.version,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "homography": self.homography,
            "validated": self.validated,
            "seats": [
                {
                    "label": s.label,
                    "polygon": s.polygon,
                    "zones": s.zones,
                    "desk_line_y": s.desk_line_y,
                }
                for s in self.seats
            ],
            "masks": [
                {"kind": m.kind, "polygon": m.polygon, "note": m.note}
                for m in self.masks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        cal = cls(
            camera_id=data["camera_id"],
            frame_w=data["frame_w"],
            frame_h=data["frame_h"],
            version=data.get("version", 1),
            homography=data.get("homography"),
            validated=data.get("validated", False),
            seats=[
                Seat(
                    label=s["label"],
                    polygon=[tuple(p) for p in s["polygon"]],
                    zones={
                        k: [tuple(p) for p in v] for k, v in s.get("zones", {}).items()
                    },
                    desk_line_y=s.get("desk_line_y"),
                )
                for s in data["seats"]
            ],
            masks=[
                MaskRegion(
                    kind=m["kind"],
                    polygon=[tuple(p) for p in m["polygon"]],
                    note=m.get("note"),
                )
                for m in data.get("masks", [])
            ],
        )
        cal.validate()
        return cal

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------ rasterisation --


def _fill(polygon: list[Point], w: int, h: int) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas


def _to_blocks(pixel_mask: np.ndarray, block: int, coverage: float) -> np.ndarray:
    """Downsample a pixel mask to the block grid.

    A block belongs to a region when at least `coverage` of its pixels do. This
    keeps a seat polygon from claiming blocks it barely clips.
    """
    h, w = pixel_mask.shape
    bh, bw = h // block, w // block
    trimmed = pixel_mask[: bh * block, : bw * block]
    pooled = trimmed.reshape(bh, block, bw, block).mean(axis=(1, 3))
    return pooled >= coverage


@dataclass
class BlockMasks:
    """Block-grid masks consumed by the tier-1 motion scan."""

    block: int
    grid_h: int
    grid_w: int
    # True where motion must be ignored entirely: overlays, reflections,
    # environmental motion, and every seat's screen.
    excluded: np.ndarray
    # (seat_label, zone) -> boolean block mask, already excluding `excluded`.
    zones: dict[tuple[str, str], np.ndarray]
    # seat_label -> block row index of the desk line, or None.
    desk_line_row: dict[str, int | None]

    def zone_block_count(self, seat_label: str, zone: str) -> int:
        return int(self.zones[(seat_label, zone)].sum())


def rasterize(cal: Calibration, block: int = BLOCK, coverage: float = 0.5) -> BlockMasks:
    """Turn polygons into block-grid masks.

    Screen zones are folded into `excluded` rather than kept as a scoreable
    zone: a live monitor is an aperiodic motion source that the periodicity
    filter cannot suppress, so it is removed before statistics, not after.
    """
    cal.validate()
    w, h = cal.frame_w, cal.frame_h

    excluded_px = np.zeros((h, w), dtype=np.uint8)
    for mask in cal.masks:
        excluded_px |= _fill(mask.polygon, w, h)
    for seat in cal.seats:
        if "screen" in seat.zones:
            excluded_px |= _fill(seat.zones["screen"], w, h)

    excluded = _to_blocks(excluded_px, block, coverage)
    grid_h, grid_w = excluded.shape

    zones: dict[tuple[str, str], np.ndarray] = {}
    desk_line_row: dict[str, int | None] = {}
    for seat in cal.seats:
        for kind in ("desk", "torso"):
            polygon = seat.zones.get(kind)
            if polygon is None:
                # No explicit zone: fall back to the full seat footprint so a
                # partially calibrated seat still produces statistics.
                polygon = seat.polygon
            blocks = _to_blocks(_fill(polygon, w, h), block, coverage)
            zones[(seat.label, kind)] = blocks & ~excluded
        desk_line_row[seat.label] = (
            int(seat.desk_line_y // block) if seat.desk_line_y is not None else None
        )

    return BlockMasks(
        block=block,
        grid_h=grid_h,
        grid_w=grid_w,
        excluded=excluded,
        zones=zones,
        desk_line_row=desk_line_row,
    )


# -------------------------------------------------------------- persistence --


def save(conn: sqlite3.Connection, cal: Calibration) -> int:
    """Persist a calibration as a new version. Never overwrites a validated one."""
    cal.validate()
    row = db.one(
        conn,
        "SELECT COALESCE(MAX(version), 0) v FROM calibration WHERE camera_id = ?",
        (cal.camera_id,),
    )
    cal.version = int(row["v"]) + 1

    cal_id = db.insert(
        conn,
        "calibration",
        camera_id=cal.camera_id,
        version=cal.version,
        frame_w=cal.frame_w,
        frame_h=cal.frame_h,
        homography=json.dumps(cal.homography) if cal.homography else None,
        validated=int(cal.validated),
    )
    for seat in cal.seats:
        seat_id = db.insert(
            conn,
            "seat",
            calibration_id=cal_id,
            seat_label=seat.label,
            polygon=json.dumps(seat.polygon),
            desk_line_y=seat.desk_line_y,
        )
        for kind, polygon in seat.zones.items():
            db.insert(
                conn,
                "seat_zone",
                seat_id=seat_id,
                kind=kind,
                polygon=json.dumps(polygon),
            )
    for mask in cal.masks:
        db.insert(
            conn,
            "mask_region",
            calibration_id=cal_id,
            kind=mask.kind,
            polygon=json.dumps(mask.polygon),
            note=mask.note,
        )
    db.audit(conn, "save", "calibration", cal_id, detail=f"v{cal.version}")
    return cal_id


def load(conn: sqlite3.Connection, calibration_id: int) -> Calibration:
    head = db.one(conn, "SELECT * FROM calibration WHERE id = ?", (calibration_id,))
    if head is None:
        raise KeyError(f"no calibration {calibration_id}")

    seats: list[Seat] = []
    for srow in db.all_rows(
        conn, "SELECT * FROM seat WHERE calibration_id = ? ORDER BY id", (calibration_id,)
    ):
        zones = {
            z["kind"]: [tuple(p) for p in json.loads(z["polygon"])]
            for z in db.all_rows(
                conn, "SELECT * FROM seat_zone WHERE seat_id = ?", (srow["id"],)
            )
        }
        seats.append(
            Seat(
                label=srow["seat_label"],
                polygon=[tuple(p) for p in json.loads(srow["polygon"])],
                zones=zones,
                desk_line_y=srow["desk_line_y"],
            )
        )

    masks = [
        MaskRegion(
            kind=m["kind"],
            polygon=[tuple(p) for p in json.loads(m["polygon"])],
            note=m["note"],
        )
        for m in db.all_rows(
            conn, "SELECT * FROM mask_region WHERE calibration_id = ?", (calibration_id,)
        )
    ]

    return Calibration(
        camera_id=head["camera_id"],
        frame_w=head["frame_w"],
        frame_h=head["frame_h"],
        version=head["version"],
        seats=seats,
        masks=masks,
        homography=json.loads(head["homography"]) if head["homography"] else None,
        validated=bool(head["validated"]),
    )


def latest_for_camera(conn: sqlite3.Connection, camera_id: int) -> Calibration:
    row = db.one(
        conn,
        "SELECT id FROM calibration WHERE camera_id = ? ORDER BY version DESC LIMIT 1",
        (camera_id,),
    )
    if row is None:
        raise KeyError(f"camera {camera_id} has no calibration")
    return load(conn, int(row["id"]))


def seat_ids(conn: sqlite3.Connection, calibration_id: int) -> dict[str, int]:
    """Map seat label to database id, for writing motion windows."""
    return {
        r["seat_label"]: int(r["id"])
        for r in db.all_rows(
            conn,
            "SELECT id, seat_label FROM seat WHERE calibration_id = ?",
            (calibration_id,),
        )
    }
