"""Seat-crop extraction with supersampling.

PRD 3.4: a phone measures roughly 30x20 px at a near seat in this footage.
Feeding the whole 1280x720 frame to a 640 detector throws away half of that
before inference starts. Slicing the frame (the standard SAHI approach) buys
little here because the frame is barely 2x the network input to begin with --
those published gains come from aerial imagery where the frame is 20x larger.

So the crop goes the other way: take the seat region at native resolution and
*upscale* it into the detector, which is genuine supersampling. A 200x200 crop
into a 640 input is 3.2x.

Every crop carries the transform that produced it, so detector boxes map back to
frame coordinates exactly and evidence stays traceable to the source pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropTransform:
    """Maps between frame coordinates and detector-input coordinates."""

    x0: int
    y0: int
    crop_w: int
    crop_h: int
    scale: float
    pad_x: int
    pad_y: int
    input_size: int

    @property
    def supersampling(self) -> float:
        """How much detail the detector gains over full-frame inference.

        Above 1.0 the crop is magnified; at or below 1.0 it is shrunk and small
        objects lose pixels.
        """
        return self.scale

    def to_frame(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Detector-input box -> frame box."""
        x1, y1, x2, y2 = box
        return (
            (x1 - self.pad_x) / self.scale + self.x0,
            (y1 - self.pad_y) / self.scale + self.y0,
            (x2 - self.pad_x) / self.scale + self.x0,
            (y2 - self.pad_y) / self.scale + self.y0,
        )

    def to_input(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Frame box -> detector-input box."""
        x1, y1, x2, y2 = box
        return (
            (x1 - self.x0) * self.scale + self.pad_x,
            (y1 - self.y0) * self.scale + self.pad_y,
            (x2 - self.x0) * self.scale + self.pad_x,
            (y2 - self.y0) * self.scale + self.pad_y,
        )

    def frame_area(self, input_area: float) -> float:
        """Convert an area measured in input pixels back to frame pixels.

        The abstention rule is defined on the *source* footprint, so an object
        must never look bigger simply because the crop was magnified.
        """
        return input_area / (self.scale ** 2)


def polygon_bounds(
    polygon: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
    context_px: int = 0,
) -> tuple[int, int, int, int]:
    """Axis-aligned bounds of a polygon, padded and clamped to the frame."""
    pts = np.asarray(polygon, dtype=np.float32)
    x1 = int(max(np.floor(pts[:, 0].min()) - context_px, 0))
    y1 = int(max(np.floor(pts[:, 1].min()) - context_px, 0))
    x2 = int(min(np.ceil(pts[:, 0].max()) + context_px, frame_w))
    y2 = int(min(np.ceil(pts[:, 1].max()) + context_px, frame_h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("polygon has no area inside the frame")
    return x1, y1, x2, y2


def extract(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    input_size: int,
    min_crop_px: int = 96,
) -> tuple[np.ndarray, CropTransform]:
    """Cut a region and letterbox it into a square detector input.

    Aspect ratio is preserved -- stretching a seat crop would distort the very
    geometry used to tell a phone from a folded sheet.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, fw), min(y2, fh)

    # A crop far smaller than the input would be magnified enormously, which
    # invents detail rather than revealing it. Widen it instead.
    if x2 - x1 < min_crop_px:
        cx = (x1 + x2) // 2
        x1 = max(cx - min_crop_px // 2, 0)
        x2 = min(x1 + min_crop_px, fw)
        x1 = max(x2 - min_crop_px, 0)
    if y2 - y1 < min_crop_px:
        cy = (y1 + y2) // 2
        y1 = max(cy - min_crop_px // 2, 0)
        y2 = min(y1 + min_crop_px, fh)
        y1 = max(y2 - min_crop_px, 0)

    crop = frame[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    scale = min(input_size / crop_w, input_size / crop_h)
    new_w, new_h = max(int(round(crop_w * scale)), 1), max(int(round(crop_h * scale)), 1)

    # INTER_CUBIC when magnifying, INTER_AREA when shrinking.
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((input_size, input_size, frame.shape[2]), dtype=frame.dtype)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, CropTransform(
        x0=x1, y0=y1, crop_w=crop_w, crop_h=crop_h,
        scale=scale, pad_x=pad_x, pad_y=pad_y, input_size=input_size,
    )


def seat_crop(
    frame: np.ndarray,
    polygon: list[tuple[float, float]],
    input_size: int,
    min_crop_px: int = 96,
    context_px: int = 24,
) -> tuple[np.ndarray, CropTransform]:
    """Extract one seat's region, ready for the detector."""
    fh, fw = frame.shape[:2]
    bounds = polygon_bounds(polygon, fw, fh, context_px)
    return extract(frame, bounds, input_size, min_crop_px)
