"""Synthetic source video with known ground truth.

Used to validate the ingest and motion stages without depending on the real
corpus. Also the basis for the long-form test recording described in PRD 17.1:
real incident clips concatenated into quiet footage with recorded insertion
timestamps, so event recall can be measured before continuous footage arrives.
"""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def make_video(
    path: str | Path,
    *,
    seconds: float = 4.0,
    fps: int = 25,
    width: int = 320,
    height: int = 240,
    codec: str = "libx264",
    moving_box: bool = True,
    gop: int = 25,
) -> Path:
    """Write a short clip with a box moving across a static gradient.

    The moving box guarantees non-trivial codec motion vectors; the static
    background guarantees most blocks are still, which is what the tier-1 gate
    has to be able to tell apart.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(seconds * fps))

    # Static background: a horizontal gradient, textured enough that the encoder
    # produces real residuals rather than collapsing to flat blocks.
    ramp = np.linspace(30, 200, width, dtype=np.float32)
    background = np.repeat(ramp[None, :], height, axis=0)
    background = background + np.linspace(0, 25, height, dtype=np.float32)[:, None]
    background = np.clip(background, 0, 255).astype(np.uint8)
    base = np.dstack([background] * 3)

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.gop_size = gop

        for i in range(n_frames):
            img = base.copy()
            if moving_box:
                bw, bh = 40, 40
                x = int((width - bw) * (i / max(n_frames - 1, 1)))
                y = height // 2 - bh // 2
                img[y : y + bh, x : x + bw] = (250, 40, 40)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return path


# Layout of the scene fixture, in pixels on a 640x480 frame. Mirrors the
# structure of the real CBT footage: a per-frame-changing overlay in the
# corner, two seats side by side, activity confined to one of them.
SCENE = {
    "size": (640, 480),
    "overlay": (10, 10, 210, 60),
    "seat_a": (40, 190, 260, 330),
    "seat_b": (360, 190, 580, 330),
}


def make_scene_video(
    path: str | Path,
    *,
    seconds: float = 8.0,
    fps: int = 25,
    event_window: tuple[float, float] = (3.0, 5.0),
    codec: str = "libx264",
) -> Path:
    """Two seats, a live overlay, and activity in seat A only.

    The overlay region changes every single frame, exactly like the burned-in
    date/time in the real corpus. If masking works, it contributes nothing; if
    it leaks, it dominates every window.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = SCENE["size"]
    n_frames = int(round(seconds * fps))
    rng = np.random.default_rng(7)

    ramp = np.linspace(40, 190, width, dtype=np.float32)
    background = np.repeat(ramp[None, :], height, axis=0)
    background += np.linspace(0, 30, height, dtype=np.float32)[:, None]
    # Static texture across the whole frame. A smooth gradient gives the encoder
    # almost no distinctive reference blocks, so when one region changes it
    # emits spurious motion vectors elsewhere -- which showed up as the idle
    # seat firing during the active seat's event. Real rooms are textured
    # everywhere and do not have this problem; the fixture did.
    background += rng.normal(0.0, 9.0, background.shape)
    base = np.dstack([np.clip(background, 0, 255).astype(np.uint8)] * 3)

    # Static desk furniture, textured for the same reason as the background:
    # a flat fill inside the seat gives the encoder nothing to reference and
    # suppresses the very vectors the gate is meant to read.
    for key in ("seat_a", "seat_b"):
        x1, y1, x2, y2 = SCENE[key]
        desk = np.full((y2 - y1, x2 - x1, 3), (200, 200, 195), dtype=np.float32)
        desk += rng.normal(0.0, 9.0, desk.shape)
        base[y1:y2, x1:x2] = np.clip(desk, 0, 255).astype(np.uint8)
        strip = np.full((30, x2 - x1 - 20, 3), (60, 60, 65), dtype=np.float32)
        strip += rng.normal(0.0, 9.0, strip.shape)
        base[y1 + 10 : y1 + 40, x1 + 10 : x2 - 10] = np.clip(strip, 0, 255).astype(np.uint8)

    ev_start, ev_end = event_window

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.gop_size = fps

        for i in range(n_frames):
            t = i / fps
            img = base.copy()

            # Overlay: fresh noise every frame, guaranteeing motion vectors.
            ox1, oy1, ox2, oy2 = SCENE["overlay"]
            img[oy1:oy2, ox1:ox2] = rng.integers(
                0, 255, (oy2 - oy1, ox2 - ox1, 3), dtype=np.uint8
            )

            # Seat A: a hand-sized block sweeping downward during the event.
            if ev_start <= t < ev_end:
                x1, y1, x2, y2 = SCENE["seat_a"]
                span = (t - ev_start) / max(ev_end - ev_start, 1e-6)
                # Contrast is deliberately moderate. An earlier version used a
                # near-black box on a near-white desk -- a ~170-level step --
                # and H.264 propagated motion vectors from it well into the
                # neighbouring seat, firing the idle seat during the active
                # seat's event. Real CCTV of people in clothing produces no
                # edges that extreme, so that artefact belonged to the fixture
                # rather than the pipeline. Confirmed against real footage in
                # test_realworld_pipeline, where disjoint seats stay clean.
                # Sweep back and forth rather than drifting once: a single slow
                # traverse moves ~1 px/frame, under the block motion threshold
                # and slower than real hand movement.
                swing = 0.5 * (1.0 - np.cos(2 * np.pi * 3 * span))
                bw, bh = 46, 34
                bx = x1 + 30
                by = int(y1 + 45 + swing * (y2 - y1 - bh - 50))
                img[by : by + bh, bx : bx + bw] = (120, 118, 124)

            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return path
