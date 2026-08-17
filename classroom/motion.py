"""Phase 3 - tier-1 whole-video motion scan.

Reads codec motion vectors straight out of the bitstream, bins them onto a
16x16 block grid, compensates whole-frame motion, and reduces each analysis
window to per-seat-zone robust statistics.

Two details that are easy to get wrong and silently poison every number
downstream:

  * `motion_x` is in 1/`motion_scale` pixel units -- quarter-pel for H.264,
    half-pel for MPEG-4. Using it raw inflates magnitudes by 2-4x.
  * `dst_x`/`dst_y` are block *centres*, not corners, so binning by
    `dst // BLOCK` is correct even when the encoder uses sub-16 partitions.

Timing comes from presentation timestamps throughout. Three of the six profiled
source files are variable frame rate; frame-index arithmetic would drift.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from . import db
from .calibration import BlockMasks
from .config import BaselineConfig, MotionConfig

ZONES = ("desk", "torso")


@dataclass
class ZoneStats:
    mag_median: float
    mag_p90: float
    mag_p95: float
    area_ratio: float
    coherence: float
    residual: float
    below_desk: float


@dataclass
class Window:
    """One analysis window: whole-frame diagnostics plus per-seat-zone stats."""

    t_start: float
    t_end: float
    frames: int
    global_dx: float
    global_dy: float
    global_ratio: float
    luma_mean: float
    zones: dict[tuple[str, str], ZoneStats] = field(default_factory=dict)
    # Filled by apply_baselines.
    luma_delta: float = 0.0
    exposure_step: bool = False
    camera_moved: bool = False


# --------------------------------------------------------------- decoding --


def _luma_mean(frame: av.VideoFrame) -> float:
    """Mean of the Y plane, avoiding a full colourspace conversion."""
    plane = frame.planes[0]
    buf = np.frombuffer(plane, dtype=np.uint8)
    stride = plane.line_size
    height = frame.height
    if stride == frame.width:
        return float(buf[: frame.width * height].mean())
    usable = buf[: stride * height].reshape(height, stride)[:, : frame.width]
    return float(usable.mean())


def _accumulate(
    side_array: np.ndarray, grid_h: int, grid_w: int, block: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a frame's motion vectors onto the block grid.

    Returns (dx, dy) in pixels. Sub-16 partitions land in the same cell and are
    averaged, which is the behaviour we want: one magnitude per block.
    """
    scale = side_array["motion_scale"].astype(np.float32)
    scale[scale == 0] = 1.0
    dx = side_array["motion_x"].astype(np.float32) / scale
    dy = side_array["motion_y"].astype(np.float32) / scale

    col = np.clip(side_array["dst_x"].astype(np.int32) // block, 0, grid_w - 1)
    row = np.clip(side_array["dst_y"].astype(np.int32) // block, 0, grid_h - 1)
    flat = row * grid_w + col
    size = grid_h * grid_w

    counts = np.bincount(flat, minlength=size).astype(np.float32)
    sum_x = np.bincount(flat, weights=dx, minlength=size)
    sum_y = np.bincount(flat, weights=dy, minlength=size)
    nonzero = counts > 0
    sum_x[nonzero] /= counts[nonzero]
    sum_y[nonzero] /= counts[nonzero]
    return sum_x.reshape(grid_h, grid_w), sum_y.reshape(grid_h, grid_w)


def _zone_stats(
    residual: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    zone_mask: np.ndarray,
    threshold: float,
    desk_row: int | None,
) -> ZoneStats:
    values = residual[zone_mask]
    if values.size == 0:
        return ZoneStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    moving = values > threshold
    area_ratio = float(moving.mean())

    if moving.any():
        mx = dx[zone_mask][moving]
        my = dy[zone_mask][moving]
        norms = np.hypot(mx, my)
        norms[norms == 0] = 1.0
        # Resultant length of the unit direction vectors: 1.0 when every moving
        # block agrees on a direction, ~0 when they scatter. Coherent motion
        # across a whole zone usually means camera or lighting, not a person.
        coherence = float(
            np.hypot((mx / norms).mean(), (my / norms).mean())
        )
    else:
        coherence = 0.0

    below = 0.0
    if desk_row is not None and moving.any():
        rows = np.nonzero(zone_mask)[0][moving]
        below = float((rows >= desk_row).mean())

    return ZoneStats(
        mag_median=float(np.median(values)),
        mag_p90=float(np.percentile(values, 90)),
        mag_p95=float(np.percentile(values, 95)),
        area_ratio=area_ratio,
        coherence=coherence,
        residual=float(values.mean()),
        below_desk=below,
    )


def scan(
    video_path: str | Path,
    masks: BlockMasks,
    cfg: MotionConfig,
    progress_every: float = 0.0,
) -> Iterator[Window]:
    """Stream the whole video, yielding one Window per `cfg.window_s`."""
    path = Path(video_path)
    grid_h, grid_w, block = masks.grid_h, masks.grid_w, masks.block
    scoreable = ~masks.excluded
    scoreable_count = int(scoreable.sum())
    if scoreable_count == 0:
        raise ValueError("calibration excludes every block; nothing to score")

    acc_res = np.zeros((grid_h, grid_w), dtype=np.float64)
    acc_dx = np.zeros((grid_h, grid_w), dtype=np.float64)
    acc_dy = np.zeros((grid_h, grid_w), dtype=np.float64)
    frames = 0
    gdx_sum = gdy_sum = gratio_sum = 0.0
    luma_values: list[float] = []
    window_start: float | None = None
    next_report = progress_every

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.options = {"flags2": "+export_mvs"}
        time_base = float(stream.time_base)
        # Counted per window, not globally: a short trailing window must still
        # sample its own luma, or it reports zero and fabricates an exposure step.
        window_frame = 0

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = frame.pts * time_base
            if window_start is None:
                window_start = t

            if window_frame % cfg.luma_sample_stride == 0:
                luma_values.append(_luma_mean(frame))
            window_frame += 1

            side = frame.side_data.get("MOTION_VECTORS")
            if side is not None:
                dx, dy = _accumulate(side.to_ndarray(), grid_h, grid_w, block)

                # Whole-frame motion: the median over scoreable blocks is the
                # dominant translation. A static camera sits at zero.
                gdx = float(np.median(dx[scoreable]))
                gdy = float(np.median(dy[scoreable]))
                raw_mag = np.hypot(dx, dy)
                gratio = float((raw_mag[scoreable] > cfg.block_mag_threshold).mean())

                res = np.hypot(dx - gdx, dy - gdy)
                acc_res += res
                acc_dx += dx - gdx
                acc_dy += dy - gdy
                gdx_sum += gdx
                gdy_sum += gdy
                gratio_sum += gratio
                frames += 1

            if t - window_start >= cfg.window_s:
                if frames:
                    yield _emit(
                        window_start, t, frames, acc_res, acc_dx, acc_dy,
                        gdx_sum, gdy_sum, gratio_sum, luma_values, masks, cfg,
                    )
                acc_res[:] = 0.0
                acc_dx[:] = 0.0
                acc_dy[:] = 0.0
                gdx_sum = gdy_sum = gratio_sum = 0.0
                frames = 0
                luma_values = [_luma_mean(frame)]
                window_start = t
                window_frame = 1

                if progress_every and t >= next_report:
                    print(f"    scanned {t:8.1f}s", flush=True)
                    next_report = t + progress_every

        if frames and window_start is not None:
            yield _emit(
                window_start, window_start + cfg.window_s, frames,
                acc_res, acc_dx, acc_dy, gdx_sum, gdy_sum, gratio_sum,
                luma_values, masks, cfg,
            )


def _emit(
    t_start: float,
    t_end: float,
    frames: int,
    acc_res: np.ndarray,
    acc_dx: np.ndarray,
    acc_dy: np.ndarray,
    gdx_sum: float,
    gdy_sum: float,
    gratio_sum: float,
    luma_values: list[float],
    masks: BlockMasks,
    cfg: MotionConfig,
) -> Window:
    residual = acc_res / frames
    dx = acc_dx / frames
    dy = acc_dy / frames

    window = Window(
        t_start=t_start,
        t_end=t_end,
        frames=frames,
        global_dx=gdx_sum / frames,
        global_dy=gdy_sum / frames,
        global_ratio=gratio_sum / frames,
        luma_mean=float(np.mean(luma_values)) if luma_values else 0.0,
    )
    for (label, zone), zone_mask in masks.zones.items():
        if zone not in ZONES:
            continue
        window.zones[(label, zone)] = _zone_stats(
            residual, dx, dy, zone_mask, cfg.block_mag_threshold,
            masks.desk_line_row.get(label),
        )
    return window


# -------------------------------------------------------------- baselines --


@dataclass
class Baseline:
    """Robust centre and spread of one seat-zone's motion distribution."""

    median: float
    mad: float
    p95: float
    typing_median: float | None
    typing_mad: float | None
    sample_windows: int

    def deviation(self, value: float, cfg: BaselineConfig) -> tuple[float, str]:
        """Robust z-score against the regime the value itself falls in.

        In a CBT hall every occupied seat types, so 'deviation from still' is
        the wrong null hypothesis. A seat that is typing is compared against
        its typing distribution, not against silence.
        """
        idle_mad = max(self.mad, cfg.min_mad)
        idle_z = (value - self.median) / idle_mad

        if self.typing_median is None or idle_z < cfg.typing_z:
            return idle_z, "idle"

        typing_mad = max(self.typing_mad or cfg.min_mad, cfg.min_mad)
        return (value - self.typing_median) / typing_mad, "typing"


def learn_baselines(
    windows: list[Window], cfg: BaselineConfig
) -> dict[tuple[str, str], Baseline]:
    """Derive per-seat-zone baselines from the calibration span."""
    horizon = cfg.learn_s
    sample = [w for w in windows if w.t_start <= horizon] or windows

    baselines: dict[tuple[str, str], Baseline] = {}
    keys = {key for w in sample for key in w.zones}
    for key in keys:
        values = np.array(
            [w.zones[key].residual for w in sample if key in w.zones], dtype=np.float64
        )
        if values.size < cfg.min_windows:
            # Too little evidence for a trustworthy baseline; use what exists
            # but do not attempt to split regimes.
            median = float(np.median(values)) if values.size else 0.0
            mad = float(np.median(np.abs(values - median))) if values.size else 0.0
            baselines[key] = Baseline(median, mad, median, None, None, int(values.size))
            continue

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        p95 = float(np.percentile(values, 95))

        # Split the distribution: anything above idle + typing_z * MAD is the
        # active regime. A regime is only real when the seat spends a large
        # share of the span in it -- otherwise these are events, and folding
        # them into the baseline would teach it to ignore them.
        cut = median + cfg.typing_z * max(mad, cfg.min_mad)
        active = values[values > cut]
        if active.size / values.size >= cfg.min_regime_fraction:
            typing_median = float(np.median(active))
            typing_mad = float(np.median(np.abs(active - typing_median)))
        else:
            typing_median = typing_mad = None

        baselines[key] = Baseline(
            median, mad, p95, typing_median, typing_mad, int(values.size)
        )
    return baselines


@dataclass
class ScoredWindow:
    window: Window
    deviation: dict[tuple[str, str], float]
    regime: dict[tuple[str, str], str]


def apply_baselines(
    windows: list[Window],
    baselines: dict[tuple[str, str], Baseline],
    motion_cfg: MotionConfig,
    baseline_cfg: BaselineConfig,
) -> list[ScoredWindow]:
    """Attach deviation, regime, exposure and camera-motion flags."""
    scored: list[ScoredWindow] = []
    previous_luma: float | None = None

    for window in windows:
        if previous_luma is not None:
            window.luma_delta = window.luma_mean - previous_luma
            window.exposure_step = abs(window.luma_delta) > motion_cfg.exposure_step_delta
        previous_luma = window.luma_mean

        window.camera_moved = (
            window.global_ratio > motion_cfg.global_motion_ratio
            and np.hypot(window.global_dx, window.global_dy) > motion_cfg.global_motion_mag
        )

        deviation: dict[tuple[str, str], float] = {}
        regime: dict[tuple[str, str], str] = {}
        for key, stats in window.zones.items():
            baseline = baselines.get(key)
            if baseline is None:
                deviation[key] = 0.0
                regime[key] = "unknown"
                continue
            z, which = baseline.deviation(stats.residual, baseline_cfg)
            # A frame-wide exposure step or camera bump explains the motion
            # better than the candidate does. Suppress rather than escalate.
            if window.exposure_step or window.camera_moved:
                z = min(z, 0.0)
            deviation[key] = z
            regime[key] = which
        scored.append(ScoredWindow(window, deviation, regime))
    return scored


# ------------------------------------------------------------ persistence --


def persist(
    conn: sqlite3.Connection,
    source_video_id: int,
    seat_ids: dict[str, int],
    scored: list[ScoredWindow],
) -> int:
    """Write the scan output. Idempotent: replaces any prior scan of this video."""
    conn.execute("DELETE FROM motion_window WHERE source_video_id = ?", (source_video_id,))
    conn.execute("DELETE FROM frame_window WHERE source_video_id = ?", (source_video_id,))

    frame_rows = [
        (
            source_video_id, s.window.t_start, s.window.t_end,
            s.window.global_dx, s.window.global_dy, s.window.global_ratio,
            s.window.luma_mean, s.window.luma_delta,
            int(s.window.exposure_step), int(s.window.camera_moved),
        )
        for s in scored
    ]
    db.insert_many(
        conn, "frame_window",
        ["source_video_id", "t_start", "t_end", "global_dx", "global_dy",
         "global_ratio", "luma_mean", "luma_delta", "exposure_step", "camera_moved"],
        frame_rows,
    )

    motion_rows = []
    for s in scored:
        for (label, zone), stats in s.window.zones.items():
            seat_id = seat_ids.get(label)
            if seat_id is None:
                continue
            motion_rows.append((
                source_video_id, seat_id, zone, s.window.t_start, s.window.t_end,
                stats.mag_median, stats.mag_p90, stats.mag_p95, stats.area_ratio,
                stats.coherence, stats.residual, stats.below_desk,
                s.deviation[(label, zone)], s.regime[(label, zone)],
            ))
    db.insert_many(
        conn, "motion_window",
        ["source_video_id", "seat_id", "zone", "t_start", "t_end",
         "mag_median", "mag_p90", "mag_p95", "area_ratio", "coherence",
         "residual", "below_desk", "deviation", "regime"],
        motion_rows,
    )
    db.audit(conn, "scan", "source_video", source_video_id,
             detail=f"{len(scored)} windows, {len(motion_rows)} zone rows")
    return len(motion_rows)


def persist_baselines(
    conn: sqlite3.Connection,
    calibration_id: int,
    seat_ids: dict[str, int],
    baselines: dict[tuple[str, str], Baseline],
) -> None:
    conn.execute(
        "DELETE FROM seat_baseline WHERE calibration_id = ?", (calibration_id,)
    )
    rows = []
    for (label, zone), baseline in baselines.items():
        seat_id = seat_ids.get(label)
        if seat_id is None:
            continue
        rows.append((
            calibration_id, seat_id, zone, baseline.median, baseline.mad,
            baseline.p95, baseline.typing_median, baseline.typing_mad,
            baseline.sample_windows,
        ))
    db.insert_many(
        conn, "seat_baseline",
        ["calibration_id", "seat_id", "zone", "median", "mad", "p95",
         "typing_median", "typing_mad", "sample_windows"],
        rows,
    )
