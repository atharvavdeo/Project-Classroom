"""Full-stream ingest and decode inventory (PRD 6.1).

Decodes the **complete** stream once, and from that single pass produces
everything the later stages need to trust their inputs:

  * a container and stream inventory — dimensions, codec, declared rate,
    bitrate, colour metadata, rotation;
  * a per-frame timestamp index with keyframes, repeated timestamps and gaps;
  * per-frame image statistics at a reduced scale, which is what the health
    conditions in `health.py` are computed from;
  * regular contact sheets, plus one for every integrity incident.

**Why statistics are gathered here and not in a second pass.** Health detection
needs mean luma, near-black and near-white fractions, sharpness and a temporal
hash for every frame. Decoding a 282-second 720p file costs about a minute;
doing it twice costs two. Both passes would need identical frame alignment
anyway, and any drift between them would be invisible and wrong.

**Why statistics are computed at reduced scale.** A 64x36 grey thumbnail per
frame is enough for luma distribution, saturation fraction and a difference
hash, and costs a fraction of full-resolution work. Sharpness is measured on a
larger 320-wide thumbnail because a Laplacian variance at 64 px wide is
dominated by resampling rather than by the scene. Both scales are recorded, so
a threshold can be re-derived rather than guessed at if they ever change.

Nothing here interpolates. An undecodable span is recorded as a gap and stays
visible; PRD 6.1 forbids inventing a frame to cover one.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np

from .artifacts import sha256_file
from .timestamps import TimestampIndex, build_index

# Statistics thumbnail. Small enough to be nearly free, large enough that the
# luma distribution and difference hash are stable.
STAT_W, STAT_H = 64, 36

# Sharpness thumbnail. Laplacian variance at 64 px wide measures the resampler,
# not the scene.
SHARP_W = 320

# Luma is 0-255. Near-black and near-white are deliberately generous: a
# blackout on CCTV is not mathematically zero, it is sensor floor plus noise.
NEAR_BLACK = 24
NEAR_WHITE = 232

# How many consecutive decode failures before abandoning a stream. A single
# corrupt packet is recoverable and the frames after it are worth keeping; a
# sustained run means the stream is unreadable from here and continuing only
# lengthens the error list.
MAX_CONSECUTIVE_DECODE_ERRORS = 32


@dataclass
class StreamInventory:
    """What the container claims, recorded so it can be contradicted."""

    path: str
    sha256: str
    size_bytes: int
    container: str
    codec: str
    pix_fmt: str
    width: int
    height: int
    declared_fps: float
    declared_duration_s: float
    bit_rate_mbps: float | None
    colour_primaries: str | None = None
    colour_transfer: str | None = None
    colour_space: str | None = None
    colour_range: str | None = None
    rotation: int = 0
    time_base: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameStats:
    """Per-frame measurements, at the scales documented above."""

    index: int
    pts_ms: float
    luma_mean: float
    luma_std: float
    near_black_fraction: float
    near_white_fraction: float
    sharpness: float
    dhash: int                  # 64-bit scene fingerprint, for reposition checks
    # Mean absolute difference against the previous frame's thumbnail, 0-255.
    # This is what freeze detection uses; see the note on _dhash below for why
    # the hash cannot do that job on a fixed camera.
    frame_delta: float = 0.0

    def to_row(self) -> dict:
        return {
            "index": self.index,
            "pts_ms": round(self.pts_ms, 3),
            "luma_mean": round(self.luma_mean, 3),
            "luma_std": round(self.luma_std, 3),
            "near_black_fraction": round(self.near_black_fraction, 5),
            "near_white_fraction": round(self.near_white_fraction, 5),
            "sharpness": round(self.sharpness, 4),
            "dhash": self.dhash,
            "frame_delta": round(self.frame_delta, 5),
        }


@dataclass
class IngestResult:
    inventory: StreamInventory
    index: TimestampIndex
    stats: list[FrameStats] = field(default_factory=list)

    @property
    def declared_vs_measured_fps(self) -> tuple[float, float]:
        return self.inventory.declared_fps, self.index.measured_fps

    @property
    def fps_declaration_trustworthy(self) -> bool:
        """Whether the container's declared rate matches what was decoded.

        Measured on this corpus: two of six files declare a rate that is wrong,
        both of them VFR. A declared rate is a hint, and this says whether the
        hint held.
        """
        declared, measured = self.declared_vs_measured_fps
        if declared <= 0 or measured <= 0:
            return False
        return abs(declared - measured) / measured < 0.02

    def summary(self) -> dict:
        return {
            **self.inventory.to_dict(),
            **self.index.summary(),
            "fps_declaration_trustworthy": self.fps_declaration_trustworthy,
            "stats_frames": len(self.stats),
        }


def _rotation(stream) -> int:
    for key in ("rotate", "rotation"):
        value = (stream.metadata or {}).get(key)
        if value:
            try:
                return int(float(value))
            except ValueError:
                pass
    return 0


def probe(path: str | Path) -> StreamInventory:
    """Container and stream metadata, without decoding the whole file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"source video not found: {path}")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        ctx = stream.codec_context
        declared_fps = float(stream.average_rate) if stream.average_rate else 0.0
        duration_s = float(stream.duration * stream.time_base) if stream.duration \
            else (float(container.duration) / av.time_base if container.duration else 0.0)
        bit_rate = container.bit_rate or stream.bit_rate
        return StreamInventory(
            path=str(path),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            container=container.format.name,
            codec=ctx.name,
            pix_fmt=str(ctx.pix_fmt or ""),
            width=ctx.width,
            height=ctx.height,
            declared_fps=declared_fps,
            declared_duration_s=duration_s,
            bit_rate_mbps=round(bit_rate / 1e6, 3) if bit_rate else None,
            colour_primaries=str(getattr(ctx, "color_primaries", "") or "") or None,
            colour_transfer=str(getattr(ctx, "color_trc", "") or "") or None,
            colour_space=str(getattr(ctx, "colorspace", "") or "") or None,
            colour_range=str(getattr(ctx, "color_range", "") or "") or None,
            rotation=_rotation(stream),
            time_base=str(stream.time_base),
        )


def _pict_type(frame) -> str:
    """Picture type as a name.

    PyAV 18 made `pict_type` an int enum whose `.name` is not always present,
    so this reads whichever form the installed version provides rather than
    assuming one.
    """
    pt = getattr(frame, "pict_type", None)
    if pt is None:
        return ""
    return getattr(pt, "name", None) or str(pt).split(".")[-1]


def _dhash(grey: np.ndarray) -> int:
    """64-bit difference hash: 9x8 grey, each pixel compared to its neighbour.

    A *scene* fingerprint. It answers "is this the same view", which is the
    right question for camera repositioning and the wrong one for freeze
    detection on a fixed camera.

    Measured, when this was used for freezes: on `04.CCTV Candidate Talking.mkv`
    it reported `frozen_video` over **99.6% of a file with people moving in it**.
    On a static mount the frame is dominated by unchanging furniture, so the
    hash is identical between consecutive frames whether or not anybody moved.
    Freeze detection uses `frame_delta` instead.
    """
    small = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def _frame_stats(index: int, pts_ms: float, plane: np.ndarray,
                 previous_small: np.ndarray | None = None
                 ) -> tuple[FrameStats, np.ndarray]:
    """Measure one frame from its luma plane, at the documented scales.

    Returns the statistics and the thumbnail, so the caller can pass it back in
    as `previous_small` for the temporal difference without keeping every
    thumbnail alive.
    """
    small = cv2.resize(plane, (STAT_W, STAT_H), interpolation=cv2.INTER_AREA)
    total = small.size
    sharp_h = max(int(plane.shape[0] * SHARP_W / max(plane.shape[1], 1)), 1)
    sharp = cv2.resize(plane, (SHARP_W, sharp_h), interpolation=cv2.INTER_AREA)
    delta = 0.0
    if previous_small is not None:
        delta = float(np.abs(small.astype(np.int16)
                             - previous_small.astype(np.int16)).mean())
    return FrameStats(
        index=index,
        pts_ms=pts_ms,
        luma_mean=float(small.mean()),
        luma_std=float(small.std()),
        near_black_fraction=float((small <= NEAR_BLACK).sum()) / total,
        near_white_fraction=float((small >= NEAR_WHITE).sum()) / total,
        sharpness=float(cv2.Laplacian(sharp, cv2.CV_64F).var()),
        dhash=_dhash(small),
        frame_delta=delta,
    ), small


def decode(path: str | Path, progress_every_s: float = 0.0,
           on_progress=None) -> IngestResult:
    """One complete decode pass: inventory, timestamp index, per-frame stats.

    Decode errors are collected rather than raised. PRD 6.1 requires undecodable
    spans recorded as failures with the gap left visible; aborting the run on the
    first bad packet would discard every good frame after it.
    """
    inventory = probe(path)
    records: list[tuple[float, bool, str]] = []
    stats: list[FrameStats] = []
    errors: list[str] = []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        next_report = progress_every_s

        # The generator is created ONCE, and this matters more than it looks.
        # An earlier version called `next(_frames(container, stream))` inside
        # the loop, constructing a fresh generator on every iteration. Frames
        # still came out in order, so it appeared to work -- and it lost the
        # last 11 frames of every file, because the decoder's internal buffer is
        # only flushed when a generator runs to completion, and a freshly
        # created one raised EOFError instead.
        #
        # Eleven frames of 1146 is under 1%, invisible in any summary, and a
        # direct violation of the first requirement in PRD 18: the complete
        # decodable duration is processed, or every missing range is explicitly
        # failed. It was neither.
        frames = container.decode(stream)
        previous_small: np.ndarray | None = None
        consecutive_errors = 0
        i = 0
        while True:
            try:
                frame = next(frames)
                consecutive_errors = 0
            except StopIteration:
                break
            except EOFError:
                # PyAV surfaces end-of-stream from avcodec_send_packet as
                # EOFError. That is normal termination, not a decode failure,
                # and recording it as one would put a spurious error on every
                # file that decoded perfectly.
                break
            except av.error.FFmpegError as exc:  # noqa: PERF203 - per-frame recovery
                # One bad packet must not discard every good frame after it.
                errors.append(f"frame {i}: {type(exc).__name__}: {exc}")
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_DECODE_ERRORS:
                    errors.append(
                        f"stopped after {consecutive_errors} consecutive decode "
                        f"errors at frame {i}")
                    break
                continue

            pts_ms = float(frame.pts * time_base * 1000.0) if frame.pts is not None \
                else (records[-1][0] if records else 0.0)
            records.append((pts_ms, bool(frame.key_frame), _pict_type(frame)))

            # The luma plane directly: no colourspace conversion, which is the
            # dominant per-frame cost and buys nothing for these measurements.
            plane = frame.to_ndarray(format="yuv420p")[: frame.height, : frame.width]
            measured, previous_small = _frame_stats(i, pts_ms, plane, previous_small)
            stats.append(measured)

            if progress_every_s and pts_ms / 1000.0 >= next_report:
                if on_progress:
                    on_progress(pts_ms / 1000.0, i)
                next_report += progress_every_s
            i += 1

    index = build_index(records)
    index.decode_errors = errors
    return IngestResult(inventory=inventory, index=index, stats=stats)



# --------------------------------------------------------------- artifacts --

def contact_sheet(path: str | Path, times_ms: list[float], columns: int = 4,
                  tile: int = 320) -> np.ndarray:
    """Tile frames at given timestamps, each labelled with its source PTS.

    Every tile carries its timestamp because PRD 6.1 requires that no generated
    image lacks source provenance. A contact sheet whose tiles cannot be located
    in the source is decoration.
    """
    if not times_ms:
        raise ValueError("no timestamps supplied for the contact sheet")
    frames: list[tuple[float, np.ndarray]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        for target_ms in times_ms:
            container.seek(int((target_ms / 1000.0) / time_base), stream=stream,
                           backward=True)
            picked = None
            for frame in container.decode(stream):
                t = float(frame.pts * time_base * 1000.0) if frame.pts is not None else 0.0
                if t >= target_ms - 1e-6:
                    picked = (t, frame.to_ndarray(format="bgr24"))
                    break
            if picked is not None:
                frames.append(picked)

    if not frames:
        raise ValueError(f"no frames could be decoded from {path}")

    rows = (len(frames) + columns - 1) // columns
    sheet = np.zeros((rows * tile, columns * tile, 3), dtype=np.uint8)
    for i, (pts_ms, image) in enumerate(frames):
        h, w = image.shape[:2]
        scale = min(tile / w, tile / h)
        rw, rh = max(int(w * scale), 1), max(int(h * scale), 1)
        r, c = divmod(i, columns)
        y, x = r * tile + (tile - rh) // 2, c * tile + (tile - rw) // 2
        sheet[y:y + rh, x:x + rw] = cv2.resize(image, (rw, rh),
                                               interpolation=cv2.INTER_AREA)
        cv2.rectangle(sheet, (c * tile, r * tile), (c * tile + tile, r * tile + 22),
                      (0, 0, 0), -1)
        cv2.putText(sheet, f"{pts_ms / 1000.0:9.3f}s", (c * tile + 6, r * tile + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 210, 255), 1, cv2.LINE_AA)
    return sheet


def regular_sheet_times(index: TimestampIndex, count: int = 16) -> list[float]:
    """Evenly spaced timestamps across the decodable span, avoiding gaps."""
    if index.count == 0:
        return []
    times = np.linspace(index.first_pts_ms, index.last_pts_ms, count)
    return [float(t) for t in times if index.inside_gap(float(t)) is None]


def stats_series(stats: list[FrameStats]) -> dict[str, list[float]]:
    """Per-frame statistics as parallel series, for the health detectors."""
    return {
        "pts_ms": [s.pts_ms for s in stats],
        "luma_mean": [s.luma_mean for s in stats],
        "luma_std": [s.luma_std for s in stats],
        "near_black": [s.near_black_fraction for s in stats],
        "near_white": [s.near_white_fraction for s in stats],
        "sharpness": [s.sharpness for s in stats],
        "frame_delta": [s.frame_delta for s in stats],
    }


def median_sharpness(stats: list[FrameStats]) -> float:
    """Per-camera sharpness baseline.

    PRD 6.2 defines blur relative to a per-camera baseline, not an absolute
    number, because sharpness varies by an order of magnitude between a
    well-lit near seat and a distant one. An absolute threshold calibrated on
    one camera means nothing on another.
    """
    values = [s.sharpness for s in stats]
    return statistics.median(values) if values else 0.0
