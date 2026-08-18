"""Phase 1 - ingest and validation.

Reads container and stream properties, measures the real frame count and PTS
behaviour by demuxing, and confirms codec motion vectors are available by
decoding a short prefix.

Two things this module refuses to do, because both silently corrupt every
downstream timestamp:

  * trust the container's frame count (three of six profiled files report 0)
  * derive time from frame indices (three of six files are variable frame rate)
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

import av
import numpy as np

from . import db

# Frames decoded when checking motion-vector availability. Enough to cross at
# least one GOP boundary on every profiled file (longest observed GOP: 50).
MV_PROBE_FRAMES = 150

# A stream is variable frame rate when the standard deviation of its PTS deltas
# exceeds this fraction of the median delta.
VFR_TOLERANCE = 0.15


@dataclass
class VideoProfile:
    path: str
    sha256: str
    size_bytes: int
    container: str
    codec: str
    pix_fmt: str
    width: int
    height: int
    avg_fps: float
    duration_s: float
    bit_rate_mbps: float | None
    frame_count: int
    is_vfr: bool
    pts_delta_med: float
    pts_delta_std: float
    has_mv: bool
    mv_per_frame: float

    def summary(self) -> str:
        vfr = "VFR" if self.is_vfr else "CFR"
        mv = f"{self.mv_per_frame:.0f} mv/frame" if self.has_mv else "NO MOTION VECTORS"
        return (
            f"{Path(self.path).name}\n"
            f"  {self.container} / {self.codec} / {self.pix_fmt}  "
            f"{self.width}x{self.height} @ {self.avg_fps:.2f} fps  {vfr}\n"
            f"  {self.duration_s:.2f} s  {self.frame_count} frames  "
            f"{self.bit_rate_mbps:.2f} Mbps  {mv}"
        )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _timing(path: Path) -> tuple[int, float, float, float, bool]:
    """Demux (no decode) to count frames and characterise PTS behaviour.

    Demuxing is far cheaper than decoding and gives an exact packet count plus
    real presentation timestamps. Returns
    (count, duration_s, delta_median, delta_std, is_vfr).
    """
    pts_s: list[float] = []
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            count += 1
            pts_s.append(packet.pts * tb)

    if count < 2:
        raise ValueError(f"{path.name}: fewer than 2 timestamped packets")

    arr = np.sort(np.asarray(pts_s, dtype=np.float64))
    deltas = np.diff(arr)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        raise ValueError(f"{path.name}: all packets share one timestamp")

    med = float(np.median(deltas))
    std = float(np.std(deltas))
    duration = float(arr[-1] - arr[0] + med)
    return count, duration, med, std, bool(std > VFR_TOLERANCE * med)


def _motion_vectors(path: Path) -> tuple[bool, float]:
    """Decode a prefix and confirm codec motion vectors are exposed."""
    counts: list[int] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.options = {"flags2": "+export_mvs"}
        for index, frame in enumerate(container.decode(stream)):
            side = frame.side_data.get("MOTION_VECTORS")
            if side is not None:
                counts.append(len(side.to_ndarray()))
            if index + 1 >= MV_PROBE_FRAMES:
                break
    if not counts:
        return False, 0.0
    return True, float(np.median(counts))


def probe(path: str | Path) -> VideoProfile:
    """Measure a source file. Raises on anything that would corrupt timing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path.name}: no video stream")
        stream = container.streams.video[0]
        ctx = stream.codec_context
        container_name = container.format.name
        codec = ctx.name
        pix_fmt = ctx.pix_fmt
        width, height = ctx.width, ctx.height
        bit_rate = container.bit_rate

    count, duration, med, std, is_vfr = _timing(path)
    has_mv, mv_per_frame = _motion_vectors(path)
    size = path.stat().st_size

    return VideoProfile(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=size,
        container=container_name,
        codec=codec,
        pix_fmt=pix_fmt,
        width=width,
        height=height,
        avg_fps=count / duration if duration > 0 else 0.0,
        duration_s=duration,
        bit_rate_mbps=(bit_rate / 1e6) if bit_rate else (size * 8 / duration / 1e6),
        frame_count=count,
        is_vfr=is_vfr,
        pts_delta_med=med,
        pts_delta_std=std,
        has_mv=has_mv,
        mv_per_frame=mv_per_frame,
    )


def register(
    conn: sqlite3.Connection,
    profile: VideoProfile,
    session_id: int,
    camera_label: str,
    room: str | None = None,
) -> int:
    """Persist a probed video. Idempotent on content hash."""
    existing = db.one(
        conn, "SELECT id FROM source_video WHERE sha256 = ?", (profile.sha256,)
    )
    if existing is not None:
        return int(existing["id"])

    camera = db.one(
        conn,
        "SELECT id FROM camera WHERE session_id = ? AND label = ?",
        (session_id, camera_label),
    )
    camera_id = (
        int(camera["id"])
        if camera is not None
        else db.insert(
            conn, "camera", session_id=session_id, label=camera_label, room=room
        )
    )

    fields = asdict(profile)
    fields["is_vfr"] = int(profile.is_vfr)
    fields["has_mv"] = int(profile.has_mv)
    video_id = db.insert(conn, "source_video", camera_id=camera_id, **fields)
    db.audit(conn, "ingest", "source_video", video_id, detail=profile.path)
    return video_id
