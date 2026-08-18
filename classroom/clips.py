"""Evidence clip and thumbnail extraction.

Clips are cut from the source by seeking on presentation timestamps, never by
frame index, and the true source timestamp is preserved on every artefact so an
exported clip can always be traced back to the immutable original (FR-21).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import av
import cv2
import numpy as np

from . import db
from .calibration import Calibration
from .config import SegmentConfig


def _seek(container: av.container.InputContainer, stream, t: float) -> None:
    # Seeking lands on the nearest preceding keyframe; decoding then runs
    # forward to the requested timestamp.
    container.seek(max(int(t / float(stream.time_base)), 0), stream=stream)


def extract_clip(
    source: str | Path,
    out_path: str | Path,
    t_start: float,
    t_end: float,
    pre_roll: float = 3.0,
    post_roll: float = 3.0,
) -> tuple[Path, float, float]:
    """Write an evidence clip. Returns (path, actual_start, actual_end)."""
    source, out_path = Path(source), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    begin = max(t_start - pre_roll, 0.0)
    finish = t_end + post_roll

    with av.open(str(source)) as src:
        in_stream = src.streams.video[0]
        in_stream.thread_type = "AUTO"
        rate = in_stream.average_rate or 25

        with av.open(str(out_path), mode="w") as dst:
            out_stream = dst.add_stream("libx264", rate=rate)
            out_stream.width = in_stream.codec_context.width
            out_stream.height = in_stream.codec_context.height
            out_stream.pix_fmt = "yuv420p"

            _seek(src, in_stream, begin)
            tb = float(in_stream.time_base)
            first = last = None

            for frame in src.decode(in_stream):
                if frame.pts is None:
                    continue
                t = frame.pts * tb
                if t < begin:
                    continue
                if t > finish:
                    break
                if first is None:
                    first = t
                last = t
                out_frame = av.VideoFrame.from_ndarray(
                    frame.to_ndarray(format="rgb24"), format="rgb24"
                )
                for packet in out_stream.encode(out_frame):
                    dst.mux(packet)

            for packet in out_stream.encode():
                dst.mux(packet)

    return out_path, (first if first is not None else begin), (last if last is not None else finish)


def extract_thumbnail(
    source: str | Path,
    out_path: str | Path,
    t: float,
    cal: Calibration | None = None,
    seat_label: str | None = None,
) -> Path:
    """Grab one frame, optionally outlining the seat the event belongs to."""
    source, out_path = Path(source), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        _seek(container, stream, t)
        tb = float(stream.time_base)
        image = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if frame.pts * tb >= t or image is None:
                image = frame.to_ndarray(format="bgr24")
                if frame.pts * tb >= t:
                    break
        if image is None:
            raise RuntimeError(f"no frame decodable at {t:.2f}s in {source.name}")

    if cal is not None and seat_label is not None:
        seat = next((s for s in cal.seats if s.label == seat_label), None)
        if seat is not None:
            pts = np.asarray(seat.polygon, np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], True, (60, 210, 255), 2, cv2.LINE_AA)
            x, y = np.asarray(seat.polygon, np.int32).min(axis=0)
            cv2.putText(image, f"seat {seat_label}", (int(x), max(int(y) - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 210, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def materialise(
    conn: sqlite3.Connection,
    source_video_id: int,
    media_root: str | Path,
    cal: Calibration,
    cfg: SegmentConfig,
    limit: int | None = None,
    verbose: bool = True,
) -> int:
    """Generate clips and thumbnails for events that do not have them yet."""
    media_root = Path(media_root)
    video = db.one(conn, "SELECT * FROM source_video WHERE id = ?", (source_video_id,))
    if video is None:
        raise KeyError(f"no source_video {source_video_id}")

    rows = db.all_rows(
        conn,
        """
        SELECT e.id, e.t_start, e.t_end, e.peak_t, s.seat_label
        FROM event e JOIN seat s ON s.id = e.seat_id
        WHERE e.source_video_id = ? AND (e.clip_path IS NULL OR e.thumb_path IS NULL)
        ORDER BY e.peak_deviation DESC
        """,
        (source_video_id,),
    )
    if limit is not None:
        rows = rows[:limit]

    out_dir = media_root / f"video_{source_video_id}"
    made = 0
    for row in rows:
        clip_path = out_dir / f"event_{row['id']}.mp4"
        thumb_path = out_dir / f"event_{row['id']}.jpg"
        extract_clip(
            video["path"], clip_path, row["t_start"], row["t_end"],
            cfg.pre_roll_s, cfg.post_roll_s,
        )
        extract_thumbnail(
            video["path"], thumb_path, row["peak_t"], cal, row["seat_label"]
        )
        conn.execute(
            "UPDATE event SET clip_path = ?, thumb_path = ? WHERE id = ?",
            (str(clip_path), str(thumb_path), row["id"]),
        )
        made += 1
        if verbose:
            print(f"    event {row['id']}  seat {row['seat_label']}  "
                  f"{row['t_start']:.2f}-{row['t_end']:.2f}s")

    db.audit(conn, "materialise", "source_video", source_video_id,
             detail=f"{made} clips")
    return made
