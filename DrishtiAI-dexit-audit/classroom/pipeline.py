"""Phase orchestration.

Runs ingest -> motion -> segment for one video against one calibration. Each
stage checkpoints to SQLite and replaces its own prior output, so re-running is
safe and resumes cleanly after an interruption.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from . import calibration as cal_mod, db, motion, segment
from .config import Config


@dataclass
class RunResult:
    job_id: int
    windows: int
    motion_rows: int
    events: int
    seconds: float
    candidate_fraction: float


def run(
    conn: sqlite3.Connection,
    source_video_id: int,
    calibration_id: int,
    cfg: Config,
    verbose: bool = True,
) -> RunResult:
    video = db.one(conn, "SELECT * FROM source_video WHERE id = ?", (source_video_id,))
    if video is None:
        raise KeyError(f"no source_video {source_video_id}")

    cal = cal_mod.load(conn, calibration_id)
    if (cal.frame_w, cal.frame_h) != (video["width"], video["height"]):
        raise ValueError(
            f"calibration is {cal.frame_w}x{cal.frame_h} but video is "
            f"{video['width']}x{video['height']}; seat polygons would be misplaced"
        )

    job_id = db.insert(
        conn, "job", source_video_id=source_video_id, calibration_id=calibration_id,
        stage="motion", status="running", started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    started = time.perf_counter()

    try:
        masks = cal_mod.rasterize(cal)
        seat_ids = cal_mod.seat_ids(conn, calibration_id)

        if verbose:
            scoreable = int((~masks.excluded).sum())
            total = masks.grid_h * masks.grid_w
            print(f"  grid {masks.grid_h}x{masks.grid_w} blocks, "
                  f"{scoreable}/{total} scoreable, {len(cal.seats)} seats")

        windows = list(motion.scan(
            video["path"], masks, cfg.motion,
            progress_every=60.0 if verbose else 0.0,
        ))
        if not windows:
            raise ValueError("scan produced no windows")

        baselines = motion.learn_baselines(windows, cfg.baseline)
        scored = motion.apply_baselines(windows, baselines, cfg.motion, cfg.baseline)

        motion.persist_baselines(conn, calibration_id, seat_ids, baselines)
        motion_rows = motion.persist(conn, source_video_id, seat_ids, scored)

        conn.execute("UPDATE job SET stage = 'segment' WHERE id = ?", (job_id,))
        events = segment.segment(scored, cfg.segment)
        event_count = segment.persist(conn, source_video_id, seat_ids, events)

        # The fraction of the recording that reaches the expensive stages. PRD
        # 14.4 requires this to be measured, never assumed.
        covered = sum(e.t_end - e.t_start for e in events)
        fraction = covered / max(video["duration_s"], 1e-9)

        elapsed = time.perf_counter() - started
        conn.execute(
            "UPDATE job SET stage = 'done', status = 'complete', progress = 1.0, "
            "finished_at = ? WHERE id = ?",
            (time.strftime("%Y-%m-%d %H:%M:%S"), job_id),
        )
        return RunResult(job_id, len(windows), motion_rows, event_count, elapsed, fraction)

    except Exception as exc:
        conn.execute(
            "UPDATE job SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
            (str(exc), time.strftime("%Y-%m-%d %H:%M:%S"), job_id),
        )
        raise
