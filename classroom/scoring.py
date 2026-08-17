"""Event priority scoring.

A ranking signal, not a probability of misconduct. Weights are hand-set and
documented rather than learned: the available event count cannot support fitting
them without overfitting, and a reviewer needs to see why an event was ranked
where it was.

Every contribution is stored alongside the total so the console can show the
breakdown per event (PRD 7.4).
"""
from __future__ import annotations

import json
import sqlite3

from . import db
from .config import ScoreConfig

# Factors from model evidence are absent until phases 6-7 run. They are listed
# here so the breakdown shown to a reviewer makes the absence explicit rather
# than silently omitting them.
MODEL_FACTORS = ("object", "head_pitch")


def score_event(
    peak_deviation: float,
    duration_s: float,
    window_count: int,
    below_desk_peak: float,
    camera_moved_fraction: float,
    exposure_step_fraction: float,
    cfg: ScoreConfig,
    object_evidence: float | None = None,
    head_pitch_evidence: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Return (score, per-factor contributions)."""
    factors: dict[str, float] = {}

    # Motion significance: the peak robust z, softly capped so one enormous
    # window cannot dominate the ranking on its own.
    factors["motion"] = cfg.w_motion * min(peak_deviation, 12.0) / 4.0

    # Persistence: sustained activity is more interesting than a brief spike.
    factors["persistence"] = cfg.w_persistence * min(duration_s / 10.0, 1.5)

    # Below-desk excursion: the dominant measurable pattern in this deployment.
    factors["below_desk"] = cfg.w_below_desk * below_desk_peak

    if object_evidence is not None:
        factors["object"] = cfg.w_object * object_evidence
    if head_pitch_evidence is not None:
        factors["head_pitch"] = cfg.w_head_pitch * head_pitch_evidence

    # Penalties: a frame-wide cause explains the motion better than the
    # candidate does.
    factors["global_motion"] = cfg.w_global_motion * camera_moved_fraction
    factors["environmental"] = cfg.w_environmental * exposure_step_fraction

    # Uncertainty: short events built from few windows are less trustworthy.
    if window_count < 4:
        factors["uncertainty"] = cfg.w_uncertainty * (4 - window_count) / 4.0

    return sum(factors.values()), factors


def score_video(
    conn: sqlite3.Connection, source_video_id: int, cfg: ScoreConfig
) -> int:
    """Score every event of one video from the evidence currently in the store."""
    events = db.all_rows(
        conn,
        "SELECT id, t_start, t_end, peak_deviation FROM event WHERE source_video_id = ?",
        (source_video_id,),
    )
    updated = 0
    for event in events:
        windows = db.all_rows(
            conn,
            """
            SELECT below_desk FROM motion_window
            WHERE source_video_id = ? AND t_start >= ? AND t_end <= ?
            """,
            (source_video_id, event["t_start"], event["t_end"]),
        )
        frames = db.all_rows(
            conn,
            """
            SELECT camera_moved, exposure_step FROM frame_window
            WHERE source_video_id = ? AND t_start >= ? AND t_end <= ?
            """,
            (source_video_id, event["t_start"], event["t_end"]),
        )
        below = max((w["below_desk"] for w in windows), default=0.0)
        moved = (
            sum(f["camera_moved"] for f in frames) / len(frames) if frames else 0.0
        )
        exposed = (
            sum(f["exposure_step"] for f in frames) / len(frames) if frames else 0.0
        )

        total, factors = score_event(
            peak_deviation=event["peak_deviation"],
            duration_s=event["t_end"] - event["t_start"],
            window_count=len(frames),
            below_desk_peak=below,
            camera_moved_fraction=moved,
            exposure_step_fraction=exposed,
            cfg=cfg,
        )
        missing = [f for f in MODEL_FACTORS if f not in factors]
        quality = "limited" if missing else "sufficient"

        conn.execute(
            "UPDATE event SET score = ?, score_factors = ?, evidence_quality = ?, "
            "explanation = ? WHERE id = ?",
            (total, json.dumps(factors), quality, describe(event, below), event["id"]),
        )
        updated += 1
    return updated


def describe(event: sqlite3.Row, below_desk_peak: float) -> str:
    """Observational sentence. No verdict, no intent, no accusation (FR-19)."""
    duration = event["t_end"] - event["t_start"]
    parts = [f"Sustained motion for {duration:.1f} s"]
    if below_desk_peak > 0.5:
        parts.append("predominantly below the desk line")
    elif below_desk_peak > 0.15:
        parts.append("partly below the desk line")
    parts.append(f"peaking at {event['peak_deviation']:.1f} standard deviations "
                 f"above this seat's baseline")
    return ", ".join(parts) + "."
