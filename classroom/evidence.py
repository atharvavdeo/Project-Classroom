"""Phase 6 orchestration - model evidence on candidate events only.

Runs over the small fraction of the recording that segmentation flagged, never
over the whole video. Each event is sampled at a few timestamps; the seat crop
is cut at native resolution and magnified into the detector, pose is read from
the full frame so the estimator sees whole bodies, and both are reduced to the
scalar factors the priority score consumes.

A model that is not configured contributes no factor at all. It is not defaulted
to zero and not silently skipped: the event's evidence_quality records that the
stream was absent, so a reviewer can tell "no object was found" apart from "no
detector ran".
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from . import crops, db, detect, pose
from .calibration import Calibration
from .config import DetectConfig
from .detect import Detection, ONNXDetector
from .pose import PoseObservation, RTMOPose

SAMPLES_PER_EVENT = 6


@dataclass
class EventEvidence:
    event_id: int
    seat_label: str
    detections: list[Detection]
    poses: list[PoseObservation]
    object_evidence: float | None
    pitch_evidence: float | None
    below_desk_evidence: float | None

    @property
    def streams_present(self) -> list[str]:
        present = []
        if self.object_evidence is not None:
            present.append("object")
        if self.pitch_evidence is not None:
            present.append("pose")
        return present


def sample_times(t_start: float, t_end: float, count: int) -> list[float]:
    if count <= 1:
        return [(t_start + t_end) / 2.0]
    return list(np.linspace(t_start, t_end, count))


def read_frames(video_path: str | Path, times: list[float]) -> list[tuple[float, np.ndarray]]:
    """Decode the requested timestamps, seeking by PTS."""
    out: list[tuple[float, np.ndarray]] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        for t in times:
            container.seek(max(int(t / tb), 0), stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if frame.pts * tb >= t:
                    out.append((frame.pts * tb, frame.to_ndarray(format="bgr24")))
                    break
    return out


def analyse_event(
    video_path: str | Path,
    event_id: int,
    seat_label: str,
    polygon: list[tuple[float, float]],
    desk_line_y: float | None,
    t_start: float,
    t_end: float,
    cfg: DetectConfig,
    detector: ONNXDetector | None = None,
    pose_model: RTMOPose | None = None,
    samples: int = SAMPLES_PER_EVENT,
) -> EventEvidence:
    """Gather object and pose evidence for one event."""
    frames = read_frames(video_path, sample_times(t_start, t_end, samples))

    per_frame_detections: list[list[Detection]] = []
    observations: list[PoseObservation] = []

    for t, frame in frames:
        if detector is not None:
            crop, transform = crops.seat_crop(
                frame, polygon, cfg.input_size, cfg.min_crop_px
            )
            found = detector(crop, transform, cfg.confidence, t)
            per_frame_detections.append(
                detect.apply_abstention(found, cfg.min_object_px_area)
            )

        if pose_model is not None:
            for keypoints in pose_model(frame):
                observation = pose.observe(keypoints, desk_line_y)
                observation.seat_label = seat_label
                observations.append(observation)

    confirmed: list[Detection] = []
    if detector is not None:
        confirmed = detect.temporal_confirm(
            per_frame_detections, cfg.temporal_confirm_frames
        )
        # Abstained detections are dropped by temporal confirmation but must
        # still reach the reviewer: "we looked and could not tell" is evidence.
        confirmed += [
            d for frame_dets in per_frame_detections for d in frame_dets if d.abstained
        ]

    return EventEvidence(
        event_id=event_id,
        seat_label=seat_label,
        detections=confirmed,
        poses=observations,
        object_evidence=(
            detect.object_evidence(confirmed) if detector is not None else None
        ),
        pitch_evidence=(
            pose.pitch_evidence(observations) if pose_model is not None else None
        ),
        below_desk_evidence=(
            pose.below_desk_evidence(observations) if pose_model is not None else None
        ),
    )


def persist(conn: sqlite3.Connection, evidence: EventEvidence) -> None:
    """Write model evidence. Idempotent per event."""
    conn.execute("DELETE FROM detection WHERE event_id = ?", (evidence.event_id,))
    conn.execute("DELETE FROM pose_observation WHERE event_id = ?", (evidence.event_id,))

    db.insert_many(
        conn, "detection",
        ["event_id", "t", "cls", "confidence", "x1", "y1", "x2", "y2",
         "px_area", "abstained"],
        [(evidence.event_id, d.t, d.cls, d.confidence, *d.box, d.px_area,
          int(d.abstained)) for d in evidence.detections],
    )
    db.insert_many(
        conn, "pose_observation",
        ["event_id", "t", "track_id", "keypoints", "head_pitch", "wrist_masked"],
        [(evidence.event_id, 0.0, o.track_id,
          str([[float(x) for x in row] for row in o.keypoints]),
          o.head_pitch, int(o.wrist_masked)) for o in evidence.poses],
    )


def run(
    conn: sqlite3.Connection,
    source_video_id: int,
    cal: Calibration,
    cfg: DetectConfig,
    detector: ONNXDetector | None = None,
    pose_model: RTMOPose | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> list[EventEvidence]:
    """Run model evidence over this video's candidate events."""
    if detector is None and pose_model is None:
        raise ValueError(
            "no model supplied; phase 6 requires a detector, a pose model, or both"
        )

    video = db.one(conn, "SELECT * FROM source_video WHERE id = ?", (source_video_id,))
    if video is None:
        raise KeyError(f"no source_video {source_video_id}")

    rows = db.all_rows(
        conn,
        """
        SELECT e.id, e.t_start, e.t_end, s.seat_label
        FROM event e JOIN seat s ON s.id = e.seat_id
        WHERE e.source_video_id = ? ORDER BY e.peak_deviation DESC
        """,
        (source_video_id,),
    )
    if limit is not None:
        rows = rows[:limit]

    by_label = {s.label: s for s in cal.seats}
    results: list[EventEvidence] = []

    for row in rows:
        seat = by_label.get(row["seat_label"])
        if seat is None:
            continue
        result = analyse_event(
            video["path"], row["id"], row["seat_label"], seat.polygon,
            seat.desk_line_y, row["t_start"], row["t_end"], cfg,
            detector, pose_model,
        )
        persist(conn, result)
        results.append(result)
        if verbose:
            print(f"    event {row['id']} seat {row['seat_label']}: "
                  f"{len(result.detections)} detections, {len(result.poses)} poses, "
                  f"streams={','.join(result.streams_present) or 'none'}")

    db.audit(conn, "evidence", "source_video", source_video_id,
             detail=f"{len(results)} events")
    return results
