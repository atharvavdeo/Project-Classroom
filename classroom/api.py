"""Phase 5 - investigator API.

    uvicorn classroom.api:app --reload

Read-only over the analysis output, except for the review endpoint. Human
feedback is append-only: a review never rewrites the deterministic record it
refers to (PRD 12.1).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import db
from .config import Config

CFG = Config()
STATIC = Path(__file__).with_name("static")

app = FastAPI(title="Project Classroom", version="3.0.0")


def conn() -> sqlite3.Connection:
    return db.connect(CFG.db_path)


class ReviewIn(BaseModel):
    verdict: str = Field(pattern="^(relevant|irrelevant|inconclusive)$")
    reviewer: str = "investigator"
    note: str | None = None


@app.get("/", response_class=HTMLResponse)
def console() -> str:
    return (STATIC / "console.html").read_text(encoding="utf-8")


@app.get("/api/videos")
def videos() -> list[dict]:
    with conn() as c:
        rows = db.all_rows(
            c,
            """
            SELECT v.id, v.path, v.width, v.height, v.avg_fps, v.duration_s,
                   v.codec, v.is_vfr, c.label AS camera, c.room,
                   (SELECT COUNT(*) FROM event e WHERE e.source_video_id = v.id) AS events
            FROM source_video v JOIN camera c ON c.id = v.camera_id
            ORDER BY v.id
            """,
        )
        return [dict(r) for r in rows]


@app.get("/api/videos/{video_id}/seats")
def seats(video_id: int) -> list[dict]:
    with conn() as c:
        rows = db.all_rows(
            c,
            """
            SELECT DISTINCT s.id, s.seat_label
            FROM seat s JOIN motion_window m ON m.seat_id = s.id
            WHERE m.source_video_id = ? ORDER BY s.seat_label
            """,
            (video_id,),
        )
        return [dict(r) for r in rows]


@app.get("/api/videos/{video_id}/timeline")
def timeline(video_id: int, zone: str = "desk") -> dict:
    """Per-seat deviation over time: the activity timeline (FR-15)."""
    with conn() as c:
        rows = db.all_rows(
            c,
            """
            SELECT s.seat_label, m.t_start, m.deviation, m.regime
            FROM motion_window m JOIN seat s ON s.id = m.seat_id
            WHERE m.source_video_id = ? AND m.zone = ?
            ORDER BY s.seat_label, m.t_start
            """,
            (video_id, zone),
        )
        series: dict[str, list[list[float]]] = {}
        for r in rows:
            series.setdefault(r["seat_label"], []).append(
                [round(r["t_start"], 3), round(r["deviation"], 4)]
            )
        frames = db.all_rows(
            c,
            """
            SELECT t_start, camera_moved, exposure_step FROM frame_window
            WHERE source_video_id = ? AND (camera_moved = 1 OR exposure_step = 1)
            """,
            (video_id,),
        )
        return {
            "series": series,
            "suppressed": [
                {"t": r["t_start"],
                 "reason": "camera moved" if r["camera_moved"] else "exposure step"}
                for r in frames
            ],
        }


@app.get("/api/videos/{video_id}/heatmap")
def heatmap(video_id: int, zone: str = "desk") -> list[dict]:
    """Total activity per seat: the spatial heatmap (FR-15)."""
    with conn() as c:
        rows = db.all_rows(
            c,
            """
            SELECT s.seat_label,
                   AVG(m.residual)  AS mean_residual,
                   MAX(m.deviation) AS peak_deviation,
                   SUM(CASE WHEN m.deviation > 4 THEN 1 ELSE 0 END) AS hot_windows,
                   COUNT(*) AS windows,
                   MAX(m.below_desk) AS below_desk_peak
            FROM motion_window m JOIN seat s ON s.id = m.seat_id
            WHERE m.source_video_id = ? AND m.zone = ?
            GROUP BY s.seat_label ORDER BY peak_deviation DESC
            """,
            (video_id, zone),
        )
        return [dict(r) for r in rows]


@app.get("/api/events")
def events(
    video_id: int | None = None,
    seat: str | None = None,
    min_score: float | None = None,
    status: str | None = Query(None, pattern="^(reviewed|unreviewed)$"),
    limit: int = 200,
) -> list[dict]:
    clauses, params = [], []
    if video_id is not None:
        clauses.append("e.source_video_id = ?")
        params.append(video_id)
    if seat:
        clauses.append("s.seat_label = ?")
        params.append(seat)
    if min_score is not None:
        clauses.append("e.score >= ?")
        params.append(min_score)
    if status == "reviewed":
        clauses.append("r.id IS NOT NULL")
    elif status == "unreviewed":
        clauses.append("r.id IS NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with conn() as c:
        rows = db.all_rows(
            c,
            f"""
            SELECT e.id, e.source_video_id, s.seat_label, e.t_start, e.t_end,
                   e.peak_t, e.peak_deviation, e.score, e.score_factors,
                   e.evidence_quality, e.explanation,
                   e.clip_path IS NOT NULL AS has_clip,
                   e.thumb_path IS NOT NULL AS has_thumb,
                   r.verdict, r.note AS review_note
            FROM event e
            JOIN seat s ON s.id = e.seat_id
            LEFT JOIN human_review r ON r.id = (
                SELECT id FROM human_review WHERE event_id = e.id
                ORDER BY id DESC LIMIT 1
            )
            {where}
            ORDER BY e.score DESC, e.peak_deviation DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        out = []
        for r in rows:
            item = dict(r)
            item["score_factors"] = (
                json.loads(r["score_factors"]) if r["score_factors"] else {}
            )
            out.append(item)
        return out


@app.get("/api/events/{event_id}")
def event_detail(event_id: int) -> dict:
    with conn() as c:
        row = db.one(
            c,
            """
            SELECT e.*, s.seat_label FROM event e
            JOIN seat s ON s.id = e.seat_id WHERE e.id = ?
            """,
            (event_id,),
        )
        if row is None:
            raise HTTPException(404, f"no event {event_id}")
        item = dict(row)
        item["score_factors"] = (
            json.loads(row["score_factors"]) if row["score_factors"] else {}
        )
        item["reviews"] = [
            dict(r) for r in db.all_rows(
                c,
                "SELECT verdict, reviewer, note, created_at FROM human_review "
                "WHERE event_id = ? ORDER BY id",
                (event_id,),
            )
        ]
        return item


@app.get("/api/events/{event_id}/clip")
def clip(event_id: int) -> FileResponse:
    with conn() as c:
        row = db.one(c, "SELECT clip_path FROM event WHERE id = ?", (event_id,))
    if row is None or not row["clip_path"]:
        raise HTTPException(404, "no clip generated for this event")
    path = Path(row["clip_path"])
    if not path.is_file():
        raise HTTPException(404, f"clip missing on disk: {path}")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/events/{event_id}/thumb")
def thumb(event_id: int) -> FileResponse:
    with conn() as c:
        row = db.one(c, "SELECT thumb_path FROM event WHERE id = ?", (event_id,))
    if row is None or not row["thumb_path"]:
        raise HTTPException(404, "no thumbnail generated for this event")
    path = Path(row["thumb_path"])
    if not path.is_file():
        raise HTTPException(404, f"thumbnail missing on disk: {path}")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/events/{event_id}/reviews")
def review(event_id: int, body: ReviewIn) -> dict:
    with conn() as c:
        if db.one(c, "SELECT id FROM event WHERE id = ?", (event_id,)) is None:
            raise HTTPException(404, f"no event {event_id}")
        review_id = db.insert(
            c, "human_review", event_id=event_id, verdict=body.verdict,
            reviewer=body.reviewer, note=body.note,
        )
        db.audit(c, "review", "event", event_id, detail=body.verdict, actor=body.reviewer)
        return {"id": review_id, "event_id": event_id, "verdict": body.verdict}


@app.get("/api/stats")
def stats() -> dict:
    with conn() as c:
        def scalar(sql: str) -> float:
            row = c.execute(sql).fetchone()
            return row[0] if row and row[0] is not None else 0

        total_events = scalar("SELECT COUNT(*) FROM event")
        reviewed = scalar("SELECT COUNT(DISTINCT event_id) FROM human_review")
        duration = scalar("SELECT SUM(duration_s) FROM source_video")
        covered = scalar("SELECT SUM(t_end - t_start) FROM event")
        return {
            "videos": scalar("SELECT COUNT(*) FROM source_video"),
            "events": total_events,
            "reviewed": reviewed,
            "footage_seconds": duration,
            "flagged_seconds": covered,
            "review_reduction": (1 - covered / duration) if duration else 0.0,
        }
