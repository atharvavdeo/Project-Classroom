-- Project Classroom event store.
-- All timing is in seconds, derived from presentation timestamps. Never frame indices.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- sources --

CREATE TABLE IF NOT EXISTS session (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    exam_date   TEXT,
    notes       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS camera (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,          -- e.g. "Camera 04", read off the burned-in overlay
    room        TEXT,                   -- used for leave-one-room-out splits
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (session_id, label)
);

CREATE TABLE IF NOT EXISTS source_video (
    id              INTEGER PRIMARY KEY,
    camera_id       INTEGER NOT NULL REFERENCES camera(id) ON DELETE CASCADE,
    path            TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    container       TEXT NOT NULL,
    codec           TEXT NOT NULL,
    pix_fmt         TEXT NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    avg_fps         REAL NOT NULL,
    duration_s      REAL NOT NULL,
    bit_rate_mbps   REAL,
    frame_count     INTEGER NOT NULL,   -- counted, not trusted from metadata
    is_vfr          INTEGER NOT NULL,   -- 0/1, measured from PTS deltas
    pts_delta_med   REAL,               -- seconds
    pts_delta_std   REAL,
    has_mv          INTEGER NOT NULL,   -- 0/1, codec motion vectors present
    mv_per_frame    REAL,
    wall_clock_utc  TEXT,               -- start time if known
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (sha256)
);

-- ------------------------------------------------------------ calibration --

CREATE TABLE IF NOT EXISTS calibration (
    id          INTEGER PRIMARY KEY,
    camera_id   INTEGER NOT NULL REFERENCES camera(id) ON DELETE CASCADE,
    version     INTEGER NOT NULL,
    frame_w     INTEGER NOT NULL,       -- resolution the polygons were drawn against
    frame_h     INTEGER NOT NULL,
    homography  TEXT,                   -- JSON 3x3, nullable until reference points marked
    validated   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (camera_id, version)
);

CREATE TABLE IF NOT EXISTS seat (
    id              INTEGER PRIMARY KEY,
    calibration_id  INTEGER NOT NULL REFERENCES calibration(id) ON DELETE CASCADE,
    seat_label      TEXT NOT NULL,      -- physical placard, e.g. "61", "13"
    polygon         TEXT NOT NULL,      -- JSON [[x,y],...] full seat footprint
    desk_line_y     REAL,               -- frame y of the desk edge; below it is lap
    UNIQUE (calibration_id, seat_label)
);

-- Three zones per seat. 'screen' is excluded from motion scoring entirely.
CREATE TABLE IF NOT EXISTS seat_zone (
    id          INTEGER PRIMARY KEY,
    seat_id     INTEGER NOT NULL REFERENCES seat(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('screen', 'desk', 'torso')),
    polygon     TEXT NOT NULL,
    UNIQUE (seat_id, kind)
);

-- Frame-global exclusions: burned-in overlay, reflective glass, staff paths.
CREATE TABLE IF NOT EXISTS mask_region (
    id              INTEGER PRIMARY KEY,
    calibration_id  INTEGER NOT NULL REFERENCES calibration(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('overlay', 'reflection', 'environment', 'staff')),
    polygon         TEXT NOT NULL,
    note            TEXT
);

-- Learned during the calibration pass, per seat.
CREATE TABLE IF NOT EXISTS seat_baseline (
    id              INTEGER PRIMARY KEY,
    calibration_id  INTEGER NOT NULL REFERENCES calibration(id) ON DELETE CASCADE,
    seat_id         INTEGER NOT NULL REFERENCES seat(id) ON DELETE CASCADE,
    zone            TEXT NOT NULL,
    median          REAL NOT NULL,      -- robust centre of the motion distribution
    mad             REAL NOT NULL,      -- median absolute deviation
    p95             REAL NOT NULL,
    typing_median   REAL,               -- centre of the active-typing regime
    typing_mad      REAL,
    sample_windows  INTEGER NOT NULL,
    UNIQUE (calibration_id, seat_id, zone)
);

-- ------------------------------------------------------------------- jobs --

CREATE TABLE IF NOT EXISTS job (
    id              INTEGER PRIMARY KEY,
    source_video_id INTEGER NOT NULL REFERENCES source_video(id) ON DELETE CASCADE,
    calibration_id  INTEGER REFERENCES calibration(id),
    stage           TEXT NOT NULL,      -- ingest|motion|segment|detect|verify|done
    status          TEXT NOT NULL,      -- pending|running|complete|failed|cancelled
    progress        REAL NOT NULL DEFAULT 0.0,
    error           TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ----------------------------------------------------------------- motion --

-- One row per (seat, zone, time window). This is the pass-1 output and the
-- only artefact retained for static intervals.
CREATE TABLE IF NOT EXISTS motion_window (
    id              INTEGER PRIMARY KEY,
    source_video_id INTEGER NOT NULL REFERENCES source_video(id) ON DELETE CASCADE,
    seat_id         INTEGER NOT NULL REFERENCES seat(id) ON DELETE CASCADE,
    zone            TEXT NOT NULL,
    t_start         REAL NOT NULL,      -- seconds, from PTS
    t_end           REAL NOT NULL,
    mag_median      REAL NOT NULL,
    mag_p90         REAL NOT NULL,
    mag_p95         REAL NOT NULL,
    area_ratio      REAL NOT NULL,      -- fraction of zone blocks in motion
    coherence       REAL NOT NULL,      -- directional agreement, 0-1
    residual        REAL NOT NULL,      -- after global-motion compensation
    below_desk      REAL NOT NULL DEFAULT 0.0,
    deviation       REAL NOT NULL,      -- robust z vs seat baseline / typing regime
    regime          TEXT NOT NULL DEFAULT 'unknown'  -- idle|typing|unknown
);

CREATE INDEX IF NOT EXISTS idx_mw_video_seat_t
    ON motion_window (source_video_id, seat_id, t_start);

-- Whole-frame per-window diagnostics: camera shake, exposure steps.
CREATE TABLE IF NOT EXISTS frame_window (
    id              INTEGER PRIMARY KEY,
    source_video_id INTEGER NOT NULL REFERENCES source_video(id) ON DELETE CASCADE,
    t_start         REAL NOT NULL,
    t_end           REAL NOT NULL,
    global_dx       REAL NOT NULL,
    global_dy       REAL NOT NULL,
    global_ratio    REAL NOT NULL,      -- fraction of frame moving coherently
    luma_mean       REAL NOT NULL,
    luma_delta      REAL NOT NULL,
    exposure_step   INTEGER NOT NULL DEFAULT 0,
    camera_moved    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_fw_video_t
    ON frame_window (source_video_id, t_start);

-- ----------------------------------------------------------------- events --

CREATE TABLE IF NOT EXISTS event (
    id              INTEGER PRIMARY KEY,
    source_video_id INTEGER NOT NULL REFERENCES source_video(id) ON DELETE CASCADE,
    seat_id         INTEGER NOT NULL REFERENCES seat(id) ON DELETE CASCADE,
    t_start         REAL NOT NULL,
    t_end           REAL NOT NULL,
    peak_t          REAL NOT NULL,
    peak_deviation  REAL NOT NULL,
    score           REAL NOT NULL DEFAULT 0.0,
    score_factors   TEXT,               -- JSON {factor: contribution}, shown in UI
    evidence_quality TEXT NOT NULL DEFAULT 'unassessed',
    explanation     TEXT,
    clip_path       TEXT,
    thumb_path      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_event_video_t ON event (source_video_id, t_start);
CREATE INDEX IF NOT EXISTS idx_event_score   ON event (score DESC);

-- ------------------------------------------------------- model evidence  --

CREATE TABLE IF NOT EXISTS detection (
    id          INTEGER PRIMARY KEY,
    event_id    INTEGER NOT NULL REFERENCES event(id) ON DELETE CASCADE,
    t           REAL NOT NULL,
    cls         TEXT NOT NULL,          -- phone_like|paper_like|other|person
    confidence  REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    px_area     REAL NOT NULL,          -- drives the abstention rule
    abstained   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pose_observation (
    id              INTEGER PRIMARY KEY,
    event_id        INTEGER NOT NULL REFERENCES event(id) ON DELETE CASCADE,
    t               REAL NOT NULL,
    track_id        INTEGER,
    keypoints       TEXT NOT NULL,      -- JSON [[x,y,conf],...] COCO-17
    head_pitch      REAL,               -- nose-to-shoulder vertical offset, normalised
    wrist_masked    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vlm_verification (
    id                      INTEGER PRIMARY KEY,
    event_id                INTEGER NOT NULL REFERENCES event(id) ON DELETE CASCADE,
    model                   TEXT NOT NULL,
    object_assessment       TEXT NOT NULL,
    interaction_assessment  TEXT NOT NULL,
    evidence_quality        TEXT NOT NULL,
    confidence              REAL NOT NULL,
    observations            TEXT NOT NULL,   -- JSON list
    review_note             TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ------------------------------------------------------------- oversight --

CREATE TABLE IF NOT EXISTS human_review (
    id          INTEGER PRIMARY KEY,
    event_id    INTEGER NOT NULL REFERENCES event(id) ON DELETE CASCADE,
    verdict     TEXT NOT NULL CHECK (verdict IN ('relevant', 'irrelevant', 'inconclusive')),
    reviewer    TEXT NOT NULL,
    note        TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_entry (
    id          INTEGER PRIMARY KEY,
    action      TEXT NOT NULL,
    entity      TEXT NOT NULL,
    entity_id   INTEGER,
    detail      TEXT,
    actor       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Ground truth for evaluation. Kept separate from system output by design.
CREATE TABLE IF NOT EXISTS annotation (
    id              INTEGER PRIMARY KEY,
    source_video_id INTEGER NOT NULL REFERENCES source_video(id) ON DELETE CASCADE,
    seat_label      TEXT,
    t_start         REAL NOT NULL,
    t_end           REAL NOT NULL,
    event_type      TEXT NOT NULL,
    visibility      TEXT NOT NULL,      -- clear|partial|occluded|insufficient
    annotator       TEXT,
    exhaustive      INTEGER NOT NULL DEFAULT 0,  -- 1 if its span was labelled exhaustively
    note            TEXT
);
