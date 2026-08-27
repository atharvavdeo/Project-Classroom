-- Drishti console store.
--
-- Two things live here, and they are different in kind.
--
-- 1. The *hierarchy* (project > centre > recording) is user-authored. Nothing
--    is seeded. A centre named "Dahisar" exists only because someone typed it,
--    because a fixture centre in the sidebar is a claim that a centre exists.
--
-- 2. The *decisions* are the beginning of a training set. They are append-only:
--    a reviewer who changes their mind writes a second row, and the first one
--    stays. The recursive-learning pipeline needs the reversal as much as the
--    verdict -- "confirmed then dismissed" is a harder and more informative
--    example than either alone. Nothing in this file ever deletes a decision.
--
-- The console reads the *latest* decision per (video, track) through the
-- `current_decision` view. The table underneath keeps everything.


CREATE TABLE IF NOT EXISTS projects (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  created_utc  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS centres (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name         TEXT NOT NULL,
  created_utc  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS centres_by_project ON centres(project_id);

-- `processed` is not a boolean the user sets; it is derived from whether a
-- bundle path was registered. A recording with no run must say "not run"
-- rather than showing an empty dashboard, which reads as "nothing was found".
CREATE TABLE IF NOT EXISTS videos (
  id           TEXT PRIMARY KEY,
  centre_id    TEXT NOT NULL REFERENCES centres(id) ON DELETE CASCADE,
  name         TEXT NOT NULL,
  media_url    TEXT,
  bundle_url   TEXT,
  overlay_url  TEXT,
  crops_url    TEXT,
  source_path  TEXT,
  duration_ms  INTEGER,
  created_utc  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS videos_by_centre ON videos(centre_id);

-- Append-only. No UPDATE, no DELETE.
--
-- `decision` uses the contract's own vocabulary so a row can be replayed into
-- the pipeline without translation: human_confirmed / human_dismissed are the
-- two states fusion.assert_machine_state() refuses to write, and this is the
-- only place they may come from. `needs_better_view` is the reviewer
-- abstaining, which is a third answer and never a quiet dismissal.
CREATE TABLE IF NOT EXISTS decisions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id      TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  track_id      INTEGER NOT NULL,
  decision      TEXT NOT NULL
                CHECK (decision IN ('human_confirmed',
                                    'human_dismissed',
                                    'needs_better_view')),
  -- What the machine had said at the moment the human answered. Without this
  -- a dismissal is unusable as a label: "wrong" is only meaningful against
  -- what was claimed.
  machine_state TEXT,
  key_class     TEXT,
  key_verdict   TEXT,
  key_pts_ms    REAL,
  -- Optional label fields for the future training pass. Left null by the
  -- current UI rather than guessed.
  object_truth     TEXT,
  evidence_quality TEXT,
  policy_context   TEXT,
  note          TEXT,
  reviewer      TEXT NOT NULL DEFAULT 'local',
  decided_utc   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS decisions_by_video ON decisions(video_id, track_id, id);

-- Latest answer per person. The history stays in `decisions`.
CREATE VIEW IF NOT EXISTS current_decision AS
SELECT d.*
FROM decisions d
JOIN (
  SELECT video_id, track_id, MAX(id) AS id
  FROM decisions
  GROUP BY video_id, track_id
) last ON last.id = d.id;

-- ----------------------------------------------------------- intervals --
--
-- A stretch of a recording a human marked, by hand, on the timeline.
--
-- This is the one kind of claim in the system that no model produced. A
-- reviewer watched the video, saw something between two timestamps, and said
-- so. It is kept apart from `decisions` because a decision is about a *person*
-- (a track id) and this is about a *span* -- often before anyone knows which
-- track, or covering several people, or covering a person tracking never
-- resolved at all.
--
-- Append-only for the same reason decisions are: this is training data. A
-- reviewer who marks a stretch and later retracts it writes a second row with
-- `retracted = 1`, and both survive. "Marked then retracted" is a more
-- informative example than either alone.
--
-- These spans are also what stage 3 of the clip database cuts on: a confirmed
-- interval is exactly the clip worth keeping.
CREATE TABLE IF NOT EXISTS intervals (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id     TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  from_ms      REAL NOT NULL,
  to_ms        REAL NOT NULL,
  -- What the reviewer is asserting. `violation` is the strong one and is the
  -- only value drawn in alarm colour; `review` means "look here", which is a
  -- weaker and much more common claim.
  kind         TEXT NOT NULL DEFAULT 'violation'
               CHECK (kind IN ('violation','review','note')),
  -- Optional: which person, when the reviewer knows. Null is normal and does
  -- not weaken the mark.
  track_id     INTEGER,
  note         TEXT,
  retracted    INTEGER NOT NULL DEFAULT 0 CHECK (retracted IN (0,1)),
  reviewer     TEXT NOT NULL DEFAULT 'local',
  created_utc  TEXT NOT NULL,
  CHECK (to_ms > from_ms)
);

CREATE INDEX IF NOT EXISTS intervals_by_video ON intervals(video_id, from_ms);

-- ---------------------------------------------------------------- jobs --
--
-- A processing job, from "a video was uploaded" to "a run exists".
--
-- The console never waits for the pipeline. Stages 1-15 are minutes of GPU
-- work on a short clip and much longer on a real recording; an HTTP request
-- held open for that is a request that times out somewhere in the middle and
-- leaves nobody sure whether the work is still running. So `POST /process`
-- writes a row here, answers 202, and everything after that is this table
-- changing state.
--
-- The GPU box *pulls* work rather than being pushed to. That is deliberate:
-- pulling needs no inbound connectivity, no port forward and no tunnel, which
-- matters when the machine sits behind a college NAT. `claimed_by` and
-- `claim_token` are what stop two agents running the same job twice.
CREATE TABLE IF NOT EXISTS jobs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id      TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  state         TEXT NOT NULL DEFAULT 'queued'
                CHECK (state IN ('queued','claimed','running','complete','failed','cancelled')),
  -- Which pipeline stage is executing, and how far through. Both are reported
  -- by the agent and are display-only: nothing branches on them.
  stage         TEXT,
  progress      REAL NOT NULL DEFAULT 0,
  -- Set when the job finishes: the run key whose artifacts the console reads.
  run_key       TEXT,
  error         TEXT,
  claimed_by    TEXT,
  claim_token   TEXT,
  attempts      INTEGER NOT NULL DEFAULT 0,
  created_utc   TEXT NOT NULL,
  updated_utc   TEXT NOT NULL,
  finished_utc  TEXT
);

CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs(state, id);
CREATE INDEX IF NOT EXISTS jobs_by_video ON jobs(video_id, id);
