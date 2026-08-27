/**
 * Every API route, once, for both runtimes.
 *
 * The console runs in two places: a Node process (a laptop, or the GPU box)
 * and a Cloudflare Worker. Those have almost nothing in common -- one has a
 * filesystem and better-sqlite3, the other has D1 and R2 -- but the contract
 * they serve has to be identical, because the same front end talks to both and
 * the same decisions have to mean the same thing in either store.
 *
 * So the routes live here as pure functions over a `store` and a small set of
 * capabilities, and each runtime supplies a thin adapter. Two copies of this
 * logic would drift, and the first thing to drift would be the decision
 * vocabulary -- the one part that must not.
 */

const DECISIONS = new Set([
  "human_confirmed",
  "human_dismissed",
  "needs_better_view",
]);

const nowUtc = () => new Date().toISOString();

/** A claim token. `crypto.randomUUID` exists in both Node 22 and Workers, so
 *  this needs no runtime-specific import. */
const cryptoToken = () => crypto.randomUUID().replace(/-/g, "");

/** Slug that stays readable in a URL and unique without a UUID column. */
export function slug(name, existing) {
  const base =
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 48) || "item";
  if (!existing.has(base)) return base;
  for (let i = 2; ; i += 1) {
    const candidate = `${base}-${i}`;
    if (!existing.has(candidate)) return candidate;
  }
}

const ok = (body, status = 200) => ({ status, body });
const bad = (status, message) => ({ status, body: { error: message } });

/**
 * @param store       the async store (see store.js / the Worker's D1 adapter)
 * @param assets      () => Promise<{media,bundles,overlays,crops}>
 * @param describe    (payload) => Promise<{status, body}>  the vision proxy
 */
export function makeRouter({ store, assets, describe, agentToken }) {
  async function tree() {
    const [projects, centres, videos, decided] = await Promise.all([
      store.all("SELECT * FROM projects ORDER BY created_utc"),
      store.all("SELECT * FROM centres ORDER BY created_utc"),
      store.all("SELECT * FROM videos ORDER BY created_utc"),
      store.all(
        "SELECT video_id, COUNT(*) AS n FROM current_decision GROUP BY video_id",
      ),
    ]);
    const decidedBy = Object.fromEntries(
      decided.map((r) => [r.video_id, Number(r.n)]),
    );

    return projects.map((p) => ({
      id: p.id,
      name: p.name,
      centres: centres
        .filter((c) => c.project_id === p.id)
        .map((c) => ({
          id: c.id,
          name: c.name,
          videos: videos
            .filter((v) => v.centre_id === c.id)
            .map((v) => ({
              id: v.id,
              name: v.name,
              video: v.media_url,
              bundle: v.bundle_url,
              overlay: v.overlay_url,
              crops: v.crops_url,
              // Derived, never stored: a recording is processed exactly when a
              // run bundle was registered for it.
              processed: Boolean(v.bundle_url),
              decided: decidedBy[v.id] ?? 0,
            })),
        })),
    }));
  }

  /**
   * @returns {Promise<{status:number, body:any}|null>} null when the path is
   *          not an API route, so the caller can fall through to static files.
   */
  return async function route(method, pathname, body) {
    const seg = pathname.replace(/^\/api\/?/, "").split("/").filter(Boolean);
    if (!pathname.startsWith("/api/")) return null;

    /* --------------------------------------------------------------- tree */

    if (method === "GET" && pathname === "/api/tree") {
      return ok({ projects: await tree() });
    }

    if (method === "GET" && pathname === "/api/health") {
      return ok({
        ok: true,
        store: store.kind,
        vision: Boolean(describe),
        projects: Number(
          (await store.get("SELECT COUNT(*) AS n FROM projects"))?.n ?? 0,
        ),
        decisions: Number(
          (await store.get("SELECT COUNT(*) AS n FROM decisions"))?.n ?? 0,
        ),
      });
    }

    if (method === "GET" && pathname === "/api/assets") {
      return ok(await assets());
    }

    /* ---------------------------------------------------------- hierarchy */

    if (method === "POST" && pathname === "/api/projects") {
      const name = String(body?.name ?? "").trim();
      if (!name) return bad(400, "A project needs a name.");
      const taken = new Set(
        (await store.all("SELECT id FROM projects")).map((r) => r.id),
      );
      const id = slug(name, taken);
      await store.run(
        "INSERT INTO projects (id, name, created_utc) VALUES (?, ?, ?)",
        [id, name, nowUtc()],
      );
      return ok({ id, name, projects: await tree() }, 201);
    }

    if (method === "POST" && seg[0] === "projects" && seg[2] === "centres") {
      const project = await store.get("SELECT id FROM projects WHERE id = ?", [
        seg[1],
      ]);
      if (!project) return bad(404, "No such project.");
      const name = String(body?.name ?? "").trim();
      if (!name) return bad(400, "A centre needs a name.");
      const taken = new Set(
        (await store.all("SELECT id FROM centres")).map((r) => r.id),
      );
      const id = slug(`${seg[1]}-${name}`, taken);
      await store.run(
        "INSERT INTO centres (id, project_id, name, created_utc) VALUES (?,?,?,?)",
        [id, seg[1], name, nowUtc()],
      );
      return ok({ id, projects: await tree() }, 201);
    }

    if (method === "POST" && seg[0] === "centres" && seg[2] === "videos") {
      const centre = await store.get("SELECT id FROM centres WHERE id = ?", [
        seg[1],
      ]);
      if (!centre) return bad(404, "No such centre.");
      const name = String(body?.name ?? "").trim();
      if (!name) return bad(400, "A recording needs a name.");
      const taken = new Set(
        (await store.all("SELECT id FROM videos")).map((r) => r.id),
      );
      const id = slug(name, taken);
      const pick = (k) => {
        const v = body?.[k];
        return typeof v === "string" && v.trim() ? v.trim() : null;
      };
      await store.run(
        `INSERT INTO videos
           (id, centre_id, name, media_url, bundle_url, overlay_url, crops_url,
            source_path, created_utc)
         VALUES (?,?,?,?,?,?,?,?,?)`,
        [
          id,
          seg[1],
          name,
          pick("media_url"),
          pick("bundle_url"),
          pick("overlay_url"),
          pick("crops_url"),
          pick("source_path"),
          nowUtc(),
        ],
      );
      return ok({ id, projects: await tree() }, 201);
    }

    // Removing a *container* is allowed; removing a decision is not. Decisions
    // cascade only because a recording that no longer exists has nothing to
    // label.
    if (
      method === "DELETE" &&
      ["projects", "centres", "videos"].includes(seg[0]) &&
      seg[1]
    ) {
      await store.run(`DELETE FROM ${seg[0]} WHERE id = ?`, [seg[1]]);
      return ok({ projects: await tree() });
    }

    /* ---------------------------------------------------------- decisions */

    if (method === "GET" && seg[0] === "videos" && seg[2] === "decisions") {
      const [rows, total] = await Promise.all([
        store.all("SELECT * FROM current_decision WHERE video_id = ?", [seg[1]]),
        store.get("SELECT COUNT(*) AS n FROM decisions WHERE video_id = ?", [
          seg[1],
        ]),
      ]);
      return ok({
        current: Object.fromEntries(rows.map((r) => [r.track_id, r])),
        revisions: Number(total?.n ?? 0),
      });
    }

    if (method === "POST" && seg[0] === "videos" && seg[2] === "decisions") {
      const video = await store.get("SELECT id FROM videos WHERE id = ?", [
        seg[1],
      ]);
      if (!video) return bad(404, "No such recording.");

      const decision = String(body?.decision ?? "");
      if (!DECISIONS.has(decision)) {
        return bad(400, `decision must be one of ${[...DECISIONS].join(", ")}`);
      }
      const trackId = Number(body?.track_id);
      if (!Number.isInteger(trackId)) {
        return bad(400, "track_id must be an integer.");
      }

      const { lastId } = await store.run(
        `INSERT INTO decisions
           (video_id, track_id, decision, machine_state, key_class, key_verdict,
            key_pts_ms, note, reviewer, decided_utc)
         VALUES (?,?,?,?,?,?,?,?,?,?)`,
        [
          seg[1],
          trackId,
          decision,
          body?.machine_state ?? null,
          body?.key_class ?? null,
          body?.key_verdict ?? null,
          body?.key_pts_ms ?? null,
          body?.note ?? null,
          body?.reviewer ?? "local",
          nowUtc(),
        ],
      );
      return ok(
        await store.get("SELECT * FROM decisions WHERE id = ?", [lastId]),
        201,
      );
    }

    /** Everything a human said was not a violation, newest first. This is the
     *  false-positive set, and the most valuable thing the console produces. */
    if (method === "GET" && seg[0] === "videos" && seg[2] === "dismissed") {
      return ok({
        rows: await store.all(
          `SELECT * FROM current_decision
            WHERE video_id = ? AND decision = 'human_dismissed'
            ORDER BY id DESC`,
          [seg[1]],
        ),
      });
    }

    /** Every proposal a reviewer overturned, across every recording.
     *
     * This is the negative half of the training set and it is deliberately not
     * per-video: the point of looking at it is to see what the machine keeps
     * getting wrong *in general*. A per-recording view answers "was this hall
     * clean", which is a different question and already has a screen.
     *
     * Reads `current_decision`, so a person confirmed after being dismissed is
     * not listed. The reversal itself stays in `decisions` and is what
     * `/api/labels.jsonl` exports. */
    if (method === "GET" && pathname === "/api/dismissed") {
      return ok({
        rows: await store.all(
          `SELECT d.*, v.name AS video_name, c.name AS centre_name
             FROM current_decision d
             JOIN videos  v ON v.id = d.video_id
             LEFT JOIN centres c ON c.id = v.centre_id
            WHERE d.decision = 'human_dismissed'
            ORDER BY d.id DESC`,
        ),
      });
    }

    /** The same set as /api/dismissed, as a spreadsheet.
     *
     * Server-side rather than built in the browser so it is scriptable: this is
     * the negative half of a training set, and something other than a person
     * clicking a button will eventually want it. Same reasoning as
     * /api/labels.jsonl, which is the full history including reversals -- this
     * is the flat current-state view. */
    if (method === "GET" && pathname === "/api/dismissed.csv") {
      const rows = await store.all(
        `SELECT d.decided_utc, c.name AS centre_name, v.name AS video_name,
                d.video_id, d.track_id, d.key_pts_ms, d.key_class,
                d.key_verdict, d.machine_state, d.object_truth,
                d.evidence_quality, d.policy_context, d.note, d.reviewer
           FROM current_decision d
           JOIN videos  v ON v.id = d.video_id
           LEFT JOIN centres c ON c.id = v.centre_id
          WHERE d.decision = 'human_dismissed'
          ORDER BY d.id DESC`,
      );
      const columns = [
        "decided_utc", "centre_name", "video_name", "video_id", "track_id",
        "key_pts_ms", "key_class", "key_verdict", "machine_state",
        "object_truth", "evidence_quality", "policy_context", "note",
        "reviewer",
      ];
      // RFC 4180: quote everything, double any inner quote. Quoting
      // unconditionally rather than only when needed keeps a recording named
      // "Hall 3 · Seat 12, paper" from splitting itself across two columns.
      const cell = (v) =>
        v === null || v === undefined ? '""' : `"${String(v).replace(/"/g, '""')}"`;
      const lines = [columns.map(cell).join(",")];
      for (const r of rows) lines.push(columns.map((k) => cell(r[k])).join(","));
      return {
        status: 200,
        // The BOM is for Excel: without it, Excel reads the file as the local
        // codepage and the centre names with a middle dot arrive as mojibake.
        text: "﻿" + lines.join("\r\n") + "\r\n",
        type: "text/csv; charset=utf-8",
        headers: {
          "content-disposition":
            'attachment; filename="drishti-rejected-proposals.csv"',
        },
      };
    }

    /** Stretches of a recording a human marked by hand. Retracted rows are
     *  kept in the table but not returned: the timeline draws what stands. */
    if (method === "GET" && seg[0] === "videos" && seg[2] === "intervals") {
      return ok({
        rows: await store.all(
          `SELECT * FROM intervals
            WHERE video_id = ? AND retracted = 0
            ORDER BY from_ms`,
          [seg[1]],
        ),
      });
    }

    if (method === "POST" && seg[0] === "videos" && seg[2] === "intervals") {
      const from = Number(body?.from_ms);
      const to = Number(body?.to_ms);
      if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) {
        return bad(400, "from_ms and to_ms must be numbers with to_ms > from_ms.");
      }
      const kind = body?.kind ?? "violation";
      if (!["violation", "review", "note"].includes(kind)) {
        return bad(400, "kind must be violation, review or note.");
      }
      const video = await store.get("SELECT id FROM videos WHERE id = ?", [seg[1]]);
      if (!video) return bad(404, "No such recording.");
      await store.run(
        `INSERT INTO intervals
           (video_id, from_ms, to_ms, kind, track_id, note, reviewer, created_utc)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          seg[1],
          from,
          to,
          kind,
          body?.track_id ?? null,
          body?.note ?? null,
          body?.reviewer ?? "local",
          new Date().toISOString(),
        ],
      );
      return ok({ ok: true });
    }

    /** The whole label set as JSONL, for the training pass. Includes
     *  reversals: a mind changed is part of the record. */
    if (method === "GET" && pathname === "/api/labels.jsonl") {
      const rows = await store.all("SELECT * FROM decisions ORDER BY id");
      return {
        status: 200,
        text:
          rows.map((r) => JSON.stringify(r)).join("\n") +
          (rows.length ? "\n" : ""),
        type: "application/x-ndjson",
      };
    }

    /* --------------------------------------------------------------- jobs */

    /**
     * Queue a run. Answers 202 and returns immediately.
     *
     * The pipeline is minutes of GPU work; holding the request open for it
     * would time out somewhere in the middle and leave nobody sure whether
     * the work was still going. The job row is the answer, and its state is
     * where the truth lives from here on.
     */
    if (method === "POST" && seg[0] === "videos" && seg[2] === "process") {
      const video = await store.get("SELECT * FROM videos WHERE id = ?", [seg[1]]);
      if (!video) return bad(404, "No such recording.");
      if (!video.media_url) {
        return bad(400, "That recording has no media to process.");
      }

      // One live job per recording. Queueing a second while the first runs
      // would have two agents writing the same run key.
      const live = await store.get(
        `SELECT * FROM jobs WHERE video_id = ?
           AND state IN ('queued','claimed','running') ORDER BY id DESC`,
        [seg[1]],
      );
      if (live) {
        return ok(
          { job: live, note: "Already queued; this is the existing job." },
          202,
        );
      }

      const at = nowUtc();
      const { lastId } = await store.run(
        `INSERT INTO jobs (video_id, state, created_utc, updated_utc)
         VALUES (?, 'queued', ?, ?)`,
        [seg[1], at, at],
      );
      return ok(
        { job: await store.get("SELECT * FROM jobs WHERE id = ?", [lastId]) },
        202,
      );
    }

    if (method === "GET" && seg[0] === "jobs" && seg[1] && !seg[2]) {
      const job = await store.get("SELECT * FROM jobs WHERE id = ?", [seg[1]]);
      return job ? ok({ job }) : bad(404, "No such job.");
    }

    if (method === "GET" && pathname === "/api/jobs") {
      return ok({
        jobs: await store.all(
          "SELECT * FROM jobs ORDER BY id DESC LIMIT 50",
        ),
      });
    }

    /**
     * The GPU agent claims the oldest queued job.
     *
     * Pull, not push. The box running the pipeline sits behind a NAT with no
     * inbound connectivity, and asking it to accept connections means a
     * tunnel, a port forward, or a public IP -- three things that can be
     * misconfigured. Polling needs none of them, and it also means an agent
     * that crashes simply stops claiming rather than dropping work on the
     * floor.
     *
     * The claim token is what the agent must present to report progress, so a
     * stale agent that wakes up after its job was reassigned cannot overwrite
     * the new one's status.
     */
    if (method === "POST" && pathname === "/api/jobs/claim") {
      // An unauthenticated claim endpoint on a public URL lets a stranger
      // drain the queue -- and worse, on a rented GPU billed by the hour, it
      // lets them spend someone else's money. The token is required whenever
      // one is configured; a deployment without one is refused rather than
      // silently open.
      if (!agentToken) {
        return bad(
          503,
          "No agent token is configured on this deployment, so the job queue " +
            "is closed. Set AGENT_TOKEN (or the Worker secret) and restart.",
        );
      }
      if (body?.agent_token !== agentToken) {
        return bad(403, "Wrong or missing agent token.");
      }
      const agent = String(body?.agent ?? "unknown").slice(0, 120);
      const job = await store.get(
        "SELECT * FROM jobs WHERE state = 'queued' ORDER BY id LIMIT 1",
      );
      if (!job) return ok({ job: null });

      const token = cryptoToken();
      const at = nowUtc();
      await store.run(
        `UPDATE jobs SET state='claimed', claimed_by=?, claim_token=?,
                         attempts = attempts + 1, updated_utc=?
         WHERE id = ? AND state = 'queued'`,
        [agent, token, at, job.id],
      );
      // Re-read: if another agent won the race the state will not be ours.
      const claimed = await store.get("SELECT * FROM jobs WHERE id = ?", [job.id]);
      if (claimed.claim_token !== token) return ok({ job: null });

      const video = await store.get("SELECT * FROM videos WHERE id = ?", [
        claimed.video_id,
      ]);
      return ok({ job: claimed, video, claim_token: token });
    }

    /** Progress and completion, reported by whoever holds the claim token. */
    if (method === "POST" && seg[0] === "jobs" && seg[2] === "status") {
      const job = await store.get("SELECT * FROM jobs WHERE id = ?", [seg[1]]);
      if (!job) return bad(404, "No such job.");
      if (!job.claim_token || body?.claim_token !== job.claim_token) {
        return bad(403, "Wrong or missing claim token for this job.");
      }

      const state = String(body?.state ?? "running");
      if (!["running", "complete", "failed"].includes(state)) {
        return bad(400, "state must be running, complete or failed.");
      }
      if (state === "complete" && !body?.run_key) {
        // A job that finishes without naming its run has produced nothing the
        // console can read, and calling that success would show an empty
        // dashboard as though the recording were clean.
        return bad(400, "A complete job must report the run_key it produced.");
      }

      const at = nowUtc();
      await store.run(
        `UPDATE jobs SET state=?, stage=?, progress=?, run_key=?, error=?,
                         updated_utc=?, finished_utc=?
         WHERE id = ?`,
        [
          state,
          body?.stage ?? job.stage,
          Number(body?.progress ?? job.progress) || 0,
          body?.run_key ?? job.run_key,
          body?.error ?? null,
          at,
          state === "running" ? null : at,
          seg[1],
        ],
      );

      // A finished run is only useful once the recording points at it. Doing
      // this here rather than in the agent keeps the agent from needing write
      // access to the hierarchy.
      if (state === "complete") {
        const key = body.run_key;
        await store.run(
          `UPDATE videos SET bundle_url=?, overlay_url=?, crops_url=?
           WHERE id = ?`,
          [
            `/data/${key}.json`,
            `/data/${key}.overlay.json`,
            `/crops/${key}`,
            job.video_id,
          ],
        );
      }

      return ok({ job: await store.get("SELECT * FROM jobs WHERE id = ?", [seg[1]]) });
    }

    /* ------------------------------------------------------------- vision */

    if (method === "POST" && pathname === "/api/vision") {
      if (!describe) return bad(503, "Image description is not configured.");
      return await describe(body ?? {});
    }

    return bad(404, `No API route for ${method} ${pathname}`);
  };
}
