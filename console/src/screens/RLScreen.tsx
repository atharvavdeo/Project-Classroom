import { useEffect, useMemo, useState } from "react";
import { api, type RejectedRow } from "../api";
import { timecode } from "../lib/format";
import { classLabel, verdictLabel } from "../lib/humanize";
import "./RLScreen.css";

/**
 * Everything a reviewer overturned, across every recording.
 *
 * This is the negative half of the label set. It is kept as a flat table on
 * purpose -- one row per overturned proposal, with the machine's own claim
 * beside the human's answer -- because that is the shape a training pass reads
 * and the shape a person needs in order to see a pattern across halls.
 *
 * Why a dismissal is worth more than a confirmation
 * -------------------------------------------------
 *
 * A confirmation says the system was right, which it already believed. A
 * dismissal says *what kind of thing it mistakes for evidence*, and the
 * `machine_state` / `key_class` / `key_verdict` columns are what make that
 * legible: "SAM 3 called it a keyboard, the detector called it paper, the
 * human said no" is a specific, fixable failure. Without the machine's claim
 * recorded next to the verdict, a dismissal is just the word "no" and cannot
 * train anything.
 *
 * What this is not
 * ----------------
 *
 * It is **not reinforcement learning**, whatever the tab is called. Nothing
 * here updates a policy, there is no reward signal, and no model is in a loop
 * with an environment. It is supervised labelling on a fixed corpus, and the
 * footer says so -- calling it RL in the interface and something honest in the
 * documentation would be the wrong way round.
 *
 * The rows are also **not deletions**. `docs/OUTPUT_CONTRACT.md` requires that
 * records are annotated rather than removed, so every row here still exists in
 * `decisions` with its full history, including reversals.
 */
export function RLScreen() {
  const [rows, setRows] = useState<RejectedRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [group, setGroup] = useState<string>("");

  useEffect(() => {
    let live = true;
    api
      .dismissed()
      .then((r) => live && setRows(r.rows))
      .catch((e) => live && setError(String(e instanceof Error ? e.message : e)));
    return () => {
      live = false;
    };
  }, []);

  // What the machine claimed, counted. This is the useful summary: not "how
  // many were wrong" but "wrong about what".
  const byClaim = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of rows ?? []) {
      const key = r.key_class ?? "no object claim";
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [rows]);

  const shown = useMemo(
    () => (group ? (rows ?? []).filter((r) => (r.key_class ?? "no object claim") === group) : rows ?? []),
    [rows, group],
  );

  if (error) {
    return (
      <section className="rl">
        <h2>Rejected proposals</h2>
        <p className="rl__error">Could not load the label set: <span className="mono">{error}</span></p>
      </section>
    );
  }

  if (!rows) {
    return (
      <section className="rl">
        <h2>Rejected proposals</h2>
        <p className="rl__empty mono">Loading…</p>
      </section>
    );
  }

  if (!rows.length) {
    return (
      <section className="rl">
        <h2>Rejected proposals</h2>
        <p className="rl__empty">
          Nothing has been overturned yet. This fills up as reviewers answer
          &ldquo;not a violation&rdquo; — every one of those is a labelled
          negative, and they are the examples worth the most.
        </p>
      </section>
    );
  }

  return (
    <section className="rl">
      <h2>
        Rejected proposals
        <span className="mono">
          {rows.length} overturned · {byClaim.length} distinct claims
        </span>
        {/* A plain link, not a fetch-and-Blob dance: the route already sets
            content-disposition, so the browser saves it, and the same URL is
            usable from curl or a notebook without going through this page. */}
        <a className="rl__export" href="/api/dismissed.csv" download>
          Export CSV
        </a>
      </h2>

      <p className="rl__lede">
        Every proposal a human answered <b>no</b> to, with what the machine was
        claiming at the moment they answered. A dismissal without the claim
        beside it is just the word &ldquo;no&rdquo; and cannot train anything.
      </p>

      <div className="rl__chips">
        <button
          type="button"
          className={group === "" ? "is-on" : ""}
          onClick={() => setGroup("")}
        >
          all ({rows.length})
        </button>
        {byClaim.map(([claim, n]) => (
          <button
            key={claim}
            type="button"
            className={group === claim ? "is-on" : ""}
            onClick={() => setGroup(group === claim ? "" : claim)}
          >
            {classLabel(claim)} ({n})
          </button>
        ))}
      </div>

      <div className="rl__scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Recording</th>
              <th scope="col">Person</th>
              <th scope="col">At</th>
              <th scope="col">Machine claimed</th>
              <th scope="col">Second model</th>
              <th scope="col">State when asked</th>
              <th scope="col">Reviewer</th>
              <th scope="col">Answered</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.id}>
                <td>
                  {r.video_name}
                  {r.centre_name && (
                    <span className="rl__sub mono">{r.centre_name}</span>
                  )}
                </td>
                <td className="mono">#{r.track_id}</td>
                <td className="mono">
                  {r.key_pts_ms != null ? timecode(r.key_pts_ms) : "—"}
                </td>
                <td>{r.key_class ? classLabel(r.key_class) : <span className="rl__na mono">no object claim</span>}</td>
                {/* The referee's own word. Where it disagrees with the column
                    to its left, that pair is the single most informative thing
                    on this page. */}
                <td>{r.key_verdict ? verdictLabel(r.key_verdict) : <span className="rl__na mono">not adjudicated</span>}</td>
                <td className="mono">{r.machine_state ?? "—"}</td>
                <td className="mono">{r.reviewer}</td>
                <td className="mono">{r.decided_utc.slice(0, 16).replace("T", " ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <footer className="rl__foot">
        <p>
          <b>This is not reinforcement learning</b>, whatever the tab is called.
          No policy is updated, there is no reward, and no model is in a loop
          with an environment. It is supervised labelling on a fixed corpus, and
          the labels are exported by{" "}
          <a href="/api/labels.jsonl">
            <code>/api/labels.jsonl</code>
          </a>{" "}
          — which includes reversals, because a reviewer changing their mind is
          part of the record.
        </p>
        <p>
          Nothing here was deleted. Every row still exists in <code>decisions</code>{" "}
          with its full history; a dismissal annotates a record, it does not
          remove one.
        </p>
      </footer>
    </section>
  );
}
