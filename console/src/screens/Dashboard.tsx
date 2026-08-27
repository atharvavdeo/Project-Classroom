import type { Bundle } from "../types";
import type { CropManifest, Decision, VideoRef } from "../review";
import type { ReactNode } from "react";
import { StateChip } from "../components/StateChip";
import { seconds, withDenominator } from "../lib/format";
import "./Dashboard.css";

const STATE_COLOR: Record<string, string> = {
  review_candidate: "var(--s-review)",
  needs_better_view: "var(--s-abstain)",
  context_observation: "var(--s-context)",
  no_action: "var(--s-none)",
};

/**
 * The run at a glance, and the way into reviewing it.
 *
 * The one number this screen refuses to show is a rate of anything per
 * candidate. Every tile carries a denominator and a plain-English caption,
 * because a bare figure on a dashboard is the exact shape that ends up in a
 * slide as an accuracy claim.
 */
export function Dashboard({
  bundle,
  crops,
  video,
  decisions,
  onReview,
  upload,
}: {
  bundle: Bundle;
  crops: CropManifest;
  video: VideoRef;
  decisions: Record<number, Decision>;
  onReview: () => void;
  /** The add-a-recording panel. Passed in rather than built here so the same
   *  panel serves the empty console, where there is no dashboard to sit on. */
  upload?: ReactNode;
}) {
  const counts = bundle.counts.states;
  const total = bundle.counts.tracks;
  const reviewable = Object.keys(crops).length;
  const confirmed = Object.values(crops).filter((c) => c.key_supported).length;
  const decided = Object.keys(decisions).length;
  const abstain = withDenominator(counts.needs_better_view ?? 0, total);

  const bars = (
    ["review_candidate", "needs_better_view", "context_observation", "no_action"] as const
  )
    .map((s) => ({ state: s, n: counts[s] ?? 0 }))
    .filter((b) => b.n > 0);

  return (
    <div className="db">
      {upload}

      <section className="db__tiles">
        <article className="tile tile--accent">
          <p>Needs a human look</p>
          <b className="mono">{reviewable}</b>
          <span>
            people with a frame worth showing, of {total} tracked
          </span>
          <button type="button" onClick={onReview}>
            Start reviewing
          </button>
        </article>

        <article className="tile">
          <p>Confirmed by second model</p>
          <b className="mono">{confirmed}</b>
          <span>
            SAM 3 backed the object claim on {confirmed} of {reviewable}
          </span>
        </article>

        <article className="tile">
          <p>Decided so far</p>
          <b className="mono">{decided}</b>
          <span>{reviewable - decided} still waiting</span>
        </article>

        <article className="tile">
          <p>Recording length</p>
          <b className="mono">{seconds(bundle.duration_ms)}</b>
          <span>{bundle.counts.samples ?? "—"} sampled frames</span>
        </article>
      </section>

      <section className="db__row">
        <article className="panel">
          <header>
            <h3>What the system concluded</h3>
            <span className="mono">{total} people</span>
          </header>

          <div className="db__stack">
            {bars.map((b) => (
              <i
                key={b.state}
                style={{
                  width: `${(b.n / total) * 100}%`,
                  background: STATE_COLOR[b.state],
                }}
                title={`${b.state}: ${b.n}`}
              />
            ))}
          </div>

          <ul className="db__legend">
            {bars.map((b) => (
              <li key={b.state}>
                <StateChip state={b.state} size="sm" />
                <b className="mono">{b.n}</b>
              </li>
            ))}
          </ul>

          <p className="db__note">
            The system declined to judge <b>{abstain.text}</b> of the people it
            saw. That is the honest headline, not a gap to read past.
          </p>
        </article>

        <article className="panel">
          <header>
            <h3>This video</h3>
          </header>
          <dl className="db__facts">
            <div>
              <dt>Name</dt>
              <dd>{video.name}</dd>
            </div>
            <div>
              <dt>Run</dt>
              <dd className="mono">{bundle.run.run_id ?? "—"}</dd>
            </div>
            <div>
              <dt>Code</dt>
              <dd className="mono">
                {(bundle.run.code_revision ?? "—").slice(0, 7)}
                {bundle.run.code_dirty ? " (dirty)" : ""}
              </dd>
            </div>
            <div>
              <dt>Seat calibration</dt>
              <dd>Not approved — people are unattributed</dd>
            </div>
            <div>
              <dt>Accuracy</dt>
              <dd>Not measurable yet — no evaluation pack</dd>
            </div>
          </dl>
        </article>
      </section>

      <section className="panel">
        <header>
          <h3>Pipeline</h3>
          <span className="mono">{bundle.stages.length} stages recorded</span>
        </header>
        {bundle.stages.length ? (
          <ul className="db__stages">
            {bundle.stages.map((s) => (
              <li key={s.stage} className={`is-${s.state}`}>
                <b className="mono">{s.stage}</b>
                <span>{s.state.replace(/_/g, " ")}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="db__note">
            No stage record was written for this run folder.
          </p>
        )}
      </section>
    </div>
  );
}
