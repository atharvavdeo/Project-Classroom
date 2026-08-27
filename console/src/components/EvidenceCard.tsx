import type { Track } from "../types";
import { StateChip } from "./StateChip";
import { ConditionLedger } from "./ConditionLedger";
import { FlagTimeline } from "./FlagTimeline";
import { seconds, subjectName, timecode, withDenominator } from "../lib/format";
import "./EvidenceCard.css";

/**
 * One track, as the reviewer's unit of work.
 *
 * `priority` always ships with its disclaimer in the same breath. It is a
 * queue order derived from modality count and association; it is not a score,
 * not a confidence, and not a probability of anything. Showing the bare number
 * would invite exactly the reading the output contract forbids.
 */
export function EvidenceCard({
  track,
  duration_ms,
  onOpen,
}: {
  track: Track;
  duration_ms: number;
  onOpen?: (id: number) => void;
}) {
  const subject = subjectName(track.track_id, track.seat, track.seat_state);
  const observed = withDenominator(track.resolvable_samples, track.samples);
  const longest = track.episodes.reduce(
    (best, e) => Math.max(best, e.to_ms - e.from_ms),
    0,
  );

  return (
    <article className="card">
      <header className="card__head">
        <div className="card__id">
          <h3>{subject.name}</h3>
          {subject.qualifier && (
            <span className="card__qual">{subject.qualifier}</span>
          )}
        </div>
        <StateChip state={track.state} />
      </header>

      <div className="card__meta mono">
        <span>
          priority {track.priority?.toFixed(2) ?? "--"}
          <em> queue order, not a score</em>
        </span>
        <span>
          seen {timecode(track.first_seen_ms)}&ndash;{timecode(track.last_seen_ms)}
        </span>
        {longest > 0 && <span>longest episode {seconds(longest)}</span>}
      </div>

      <div className="card__obs mono">
        head resolved in {observed.text}
        {!observed.measurable && <em> &mdash; nothing here is measurable</em>}
      </div>

      <ConditionLedger conditions={track.conditions} />

      {track.reason_codes.length > 0 && (
        <ul className="card__codes">
          {track.reason_codes.map((code) => (
            <li key={code} className="mono">
              {code}
            </li>
          ))}
        </ul>
      )}

      <div className="card__tl">
        <FlagTimeline track={track} duration_ms={duration_ms} compact />
      </div>

      {onOpen && (
        <footer className="card__foot">
          <button type="button" onClick={() => onOpen(track.track_id)}>
            Open person
          </button>
          <span className="card__hint">
            Review controls arrive with the decision store
          </span>
        </footer>
      )}
    </article>
  );
}
