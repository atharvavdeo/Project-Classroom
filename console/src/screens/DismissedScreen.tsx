import { useMemo } from "react";
import type { Bundle } from "../types";
import type { CropManifest } from "../review";
import type { DecisionRow } from "../api";
import { subjectName, timecode } from "../lib/format";
import { classLabel, headline, verdictLabel } from "../lib/humanize";
import "./DismissedScreen.css";

/**
 * What the machine claimed and a human overturned.
 *
 * This is the most valuable thing the console produces. A detection nobody
 * disputes teaches very little; a detection a reviewer looked at and rejected
 * is a labelled false positive, with the machine's own claim still attached to
 * it -- the class it proposed, what the second model said, and the frame it
 * happened on. That is the shape a later training pass needs, which is why
 * nothing here is deleted and why the export is one endpoint away.
 *
 * These people are gone from the review queue and from the video overlay. They
 * are not gone from the record.
 */
export function DismissedScreen({
  bundle,
  crops,
  cropBase,
  decisions,
  onRestore,
}: {
  bundle: Bundle;
  crops: CropManifest;
  cropBase: string;
  decisions: Record<number, DecisionRow>;
  onRestore: (trackId: number, extra?: Partial<DecisionRow>) => void;
}) {
  const rows = useMemo(
    () =>
      Object.values(decisions)
        .filter((d) => d.decision === "human_dismissed")
        .sort((a, b) => b.id - a.id)
        .map((d) => ({
          row: d,
          crop: crops[String(d.track_id)],
          track: bundle.tracks.find((t) => t.track_id === d.track_id),
        })),
    [decisions, crops, bundle.tracks],
  );

  if (!rows.length) {
    return (
      <div className="fp">
        <p className="fp__empty">
          Nothing has been marked as not a violation yet. When you do, it moves
          here — out of the queue and off the video, but kept as a labelled
          false positive for the next round of training.
        </p>
      </div>
    );
  }

  return (
    <div className="fp">
      <header className="fp__top">
        <div>
          <h2>Not violations</h2>
          <p>
            {rows.length} detection{rows.length === 1 ? "" : "s"} a reviewer
            overturned. Each one keeps what the machine claimed, so it can be
            replayed as a training example.
          </p>
        </div>
        <a className="fp__export mono" href="/api/labels.jsonl" download>
          Export labels (.jsonl)
        </a>
      </header>

      <ul className="fp__list">
        {rows.map(({ row, crop, track }) => {
          const subject = subjectName(
            row.track_id,
            track?.seat ?? null,
            track?.seat_state ?? null,
          );
          return (
            <li key={row.id}>
              {crop?.key && (
                <img src={`${cropBase}/${crop.key}`} alt={subject.name} />
              )}
              <div className="fp__what">
                <h3>{subject.name}</h3>
                <p>
                  {crop
                    ? headline(
                        crop.key_supported,
                        crop.key_class,
                        crop.key_verdict,
                      )
                    : `The machine proposed a ${classLabel(row.key_class)}.`}
                </p>
                <dl className="mono">
                  <div>
                    <dt>Machine said</dt>
                    <dd>{row.machine_state ?? "not recorded"}</dd>
                  </div>
                  <div>
                    <dt>Second model</dt>
                    <dd>{verdictLabel(row.key_verdict)}</dd>
                  </div>
                  <div>
                    <dt>At</dt>
                    <dd>{timecode(row.key_pts_ms ?? crop?.key_pts_ms)}</dd>
                  </div>
                  <div>
                    <dt>Decided</dt>
                    <dd>{new Date(row.decided_utc).toLocaleString()}</dd>
                  </div>
                </dl>
              </div>
              <button
                type="button"
                onClick={() =>
                  onRestore(row.track_id, {
                    machine_state: track?.state ?? null,
                    key_class: row.key_class,
                    key_verdict: row.key_verdict,
                    key_pts_ms: row.key_pts_ms,
                  })
                }
              >
                Put back
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
