import { useMemo, useState } from "react";
import type { Bundle } from "../types";
import type { CropManifest, Decision } from "../review";
import type { OverlayBundle } from "../overlay";
import { deviationEpisodes, AWAY_LABEL } from "../lib/deviation";
import { subjectName, timecode } from "../lib/format";
import { classLabel } from "../lib/humanize";
import "./FindingsTable.css";

type Tier = "human" | "second_model" | "proposed" | "attention";

interface Row {
  key: string;
  track_id: number;
  at_ms: number;
  tier: Tier;
  what: string;
  /** What backs the claim. Never a restatement of `what`. */
  backing: string;
  /** SAM 3's own annotated crop for this instant, when it exists. */
  sam3?: string;
}

/** Short standing chip. The tier is carried by a word and a colour, not by a
 *  sentence in the middle of the row. */
const STANDING: Record<Tier, string> = {
  human: "Confirmed",
  second_model: "SAM 3",
  proposed: "Detector",
  attention: "Observed",
};

/**
 * What was caught, in one list, with the video one click away.
 *
 * Four tiers, kept apart on purpose. Collapsing them into a single
 * "detections" count is exactly what the output contract exists to prevent: a
 * proposal is not a finding, a second model agreeing is not a person having
 * cheated, and where someone was looking is not either of those.
 *
 * The attention rows are a **different condition with different criteria** --
 * duration and repetition, not object geometry -- and they can never route
 * someone to review on their own. The table says so under it rather than
 * letting the shared layout imply equivalence.
 */
export function FindingsTable({
  bundle,
  crops,
  cropBase,
  overlay,
  decisions,
  onSeek,
  focus,
}: {
  bundle: Bundle;
  crops: CropManifest;
  cropBase: string;
  overlay: OverlayBundle | null;
  decisions: Record<number, Decision>;
  onSeek: (ms: number, trackId: number, label: string, standing: string) => void;
  focus: number | null;
}) {
  const [showAttention, setShowAttention] = useState(true);
  const [zoom, setZoom] = useState<string | null>(null);

  const objectRows = useMemo<Row[]>(() => {
    const rows: Row[] = [];

    for (const track of bundle.tracks) {
      const crop = crops[String(track.track_id)];
      const decision = decisions[track.track_id];
      if (decision === "human_dismissed") continue;

      const sam3 = crop?.sam3_crop ? `sam3/${crop.sam3_crop}` : undefined;

      if (decision === "human_confirmed") {
        rows.push({
          key: `h${track.track_id}`,
          track_id: track.track_id,
          at_ms: crop?.key_pts_ms ?? track.first_seen_ms ?? 0,
          tier: "human",
          what: `${classLabel(crop?.key_class)} at the hand`,
          // The reviewer's own verdict is the standing; the backing is what
          // the machine had to say, which is the useful extra fact.
          backing: crop?.key_supported
            ? `SAM 3 agreed · ${crop.supported_frames} of ${crop.sightings} frames`
            : "detector proposal only",
          sam3,
        });
        continue;
      }

      if (crop?.key_supported) {
        rows.push({
          key: `s${track.track_id}`,
          track_id: track.track_id,
          at_ms: crop.key_pts_ms,
          tier: "second_model",
          // `what` names the object; `backing` counts the evidence. It used to
          // repeat the verdict here -- "phone at the hand / Second model
          // called it a phone" -- which is the same sentence twice and reads
          // as though two different things were established.
          what: `${classLabel(crop.key_class)} at the hand`,
          backing: `${crop.supported_frames} of ${crop.sightings} frames`,
          sam3,
        });
        continue;
      }

      if (track.state === "review_candidate" && crop) {
        rows.push({
          key: `p${track.track_id}`,
          track_id: track.track_id,
          at_ms: crop.key_pts_ms,
          tier: "proposed",
          what: `possible ${classLabel(crop.key_class)}`,
          backing:
            crop.key_confidence != null
              ? `${(crop.key_confidence * 100).toFixed(0)}% · not backed`
              : "not backed",
          sam3,
        });
      }
    }

    const rank = { human: 0, second_model: 1, proposed: 2, attention: 3 };
    return rows.sort(
      (a, b) => rank[a.tier] - rank[b.tier] || a.at_ms - b.at_ms,
    );
  }, [bundle.tracks, crops, decisions]);

  const deviation = useMemo(() => deviationEpisodes(overlay), [overlay]);

  const attention = useMemo(() => {
    const live = new Set(
      bundle.tracks
        .filter((t) => decisions[t.track_id] !== "human_dismissed")
        .map((t) => t.track_id),
    );
    return deviation.episodes.filter((e) => live.has(e.track_id));
  }, [deviation, bundle.tracks, decisions]);

  const attentionRows = useMemo<Row[]>(
    () =>
      attention.map((e, i) => ({
        key: `a${e.track_id}-${i}`,
        track_id: e.track_id,
        // The peak, not the start. The first frame of a turn is a head
        // halfway round; the reviewer wants the frame where it got furthest.
        at_ms: e.peak_ms,
        tier: "attention" as const,
        what:
          e.with_track != null
            ? `turned ${AWAY_LABEL[e.away]}, with #${e.with_track}`
            : `turned ${AWAY_LABEL[e.away]}`,
        backing: e.evidence,
      })),
    [attention],
  );

  const rows = showAttention ? [...objectRows, ...attentionRows] : objectRows;

  const counts = {
    human: objectRows.filter((r) => r.tier === "human").length,
    second_model: objectRows.filter((r) => r.tier === "second_model").length,
    proposed: objectRows.filter((r) => r.tier === "proposed").length,
    attention: attentionRows.length,
  };

  const nameOf = (trackId: number) => {
    const t = bundle.tracks.find((x) => x.track_id === trackId);
    return subjectName(trackId, t?.seat ?? null, t?.seat_state ?? null).name;
  };

  if (!rows.length) {
    return (
      <section className="ft">
        <h3>What was caught</h3>
        <p className="ft__none">
          Nothing in this recording cleared the bar — no object was backed by
          the second model, nobody has been confirmed by a reviewer, and nobody
          turned away from their own resting position for long enough to
          measure. That is not the same as
          &ldquo;nothing happened&rdquo;.
        </p>
      </section>
    );
  }

  return (
    <section className="ft">
      <h3>
        What was caught
        <span className="mono">
          {counts.human} confirmed · {counts.second_model} backed by SAM 3 ·{" "}
          {counts.proposed} proposed · {counts.attention} turned away
        </span>
      </h3>

      <div className="ft__scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">At</th>
              <th scope="col">Standing</th>
              <th scope="col">Who</th>
              <th scope="col">What</th>
              <th scope="col">Backing</th>
              <th scope="col">SAM 3</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.key}
                className={`is-${r.tier} ${focus === r.track_id ? "is-on" : ""}`}
                onClick={() => onSeek(r.at_ms, r.track_id, r.what, STANDING[r.tier])}
                tabIndex={0}
                role="button"
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSeek(r.at_ms, r.track_id, r.what, STANDING[r.tier]);
                  }
                }}
              >
                <td className="mono">{timecode(r.at_ms)}</td>
                <td>
                  <span className="ft__chip">{STANDING[r.tier]}</span>
                </td>
                <td>{nameOf(r.track_id)}</td>
                <td>{r.what}</td>
                <td className="ft__detail">{r.backing}</td>
                <td>
                  {/* SAM 3's own segmentation for that instant. It exists only
                      where the referee was actually called, so an empty cell
                      means "not adjudicated", not "found nothing". */}
                  {r.sam3 ? (
                    <button
                      type="button"
                      className="ft__thumb"
                      title="What SAM 3 looked at"
                      onClick={(e) => {
                        e.stopPropagation();
                        setZoom(`${cropBase}/${r.sam3}`);
                      }}
                    >
                      <img src={`${cropBase}/${r.sam3}`} alt="SAM 3 segmentation" />
                    </button>
                  ) : (
                    <span className="ft__na mono">not adjudicated</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="ft__foot">
        <label className="ft__toggle">
          <input
            type="checkbox"
            checked={showAttention}
            onChange={(e) => setShowAttention(e.target.checked)}
          />
          Show people turning away ({counts.attention})
        </label>
        <p className="ft__note">
          Click a row to jump the video there. <b>Turning away is not a
          finding</b> — it is measured on its own criteria and can never send
          someone to review by itself. A turn is counted only where the head
          leaves <i>that person&rsquo;s own</i> resting position by 3× their own
          spread for at least 1.2s, so a chair that permanently faces sideways
          scores zero. Two people turning within the same seconds are marked as
          a pair. This is head-versus-shoulders, not gaze.
        </p>
        {/* An abstention is not a clean result, so it is stated rather than
            left as an absence of rows. */}
        {deviation.noBaseline.length > 0 && (
          <p className="ft__note">
            <b>Not judged for turning: {deviation.noBaseline.length} of{" "}
            {deviation.noBaseline.length + deviation.baselines.size} people.</b>{" "}
            They were on screen too briefly, or too small, to establish a
            resting position to compare against. That is not the same as
            &ldquo;they never turned&rdquo;.
          </p>
        )}
      </div>

      {zoom && (
        <div
          className="ft__zoom"
          role="dialog"
          aria-label="SAM 3 segmentation"
          onClick={() => setZoom(null)}
        >
          <img src={zoom} alt="SAM 3 segmentation, enlarged" />
          <button type="button" onClick={() => setZoom(null)}>
            Close
          </button>
        </div>
      )}
    </section>
  );
}
