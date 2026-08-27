import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Bundle, Track } from "../types";
import type { CropEntry, CropManifest, Decision } from "../review";
import type { DecisionRow } from "../api";
import { api } from "../api";
import { StateChip } from "../components/StateChip";
import { seconds, subjectName, timecode } from "../lib/format";
import {
  classLabel,
  conditionMeaning,
  conditionQuestion,
  decisionLabel,
  flagLabel,
  headline,
  verdictLabel,
} from "../lib/humanize";
import "./ReviewScreen.css";

interface Gemma {
  state: "idle" | "running" | "done" | "error";
  title?: string;
  description?: string;
  object_guess?: string;
  confidence?: string;
  error?: string;
}

/** Fetch an image and turn it into the data: URL the proxy expects. Done in
 *  the client so the server never has to reach back into the crop folder. */
async function toDataUrl(url: string): Promise<string> {
  const blob = await (await fetch(url)).blob();
  return await new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result));
    fr.onerror = () => reject(fr.error);
    fr.readAsDataURL(blob);
  });
}

type View = "key" | "wide" | "sam3";

/**
 * One person, one screen, one decision.
 *
 * The queue holds only people nobody has answered for yet. A decision takes
 * its subject out of it immediately -- that is the answer to "what happened
 * when I pressed Confirm", and it is also what makes the dismissals a usable
 * false-positive set rather than a filter the reviewer has to remember to
 * apply. The answer is announced, and reversible for as long as it is on
 * screen, because a queue that silently swallows a keystroke cannot be
 * trusted with the next one.
 */
export function ReviewScreen({
  bundle,
  crops,
  cropBase,
  decisions,
  onDecide,
}: {
  bundle: Bundle;
  crops: CropManifest;
  cropBase: string;
  decisions: Record<number, DecisionRow>;
  onDecide: (
    trackId: number,
    decision: Decision,
    extra?: Partial<DecisionRow>,
  ) => Promise<void>;
}) {
  const [showDecided, setShowDecided] = useState(false);
  const [index, setIndex] = useState(0);
  const [showWorking, setShowWorking] = useState(false);
  const [gemma, setGemma] = useState<Record<number, Gemma>>({});
  const [view, setView] = useState<View>("key");
  const [frame, setFrame] = useState<number | null>(null);
  const [flash, setFlash] = useState<{ track: number; d: Decision } | null>(
    null,
  );
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Only people with a picture can be reviewed. Everyone else is in the
  // register, not the queue -- asking for a verdict on a person we cannot show
  // would be asking someone to guess.
  const all = useMemo(() => {
    const withCrops = bundle.tracks.filter((t) => crops[String(t.track_id)]);
    return withCrops.sort((a, b) => {
      const ca = crops[String(a.track_id)];
      const cb = crops[String(b.track_id)];
      // Confirmed detections first, then priority. That is the order a
      // reviewer's attention is worth most.
      if (ca.key_supported !== cb.key_supported) return ca.key_supported ? -1 : 1;
      return (b.priority ?? 0) - (a.priority ?? 0);
    });
  }, [bundle.tracks, crops]);

  const queue = useMemo(
    () => (showDecided ? all : all.filter((t) => !decisions[t.track_id])),
    [all, decisions, showDecided],
  );

  // The queue shrinks under the cursor as answers land. Clamping here keeps
  // the last decision from leaving the reviewer past the end.
  const at = Math.min(index, Math.max(queue.length - 1, 0));
  const track: Track | undefined = queue[at];
  const crop: CropEntry | undefined = track
    ? crops[String(track.track_id)]
    : undefined;

  const trackKey = track?.track_id;
  useEffect(() => {
    setView("key");
    setFrame(null);
    setShowWorking(false);
  }, [trackKey]);

  const step = useCallback(
    (delta: number) =>
      setIndex((i) =>
        Math.max(0, Math.min(queue.length - 1, Math.min(i, queue.length - 1) + delta)),
      ),
    [queue.length],
  );

  const flashTimer = useRef<number | null>(null);
  const decide = useCallback(
    async (d: Decision) => {
      if (!track || saving) return;
      setSaving(true);
      setSaveError(null);
      try {
        await onDecide(track.track_id, d, {
          machine_state: track.state,
          key_class: crop?.key_class ?? null,
          key_verdict: crop?.key_verdict ?? null,
          key_pts_ms: crop?.key_pts_ms ?? null,
        });
        setFlash({ track: track.track_id, d });
        if (flashTimer.current) window.clearTimeout(flashTimer.current);
        flashTimer.current = window.setTimeout(() => setFlash(null), 6000);
      } catch (e) {
        setSaveError(String(e instanceof Error ? e.message : e));
      } finally {
        setSaving(false);
      }
    },
    [track, crop, onDecide, saving],
  );

  const undo = useCallback(async () => {
    if (!flash) return;
    // An undo is another append, not an erasure. The store keeps the reversal
    // because "confirmed then dismissed" is a more informative training
    // example than either answer alone.
    const t = all.find((x) => x.track_id === flash.track);
    await onDecide(flash.track, "needs_better_view", {
      machine_state: t?.state ?? null,
      note: "reverted in review",
    });
    setFlash(null);
  }, [flash, all, onDecide]);

  const runGemma = useCallback(async () => {
    if (!track || !crop) return;
    // Always the unannotated crop: a model shown our own boxes describes them
    // back as objects, which is not a second opinion.
    const source = crop.raw ?? crop.key;
    if (!source) return;
    const id = track.track_id;
    if (gemma[id]?.state === "running" || gemma[id]?.state === "done") return;
    setGemma((g) => ({ ...g, [id]: { state: "running" } }));
    try {
      const doc = await api.describe(await toDataUrl(`${cropBase}/${source}`));
      setGemma((g) => ({
        ...g,
        [id]: doc.parsed
          ? { state: "done", ...doc.parsed }
          : { state: "done", description: doc.raw ?? "No answer." },
      }));
    } catch (e) {
      setGemma((g) => ({
        ...g,
        [id]: { state: "error", error: String(e instanceof Error ? e.message : e) },
      }));
    }
  }, [track, crop, cropBase, gemma]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName)) return;
      if (e.key === "ArrowRight" || e.key === "n") step(1);
      else if (e.key === "ArrowLeft" || e.key === "p") step(-1);
      else if (e.key === "1" || e.key === "c") void decide("human_confirmed");
      else if (e.key === "2" || e.key === "d") void decide("human_dismissed");
      else if (e.key === "3" || e.key === "b") void decide("needs_better_view");
      else if (e.key === "g") void runGemma();
      else if (e.key === "w") setShowWorking((s) => !s);
      else if (e.key === "u") void undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, decide, runGemma, undo]);

  const tally = useMemo(() => {
    const t = { human_confirmed: 0, human_dismissed: 0, needs_better_view: 0 };
    for (const row of Object.values(decisions)) t[row.decision] += 1;
    return t;
  }, [decisions]);

  const banner = flash && (
    <div className={`rv__flash is-${flash.d}`} role="status">
      <b>
        Person #{flash.track} — {decisionLabel(flash.d)}.
      </b>
      <span>
        {flash.d === "human_dismissed"
          ? "Out of the queue and off the video. It is kept as a false positive."
          : flash.d === "human_confirmed"
            ? "Kept as a confirmed finding."
            : "Left open for a second look."}
      </span>
      <button type="button" onClick={() => void undo()}>
        Undo <kbd>u</kbd>
      </button>
    </div>
  );

  if (!all.length) {
    return (
      <p className="rv__empty">
        No person in this run has a reviewable frame. Build crops with
        <code> tools/build_review_crops.py</code>.
      </p>
    );
  }

  if (!track || !crop) {
    return (
      <div className="rv">
        {banner}
        <p className="rv__empty">
          Everyone with a picture has been answered for:{" "}
          <b>{tally.human_confirmed} confirmed</b>,{" "}
          <b>{tally.human_dismissed} not violations</b>,{" "}
          <b>{tally.needs_better_view} left open</b>.
        </p>
        <button
          type="button"
          className="rv__toggle"
          onClick={() => {
            setShowDecided(true);
            setIndex(0);
          }}
        >
          Look back through all {all.length}
        </button>
      </div>
    );
  }

  const subject = subjectName(track.track_id, track.seat, track.seat_state);
  const decided = decisions[track.track_id];
  const g = gemma[track.track_id];

  const strip = crop.strip;
  const shown =
    frame !== null && strip[frame]
      ? { src: strip[frame].file, caption: timecode(strip[frame].pts_ms) }
      : view === "wide" && crop.frame
        ? { src: crop.frame, caption: `${timecode(crop.key_pts_ms)} · whole frame` }
        : view === "sam3" && crop.sam3_crop
          ? {
              src: `sam3/${crop.sam3_crop}`,
              caption: `${timecode(crop.key_pts_ms)} · ${verdictLabel(crop.key_verdict)}`,
            }
          : {
              src: crop.key ?? "",
              caption: `${timecode(crop.key_pts_ms)} · ${classLabel(crop.key_class)}${
                crop.key_confidence != null
                  ? ` · detector ${(crop.key_confidence * 100).toFixed(0)}%`
                  : ""
              }`,
            };

  return (
    <div className="rv">
      {banner}

      <header className="rv__top">
        <div className="rv__progress">
          <b className="mono">
            {at + 1} / {queue.length}
          </b>
          <span>
            {tally.human_confirmed} confirmed · {tally.human_dismissed} not
            violations · {tally.needs_better_view} open
          </span>
          <div className="rv__bar">
            <i style={{ width: `${((at + 1) / queue.length) * 100}%` }} />
          </div>
        </div>
        <div className="rv__nav">
          <label className="rv__see">
            <input
              type="checkbox"
              checked={showDecided}
              onChange={(e) => {
                setShowDecided(e.target.checked);
                setIndex(0);
              }}
            />
            Include answered
          </label>
          <button type="button" onClick={() => step(-1)} disabled={at === 0}>
            Previous
          </button>
          <button
            type="button"
            onClick={() => step(1)}
            disabled={at === queue.length - 1}
          >
            Next
          </button>
        </div>
      </header>

      <div className="rv__body">
        <figure className="rv__shot">
          <div className="rv__views">
            <button
              type="button"
              className={view === "key" && frame === null ? "is-on" : ""}
              onClick={() => {
                setView("key");
                setFrame(null);
              }}
            >
              Key frame
            </button>
            {crop.frame && (
              <button
                type="button"
                className={view === "wide" && frame === null ? "is-on" : ""}
                onClick={() => {
                  setView("wide");
                  setFrame(null);
                }}
              >
                Whole frame
              </button>
            )}
            {/* SAM 3's own segmentation for this instant. It exists only where
                the referee was actually called, so the control appears only
                then -- a disabled tab would imply the picture exists. */}
            {crop.sam3_crop && (
              <button
                type="button"
                className={view === "sam3" && frame === null ? "is-on" : ""}
                onClick={() => {
                  setView("sam3");
                  setFrame(null);
                }}
              >
                What the second model saw
              </button>
            )}
            {frame !== null && (
              <button type="button" className="is-on" onClick={() => setFrame(null)}>
                Sampled moment · back to key frame
              </button>
            )}
          </div>
          <img
            src={`${cropBase}/${shown.src}`}
            alt={`${subject.name} at ${shown.caption}`}
          />
          <figcaption className="mono">{shown.caption}</figcaption>
        </figure>

        <div className="rv__side">
          <div className="rv__who">
            <h2>{subject.name}</h2>
            {subject.qualifier && (
              <span className="rv__qual">{subject.qualifier}</span>
            )}
            <StateChip state={decided ? decided.decision : track.state} />
          </div>

          <p className={`rv__headline ${crop.key_supported ? "is-strong" : ""}`}>
            {headline(crop.key_supported, crop.key_class, crop.key_verdict)}
          </p>

          <ul className="rv__facts">
            <li>
              <span>Second model</span>
              <b>{verdictLabel(crop.key_verdict)}</b>
            </li>
            {/* `proposals` counts every object seen near this person's hands
                across the whole track, most of it keyboards and monitors. It
                is context, not a denominator -- "5 of 8,729" reads as a 0.06%
                hit rate, which is not what either number means. */}
            <li>
              <span>Frames it confirmed</span>
              <b>{crop.supported_frames.toLocaleString()}</b>
            </li>
            <li>
              <span>Times seen</span>
              <b>{crop.sightings.toLocaleString()}</b>
            </li>
            <li>
              <span>Distance to hand</span>
              <b>
                {crop.wrist_distance_norm != null && crop.wrist_distance_norm >= 0
                  ? `${crop.wrist_distance_norm.toFixed(2)} body widths`
                  : "not recorded"}
              </b>
            </li>
            <li>
              <span>Seen for</span>
              <b>
                {seconds((track.last_seen_ms ?? 0) - (track.first_seen_ms ?? 0))}
              </b>
            </li>
          </ul>

          {track.flags.length > 0 && (
            <details className="rv__flags">
              <summary>
                {track.flags.length} flag{track.flags.length === 1 ? "" : "s"}
              </summary>
              <ul>
                {track.flags.map((f) => (
                  <li key={f.flag}>
                    <b>{flagLabel(f.flag)}</b>
                    <span>{f.evidence}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}

          <div className="rv__gemma">
            <button
              type="button"
              onClick={() => void runGemma()}
              disabled={g?.state === "running" || g?.state === "done"}
            >
              {g?.state === "running"
                ? "Reading the image…"
                : g?.state === "done"
                  ? "Analysed"
                  : "Ask Gemma what it sees"}
            </button>
            {g?.state === "done" && (
              <div className="rv__answer">
                {g.title && <h4>{g.title}</h4>}
                <p>{g.description}</p>
                {g.object_guess && (
                  <span className="mono">
                    reads as {classLabel(g.object_guess)} &middot; {g.confidence}
                  </span>
                )}
                <em>A description of the picture. Not a verdict.</em>
              </div>
            )}
            {g?.state === "error" && <p className="rv__err mono">{g.error}</p>}
          </div>
        </div>
      </div>

      {saveError && (
        <p className="rv__err mono">Not saved: {saveError}</p>
      )}

      <div className="rv__actions">
        <button
          type="button"
          className="is-confirm"
          disabled={saving}
          onClick={() => void decide("human_confirmed")}
        >
          Confirm <kbd>1</kbd>
        </button>
        <button
          type="button"
          className="is-dismiss"
          disabled={saving}
          onClick={() => void decide("human_dismissed")}
        >
          Not a violation <kbd>2</kbd>
        </button>
        <button
          type="button"
          className="is-abstain"
          disabled={saving}
          onClick={() => void decide("needs_better_view")}
        >
          Can't tell <kbd>3</kbd>
        </button>
        <button
          type="button"
          className="is-plain"
          onClick={() => setShowWorking((s) => !s)}
        >
          {showWorking ? "Hide" : "Show"} the reasoning <kbd>w</kbd>
        </button>
      </div>

      {showWorking && (
        <section className="rv__working">
          <h3>How the system got here</h3>
          <ol>
            {track.conditions.map((c) => (
              <li key={c.name} className={c.passed ? "is-pass" : "is-fail"}>
                <div className="rv__q">
                  <span>{c.passed ? "Yes" : "No"}</span>
                  <b>{conditionQuestion(c.name)}</b>
                </div>
                <p>{conditionMeaning(c.name, c.passed)}</p>
                <p className="rv__detail mono">{c.detail}</p>
              </li>
            ))}
          </ol>
        </section>
      )}

      {strip.length > 0 && (
        <section className="rv__strip">
          <h3>
            Every sampled moment
            <span className="mono">
              {strip.length} of {crop.sightings} sightings · click one to open it
            </span>
          </h3>
          <div className="rv__thumbs">
            {strip.map((s, i) => (
              <button
                key={s.file}
                type="button"
                className={[
                  s.supported ? "is-strong" : "",
                  frame === i ? "is-on" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                title={`${timecode(s.pts_ms)} · ${s.n} proposal(s)`}
                aria-pressed={frame === i}
                onClick={() => setFrame(frame === i ? null : i)}
              >
                <img src={`${cropBase}/${s.file}`} alt={timecode(s.pts_ms)} />
                <span className="mono">{timecode(s.pts_ms)}</span>
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
