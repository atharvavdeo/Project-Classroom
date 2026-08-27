import { useState } from "react";
import type { CropEntry } from "../review";
import { timecode } from "../lib/format";
import { classLabel, verdictLabel } from "../lib/humanize";
import "./Spotlight.css";

/**
 * The close-up, over the frame it came from.
 *
 * The wide shot answers *whose object was that* and nothing else — at hall
 * scale a phone is thirty pixels, and no amount of squinting at a 1280x720
 * frame settles it. The crop answers *what is it* and loses the context.
 * Putting the crop on the frame is the only arrangement where a reviewer can
 * answer both without moving.
 *
 * It minimises rather than closes, because a reviewer scrubbing through a
 * recording wants the wide shot back for a moment and then wants the evidence
 * again. Closing it entirely would mean re-selecting the row.
 */
export function Spotlight({
  trackId,
  crop,
  cropBase,
  label,
  standing,
  onClose,
  onSeek,
}: {
  trackId: number;
  crop: CropEntry | undefined;
  cropBase: string;
  label: string;
  standing: string;
  onClose: () => void;
  onSeek: (ms: number) => void;
}) {
  const [minimised, setMinimised] = useState(false);
  // Which picture: what the detector saw, or what the referee looked at.
  const [view, setView] = useState<"key" | "sam3">("key");

  if (!crop?.key) return null;

  const src =
    view === "sam3" && crop.sam3_crop
      ? `${cropBase}/sam3/${crop.sam3_crop}`
      : `${cropBase}/${crop.key}`;

  if (minimised) {
    return (
      <button
        type="button"
        className="spot spot--min"
        onClick={() => setMinimised(false)}
        title="Show the close-up again"
      >
        <img src={`${cropBase}/${crop.key}`} alt="" />
        <span className="mono">#{trackId}</span>
      </button>
    );
  }

  return (
    <aside className="spot" aria-label={`Close-up of person ${trackId}`}>
      <header className="spot__bar">
        <b className="mono">#{trackId}</b>
        <span className="spot__standing">{standing}</span>
        <button
          type="button"
          onClick={() => setMinimised(true)}
          title="Minimise"
          aria-label="Minimise"
        >
          &minus;
        </button>
        <button type="button" onClick={onClose} title="Close" aria-label="Close">
          &times;
        </button>
      </header>

      <img className="spot__shot" src={src} alt={label} />

      <footer className="spot__foot">
        <p>{label}</p>
        <div className="spot__row">
          <button
            type="button"
            className="mono"
            onClick={() => onSeek(crop.key_pts_ms)}
            title="Jump the video to this instant"
          >
            {timecode(crop.key_pts_ms)}
          </button>
          {/* Only offered when the referee was actually called on this frame.
              A disabled tab would imply the picture exists. */}
          {crop.sam3_crop && (
            <button
              type="button"
              className={`mono ${view === "sam3" ? "is-on" : ""}`}
              onClick={() => setView(view === "sam3" ? "key" : "sam3")}
            >
              {view === "sam3" ? "detector view" : "what SAM 3 saw"}
            </button>
          )}
        </div>
        <span className="spot__meta mono">
          {classLabel(crop.key_class)} &middot; {verdictLabel(crop.key_verdict)}
        </span>
      </footer>
    </aside>
  );
}
