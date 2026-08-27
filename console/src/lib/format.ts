import type { State, Lane, Severity } from "../types";

/** Timecode for a PTS millisecond value. Always mm:ss.s -- a reviewer reads
 *  these against the video scrubber, so the tenth matters. */
export function timecode(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "--:--";
  const total = ms / 1000;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

export function seconds(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "--";
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * A count and its denominator, together, always.
 *
 * This is the contract rule made structural: there is no single-number mode,
 * so `33 looking up` cannot be rendered without `of 11,151`. A code-review
 * rule gets forgotten; a missing function argument does not compile.
 *
 * Returns `not measurable` rather than `0%` when the denominator is zero --
 * dividing by nothing is an absence of evidence, not an absence of the thing.
 */
export function withDenominator(
  value: number | null | undefined,
  total: number | null | undefined,
): { text: string; measurable: boolean } {
  if (value === null || value === undefined || !total) {
    return { text: "not measurable", measurable: false };
  }
  const pct = Math.round((value / total) * 100);
  return {
    text: `${value.toLocaleString()} of ${total.toLocaleString()} (${pct}%)`,
    measurable: true,
  };
}

/** Human label for a machine or human state. Never abbreviated -- a
 *  screenshot in a report has to read correctly on its own. */
export const STATE_LABEL: Record<State, string> = {
  review_candidate: "review candidate",
  needs_better_view: "needs better view",
  context_observation: "context observation",
  no_action: "no action",
  human_confirmed: "confirmed by reviewer",
  human_dismissed: "dismissed by reviewer",
};

export const STATE_VAR: Record<State, string> = {
  review_candidate: "review",
  needs_better_view: "abstain",
  context_observation: "context",
  no_action: "none",
  human_confirmed: "confirm",
  human_dismissed: "dismiss",
};

export const LANE_LABEL: Record<Lane, string> = {
  object: "Object",
  orientation: "Orientation",
  hands: "Hands",
  presence: "Presence",
  quality: "Quality",
};

/** Severity drives mark height, never colour -- colour is the lane. */
export const SEVERITY_WEIGHT: Record<Severity, number> = {
  info: 0.35,
  low: 0.55,
  medium: 0.78,
  high: 1,
};

/** How a track is named on screen.
 *
 * Seat attribution is unapproved on every calibration profile but cam12, so
 * most tracks resolve only as far as a track id. The word "unattributed" is
 * shown, not implied: a reviewer must never infer that #118 means seat 118.
 */
export function subjectName(
  trackId: number,
  seat: string | null,
  seatState: string | null,
): { name: string; qualifier: string | null } {
  if (seat && seatState && seatState !== "unattributed") {
    return { name: `Seat ${seat}`, qualifier: null };
  }
  return { name: `Person #${trackId}`, qualifier: "unattributed" };
}
