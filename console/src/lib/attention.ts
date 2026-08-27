import type { OverlayBundle } from "../overlay";

/**
 * Where people were looking, derived from the per-frame facing labels.
 *
 * This is a **separate condition from the object lane, with its own criteria**,
 * and it is deliberately not scored the same way. An object at a hand is a
 * thing; a head turned to the right is a posture, and posture is normal. The
 * only thing that makes it worth a reviewer's second is duration or repetition,
 * so those are the two thresholds, and nothing else feeds in.
 *
 * Why this is computed in the console rather than read from the run:
 *
 * The pipeline emits `orientation_events` only for people who have a yaw
 * baseline, and a baseline needs the head to be large enough in frame to
 * resolve yaw at all. On the 640x480 recording that fails for 83 of 92 people,
 * so the run carries **zero** orientation events while the same samples carry
 * 1,777 `turned_right` and 163 `turned_left` labels. The coarse facing label
 * survives where the fine measurement does not, and throwing it away because
 * the precise version failed loses the only signal there was.
 *
 * What this is not:
 *
 * - It is **not gaze**. Nothing here knows where the eyes went. Four
 *   directions is the entire resolution of the underlying label.
 * - It is **not a finding**. `docs/OUTPUT_CONTRACT.md` is explicit that
 *   behaviour never routes a person to review on its own, and this does not
 *   change that. These spans are context a human reads next to something else,
 *   and the UI says so on every row.
 * - `not_resolvable_at_source_scale` is not "facing forward". It produces no
 *   span at all, because an absence of measurement must not read as an
 *   observation of normality.
 */

export interface AttentionConfig {
  /** A single turn has to last this long to be worth showing. Below it, this
   *  is a person glancing at their own screen. */
  sustainedMs: number;
  /** Separate turns inside this window count towards "kept looking around". */
  repeatWindowMs: number;
  /** How many turns in that window before it is worth showing. */
  repeatCount: number;
  /** A gap this long or shorter does not end a span -- one unresolvable
   *  sample in the middle of a turn is a measurement dropout, not a
   *  head snapping back and forth. */
  bridgeMs: number;
  /** A direction this dominant across a person's own samples is their resting
   *  posture relative to the camera, not a departure from it. */
  baselineShare: number;
  /** A single span covering this much of a person's observed time is a pose,
   *  however long it lasted in seconds. */
  maxSpanShare: number;
}

export const ATTENTION: AttentionConfig = {
  sustainedMs: 3000,
  repeatWindowMs: 20000,
  repeatCount: 4,
  bridgeMs: 750,
  baselineShare: 0.6,
  maxSpanShare: 0.5,
};

export type Direction = "left" | "right" | "down";

export interface AttentionSpan {
  track_id: number;
  from_ms: number;
  to_ms: number;
  direction: Direction;
  samples: number;
}

export interface AttentionFinding {
  track_id: number;
  kind: "sustained" | "repeated";
  from_ms: number;
  to_ms: number;
  direction: Direction | "mixed";
  /** Spans behind this row, so the count shown is never a bare number. */
  spans: number;
  /** What was measured, in words, for the reviewer to check the claim. */
  evidence: string;
}

const FACING: Record<string, Direction | null> = {
  turned_left: "left",
  turned_right: "right",
  downward: "down",
  frontal: null,
  not_resolvable_at_source_scale: null,
};

/** Group consecutive same-direction samples per person into spans. */
export function facingSpans(
  overlay: OverlayBundle,
  cfg: AttentionConfig = ATTENTION,
): AttentionSpan[] {
  // Per track, the run of samples currently open.
  const open = new Map<
    number,
    { dir: Direction; from: number; last: number; n: number }
  >();
  const out: AttentionSpan[] = [];

  const close = (trackId: number) => {
    const cur = open.get(trackId);
    if (!cur) return;
    out.push({
      track_id: trackId,
      from_ms: cur.from,
      to_ms: cur.last,
      direction: cur.dir,
      samples: cur.n,
    });
    open.delete(trackId);
  };

  for (const frame of overlay.frames) {
    const seen = new Set<number>();
    for (const person of frame.p) {
      seen.add(person.i);
      const dir = FACING[person.f] ?? null;
      const cur = open.get(person.i);

      if (!dir) {
        // Facing forward, or unmeasurable. Either way the turn is over.
        close(person.i);
        continue;
      }
      if (cur && cur.dir === dir && frame.t - cur.last <= cfg.bridgeMs) {
        cur.last = frame.t;
        cur.n += 1;
      } else {
        close(person.i);
        open.set(person.i, { dir, from: frame.t, last: frame.t, n: 1 });
      }
    }
    // Someone who left the frame ends their span here rather than having it
    // stretched across an absence.
    for (const id of [...open.keys()]) {
      if (!seen.has(id)) close(id);
    }
  }
  for (const id of [...open.keys()]) close(id);

  return out;
}

/**
 * Turn spans into the two things worth surfacing: one long look away, or a
 * lot of short ones. Both are reported with the arithmetic that produced them.
 */
export function attentionFindings(
  overlay: OverlayBundle | null,
  cfg: AttentionConfig = ATTENTION,
): AttentionFinding[] {
  if (!overlay) return [];
  const spans = facingSpans(overlay, cfg);
  const byTrack = new Map<number, AttentionSpan[]>();
  for (const s of spans) {
    if (!byTrack.has(s.track_id)) byTrack.set(s.track_id, []);
    byTrack.get(s.track_id)!.push(s);
  }

  const out: AttentionFinding[] = [];

  for (const [trackId, list] of byTrack) {
    list.sort((a, b) => a.from_ms - b.from_ms);

    // Each person's own baseline, before anything is called a departure from
    // it.
    //
    // Without this the biggest "finding" on the talking recording was a person
    // "looking to their right and holding it" for 143.0s of a 143.2s
    // recording. That is not a person looking away; that is where their chair
    // faces. A camera mounted off to one side makes half a hall permanently
    // `turned_right`, and reporting posture as an event would bury the actual
    // glances underneath it.
    //
    // This is the same idea as the pipeline's yaw baseline, applied to the
    // coarse label: a direction is only interesting relative to where this
    // person usually points.
    const observedFrom = Math.min(...list.map((s) => s.from_ms));
    const observedTo = Math.max(...list.map((s) => s.to_ms));
    const observed = Math.max(observedTo - observedFrom, 1);

    const held = new Map<Direction, number>();
    for (const s of list) {
      held.set(s.direction, (held.get(s.direction) ?? 0) + (s.to_ms - s.from_ms));
    }
    const totalHeld = [...held.values()].reduce((a, b) => a + b, 0) || 1;
    const resting = new Set<Direction>(
      [...held.entries()]
        .filter(([, ms]) => ms / totalHeld >= cfg.baselineShare)
        .map(([dir]) => dir),
    );

    // Downward is excluded from both tests. Head down over a desk is what
    // writing looks like, and flagging it would bury the two directions that
    // actually point at a neighbour.
    const sideways = list.filter(
      (s) =>
        s.direction !== "down" &&
        !resting.has(s.direction) &&
        (s.to_ms - s.from_ms) / observed <= cfg.maxSpanShare,
    );

    for (const s of sideways) {
      const held = s.to_ms - s.from_ms;
      if (held >= cfg.sustainedMs) {
        out.push({
          track_id: trackId,
          kind: "sustained",
          from_ms: s.from_ms,
          to_ms: s.to_ms,
          direction: s.direction,
          spans: 1,
          evidence: `held ${(held / 1000).toFixed(1)}s over ${s.samples} samples (threshold ${cfg.sustainedMs / 1000}s)`,
        });
      }
    }

    // Repeats: a sliding window over the start times.
    for (let i = 0; i < sideways.length; i += 1) {
      const window = sideways.filter(
        (s) =>
          s.from_ms >= sideways[i].from_ms &&
          s.from_ms - sideways[i].from_ms <= cfg.repeatWindowMs,
      );
      if (window.length >= cfg.repeatCount) {
        const dirs = new Set(window.map((w) => w.direction));
        out.push({
          track_id: trackId,
          kind: "repeated",
          from_ms: window[0].from_ms,
          to_ms: window[window.length - 1].to_ms,
          direction: dirs.size === 1 ? [...dirs][0] : "mixed",
          spans: window.length,
          evidence: `${window.length} turns in ${((window[window.length - 1].from_ms - window[0].from_ms) / 1000).toFixed(0)}s (threshold ${cfg.repeatCount} in ${cfg.repeatWindowMs / 1000}s)`,
        });
        // One row per burst, not one per member of it.
        i += window.length - 1;
      }
    }
  }

  return out.sort((a, b) => a.from_ms - b.from_ms);
}

export const DIRECTION_LABEL: Record<Direction | "mixed", string> = {
  left: "to their left",
  right: "to their right",
  down: "downward",
  mixed: "both ways",
};
