import type { OverlayBundle, OverlayPerson } from "../overlay";

/**
 * Turning away from your own screen, measured against your own posture.
 *
 * Why this replaces the facing-label lane for the sideways condition
 * --------------------------------------------------------------------
 *
 * The coarse four-way label is not discriminative on this footage, and the
 * numbers say so plainly. On `1509_04_talking` the labels are:
 *
 *     turned_right                    1777
 *     not_resolvable_at_source_scale  1470
 *     downward                         239
 *     turned_left                      163
 *     frontal                           92
 *
 * `turned_right` outnumbers `turned_left` eleven to one. That is not a hall
 * where everybody looks right; that is a camera mounted off to one side, so an
 * oblique view of a head facing its own monitor already reads as "turned
 * right". Person #2 is labelled `turned_right` in 573 of 573 samples -- every
 * frame they are on screen.
 *
 * The old lane then did the only sane thing available to it and suppressed a
 * direction that dominates a person's own samples, because otherwise the top
 * "finding" was somebody turning right for 143.0s of a 143.2s recording. But
 * that suppression is what silenced the real turns: #2 is 100% right, #6 is
 * 88%, #62 is 91%, #3 is 69% -- all above the 60% baseline, so every one of
 * their genuine turns was thrown away with their posture. The lane could
 * either report everybody always or report nobody ever, and there was no
 * threshold in between, because a bucket label cannot express "further right
 * than *this person* usually sits".
 *
 * What is measured instead
 * ------------------------
 *
 * A continuous quantity, per person, per frame: how far the nose sits from the
 * midpoint of the shoulders, in units of that person's own shoulder width.
 *
 *     yaw = (nose.x - (left_shoulder.x + right_shoulder.x) / 2) / shoulder_width
 *
 * Shoulder width normalises out distance from the camera, so the number is
 * comparable across a frame in which one person is twice the size of another.
 * The keypoints are already there -- AlphaPose emits nose and both shoulders,
 * and coverage on this recording is 573/573 frames for #2 and 564/573 for #4.
 * Nothing new has to run.
 *
 * A person is then compared **only against themselves**: median for centre,
 * MAD for spread, and an episode is a run of frames sitting `zEnter` robust
 * deviations off their own centre. Person #2's constant `turned_right` becomes
 * their zero and produces nothing; the same treatment surfaces 120 deviating
 * frames on #4 and 19 on #3, which the label lane reported as none.
 *
 * What this is still not
 * ----------------------
 *
 * - **Not gaze.** No eye direction is measured anywhere in this system. This
 *   is where the head sits relative to the shoulders, which is a different and
 *   much cruder thing.
 * - **Not proof of talking.** Two people turned towards each other is what
 *   talking looks like from a ceiling camera; it is also what looking at the
 *   clock looks like. `pairs()` marks the mutual case because it is worth a
 *   reviewer's second, not because it settles anything.
 * - **Not a finding on its own.** `docs/OUTPUT_CONTRACT.md` holds: behaviour
 *   does not route a person to review by itself, and no state here is written
 *   into the decision record. These rows exist to be looked at.
 * - **An abstention is not a zero.** A person whose shoulders or nose are too
 *   small or too often missing has no baseline, and is reported as
 *   `no_baseline` rather than as somebody who never turned.
 */

export interface DeviationConfig {
  /** Below this many pixels the shoulder span is too short for the ratio to
   *  mean anything -- a two-pixel error in a 10px span is a 20% swing. */
  minShoulderPx: number;
  /** A raw ratio past this is a broken pose, not a turned head. A nose one and
   *  a half shoulder-widths off centre is anatomically impossible; it is a
   *  mis-assigned keypoint, and left in it drags the median with it. */
  maxRawRatio: number;
  /** Fewer usable samples than this and the person gets no baseline at all.
   *  A median over a handful of frames is not a posture. */
  minSamples: number;
  /** Robust deviations from a person's own centre before a frame counts as
   *  turned away. */
  zEnter: number;
  /** Deviations below which the episode ends. Lower than `zEnter` on purpose:
   *  a single frame dipping under the entry bar mid-turn should not chop one
   *  turn into three. */
  zExit: number;
  /** A gap no longer than this does not end an episode -- one dropped pose is
   *  a measurement failure, not a head snapping back. */
  bridgeMs: number;
  /** How long an excursion has to hold. */
  minHoldMs: number;
  /** Two people turned towards each other within this much time of each other
   *  are reported as a pair. */
  pairToleranceMs: number;
  /** A person's own spread may not exceed this multiple of the population's
   *  median spread.
   *
   *  Without a ceiling, noise buys immunity. Person #1 on the talking
   *  recording turns right round at 4.5s -- their nose swings from 0.00 to
   *  -0.69 while their shoulder span collapses from 46px to 24px, which is
   *  what a torso rotating away from the camera looks like -- and the lane
   *  reported nothing. Their robust spread over the whole recording came out
   *  at 0.523 against a population median near 0.10, because their later
   *  samples are a mess. Dividing a real 0.7 swing by their own noise gave
   *  1.3z, under the 3z bar. The noisiest tracks were exactly the ones that
   *  could never be flagged. */
  maxScaleOverMedian: number;
  /** An excursion must also clear this much in absolute terms, in shoulder
   *  widths, regardless of how still the person normally is. This is the other
   *  half of the same problem: capping the spread makes quiet people easier to
   *  trip, so a floor keeps sub-pixel jitter from becoming an episode. */
  minAbsolute: number;
}

export const DEVIATION: DeviationConfig = {
  minShoulderPx: 12,
  maxRawRatio: 1.5,
  minSamples: 60,
  zEnter: 3,
  zExit: 2,
  bridgeMs: 750,
  minHoldMs: 1200,
  pairToleranceMs: 1500,
  maxScaleOverMedian: 2.0,
  minAbsolute: 0.25,
};

/** Which way the head went, relative to this person's own centre. */
export type Away = "left" | "right";

export interface Episode {
  track_id: number;
  from_ms: number;
  to_ms: number;
  away: Away;
  /** Frames in the episode. */
  samples: number;
  /** The largest robust deviation reached. */
  peak_z: number;
  /** Instant of that peak -- the frame worth showing a reviewer. */
  peak_ms: number;
  /** Another person turned the same way at the same time, if there was one. */
  with_track?: number;
  /** The arithmetic, in words, so the claim can be checked. */
  evidence: string;
}

export interface Baseline {
  track_id: number;
  /** Usable samples behind the baseline. */
  samples: number;
  centre: number;
  /** Robust spread (1.4826 x MAD), floored so a perfectly still person does
   *  not divide by zero and register every twitch as infinite deviation. */
  scale: number;
}

export interface DeviationResult {
  episodes: Episode[];
  baselines: Map<number, Baseline>;
  /** Tracks with too little usable pose to judge. Reported, never silent. */
  noBaseline: number[];
}

/** The yaw proxy for one person in one frame, or null if it cannot be had. */
export function yawProxy(
  person: OverlayPerson,
  joints: string[],
  cfg: DeviationConfig = DEVIATION,
): number | null {
  const idx = (name: string) => joints.indexOf(name);
  const at = (name: string): [number, number] | null => {
    const i = idx(name);
    if (i < 0) return null;
    const x = person.j[i * 2];
    const y = person.j[i * 2 + 1];
    // -1 is the bundle's "not visible". Both halves are written together, so
    // either being negative means the joint is absent.
    return x < 0 || y < 0 ? null : [x, y];
  };

  const nose = at("nose");
  const ls = at("left_shoulder");
  const rs = at("right_shoulder");
  if (!nose || !ls || !rs) return null;

  const width = Math.abs(ls[0] - rs[0]);
  if (width < cfg.minShoulderPx) return null;

  const value = (nose[0] - (ls[0] + rs[0]) / 2) / width;
  return Math.abs(value) > cfg.maxRawRatio ? null : value;
}

function median(sorted: number[]): number {
  const n = sorted.length;
  if (!n) return 0;
  const mid = n >> 1;
  return n % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Every excursion from every person's own resting head position.
 */
export function deviationEpisodes(
  overlay: OverlayBundle | null,
  cfg: DeviationConfig = DEVIATION,
): DeviationResult {
  const empty: DeviationResult = {
    episodes: [],
    baselines: new Map(),
    noBaseline: [],
  };
  if (!overlay) return empty;

  // Pass one: the series per person.
  const series = new Map<number, { t: number; v: number }[]>();
  for (const frame of overlay.frames) {
    for (const person of frame.p) {
      const v = yawProxy(person, overlay.joints, cfg);
      if (v == null) continue;
      let list = series.get(person.i);
      if (!list) {
        list = [];
        series.set(person.i, list);
      }
      list.push({ t: frame.t, v });
    }
  }

  // Pass two: each person's own centre and spread.
  const baselines = new Map<number, Baseline>();
  const noBaseline: number[] = [];
  for (const [trackId, list] of series) {
    if (list.length < cfg.minSamples) {
      noBaseline.push(trackId);
      continue;
    }
    const values = list.map((s) => s.v).sort((a, b) => a - b);
    const centre = median(values);
    const deviations = values.map((v) => Math.abs(v - centre)).sort((a, b) => a - b);
    // The floor matters: without it a person who barely moves gets scale 0 and
    // every sample becomes an infinite deviation, which is the inverse of the
    // bug this whole module exists to fix.
    const scale = Math.max(1.4826 * median(deviations), 0.02);
    baselines.set(trackId, { track_id: trackId, samples: list.length, centre, scale });
  }

  // The ceiling, applied once the whole population is known. A spread far
  // above everyone else's is a pose estimate falling apart, not a person who
  // genuinely swings their head twice as much as the room.
  const spreads = [...baselines.values()].map((b) => b.scale).sort((a, b) => a - b);
  const ceiling = spreads.length
    ? median(spreads) * cfg.maxScaleOverMedian
    : Infinity;
  for (const b of baselines.values()) b.scale = Math.min(b.scale, ceiling);

  // Pass three: runs of deviation, with hysteresis.
  const episodes: Episode[] = [];
  for (const [trackId, list] of series) {
    const base = baselines.get(trackId);
    if (!base) continue;

    let open: {
      from: number;
      last: number;
      n: number;
      peak: number;
      peakMs: number;
      sign: number;
    } | null = null;

    const shut = () => {
      if (!open) return;
      const held = open.last - open.from;
      if (held >= cfg.minHoldMs) {
        episodes.push({
          track_id: trackId,
          from_ms: open.from,
          to_ms: open.last,
          away: open.sign > 0 ? "right" : "left",
          samples: open.n,
          peak_z: open.peak,
          peak_ms: open.peakMs,
          evidence:
            `${(held / 1000).toFixed(1)}s at up to ${open.peak.toFixed(1)}x ` +
            `this person's own spread (${open.n} samples, baseline from ` +
            `${base.samples})`,
        });
      }
      open = null;
    };

    for (const { t, v } of list) {
      const offset = v - base.centre;
      const z = offset / base.scale;
      const mag = Math.abs(z);
      // Both bars, always: relative to this person, and large enough in
      // absolute terms that it is a turn rather than keypoint jitter.
      const big = Math.abs(offset) >= cfg.minAbsolute;

      if (open) {
        const stillOut = mag >= cfg.zExit && Math.sign(z) === open.sign;
        const contiguous = t - open.last <= cfg.bridgeMs;
        if (stillOut && contiguous) {
          open.last = t;
          open.n += 1;
          if (mag > open.peak) {
            open.peak = mag;
            open.peakMs = t;
          }
          continue;
        }
        shut();
      }
      if (mag >= cfg.zEnter && big) {
        open = { from: t, last: t, n: 1, peak: mag, peakMs: t, sign: Math.sign(z) || 1 };
      }
    }
    shut();
  }

  episodes.sort((a, b) => a.from_ms - b.from_ms || a.track_id - b.track_id);
  markPairs(episodes, cfg);

  return { episodes, baselines, noBaseline: noBaseline.sort((a, b) => a - b) };
}

/**
 * Two people turning towards each other at the same moment.
 *
 * This is the closest this system gets to the thing a proctor actually watches
 * for, and it is still only a coincidence of two postures. It is annotated on
 * the episode rather than promoted to its own state, so nothing downstream can
 * mistake "these two turned together" for "these two were talking".
 */
function markPairs(episodes: Episode[], cfg: DeviationConfig) {
  for (let i = 0; i < episodes.length; i += 1) {
    const a = episodes[i];
    if (a.with_track != null) continue;
    for (let k = i + 1; k < episodes.length; k += 1) {
      const b = episodes[k];
      // Sorted by start, so once a candidate begins after this episode has
      // already ended (plus the tolerance) nothing later can overlap it
      // either. Comparing against `a.to_ms` rather than `a.from_ms` is what
      // makes a long turn pair with a short one that starts partway through
      // it -- #3 at 96.5s and #4 at 98.5s are two seconds apart at the start
      // and overlap for nearly four.
      if (b.from_ms > a.to_ms + cfg.pairToleranceMs) break;
      if (b.track_id === a.track_id || b.with_track != null) continue;
      // Direction is deliberately not compared.
      //
      // The obvious rule -- two people facing each other must turn opposite
      // ways -- is only true for a camera in front of them. This one is up in
      // a corner looking down the row obliquely, so two neighbours turning
      // towards each other both move their nose the same way in image space.
      // Requiring opposite signs rejected the clearest case on the talking
      // recording: #3 and #4 turn together at 96.5s-102.3s, both to the right,
      // and the pair went unmarked. Co-occurrence in time is the whole claim.
      const overlap =
        Math.min(a.to_ms, b.to_ms) - Math.max(a.from_ms, b.from_ms);
      if (overlap <= 0) continue;
      a.with_track = b.track_id;
      b.with_track = a.track_id;
      break;
    }
  }
}

export const AWAY_LABEL: Record<Away, string> = {
  left: "away to their left",
  right: "away to their right",
};
