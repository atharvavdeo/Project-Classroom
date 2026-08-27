/** Per-frame overlay geometry, as emitted by tools/build_overlay_bundle.py.
 *
 * Keys are single letters because this file is ~4 MB of numbers before gzip;
 * the terse names are a size decision, not a style one, and they are decoded
 * once here so nothing downstream has to know about them.
 */

export interface OverlayObject {
  /** class */ c: string;
  /** box  */ b: [number, number, number, number];
  /** confidence */ p: number;
  /** wrist distance, in person widths */ d: number | null;
  /** nearest wrist */ w: string;
  /** source */ s: string;
}

export interface OverlayPerson {
  /** track id */ i: number;
  /** box */ b: [number, number, number, number];
  /** flat [x,y,...] joints; -1 means not visible */ j: number[];
  /** facing */ f: string;
  /** seat state */ st: string;
  /** objects near this person's hands */ o?: OverlayObject[];
}

export interface OverlayFrame {
  /** pts_ms */ t: number;
  p: OverlayPerson[];
}

export interface OverlayBundle {
  schema: number;
  joints: string[];
  bones: [number, number][];
  frames: OverlayFrame[];
  /** Source dimensions. Not in the file -- supplied by the caller, because the
   *  geometry is in video pixel space and the console has to know the frame
   *  size to size its canvas. */
  width: number;
  height: number;
}

/**
 * The sample whose timestamp is closest to `ms`, by binary search.
 *
 * Nearest, not last-before: the pipeline samples at 4 Hz against 25 fps video,
 * so at any given playback time the truthful answer is the measurement taken
 * closest to it. Holding the previous sample until the next one arrives would
 * lag the marks behind the picture by up to a quarter second, which on a
 * moving hand is enough to look wrong.
 */
export function nearestFrame(
  bundle: OverlayBundle,
  ms: number,
): OverlayFrame | null {
  const frames = bundle.frames;
  if (!frames.length) return null;

  let lo = 0;
  let hi = frames.length - 1;
  if (ms <= frames[lo].t) return frames[lo];
  if (ms >= frames[hi].t) return frames[hi];

  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t <= ms) lo = mid;
    else hi = mid;
  }
  return ms - frames[lo].t <= frames[hi].t - ms ? frames[lo] : frames[hi];
}
