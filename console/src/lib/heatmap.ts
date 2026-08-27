import type { OverlayBundle } from "../overlay";
import type { Episode } from "./deviation";

/**
 * Where the evidence actually landed, accumulated over the whole recording.
 *
 * A reviewer scrubbing a 143-second recording sees one instant at a time and
 * has no way to answer "which part of this hall did the system keep reacting
 * to". The findings table answers it per person; nothing answered it per
 * *place*. That is what this is for, and it is the standard way computer
 * vision presents accumulated spatial evidence.
 *
 * What is accumulated -- and what deliberately is not
 * ---------------------------------------------------
 *
 * Three sources, each switchable, because mixing them would produce a picture
 * that cannot be read back:
 *
 * - `objects`  every object proposal near a hand, at its own box centre,
 *              weighted by detector confidence.
 * - `people`   every tracked person box centre. This is an occupancy map --
 *              where people *were* -- and it is the control the other two are
 *              read against. A bright spot in `objects` that is equally bright
 *              in `people` is a busy seat, not a suspicious one.
 * - `turns`    the peak instant of each measured turn, at that person's box
 *              centre.
 *
 * The important discipline: **this is a density of proposals, not a density of
 * cheating.** `docs/OUTPUT_CONTRACT.md` forbids presenting a count of
 * proposals as a rate of anything, and a heatmap is the easiest place in a UI
 * to break that rule by accident -- red reads as "guilty here" to everyone who
 * has ever seen one. So the legend says "proposals", the scale is labelled
 * with the raw count at the maximum, and the occupancy layer exists precisely
 * so a bright region can be explained away as "that is where people sit".
 *
 * Normalisation is by the grid maximum, which means the brightest cell is
 * always fully saturated even on a recording with almost nothing in it. The
 * peak count is returned so the caller can say what the top of the scale is
 * worth; a heatmap without that number is a picture that looks identical for
 * 3 detections and 3,000.
 */

export type HeatSource = "objects" | "people" | "turns";

export interface HeatField {
  /** Column count. Rows are derived from the frame's aspect. */
  cols: number;
  rows: number;
  /** Normalised 0..1, row-major. */
  cells: Float32Array;
  /** Raw weight in the hottest cell, so the scale can be labelled. */
  peak: number;
  /** Total accumulated events, for the caption. */
  total: number;
}

/** Cell size in source pixels. Coarse on purpose: a per-pixel splat of a few
 *  hundred detections is a scatter of dots, not a density. */
const CELL_PX = 16;
/** Gaussian radius in cells. */
const BLUR = 2;

function blur(src: Float32Array, cols: number, rows: number): Float32Array {
  // Separable box blur, run three times -- the standard cheap approximation of
  // a Gaussian, and far less code than a real kernel for a result nobody can
  // tell apart at this cell size.
  let cur = src;
  for (let pass = 0; pass < 3; pass += 1) {
    const horiz = new Float32Array(cols * rows);
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        let sum = 0;
        let n = 0;
        for (let dx = -BLUR; dx <= BLUR; dx += 1) {
          const sx = x + dx;
          if (sx < 0 || sx >= cols) continue;
          sum += cur[y * cols + sx];
          n += 1;
        }
        horiz[y * cols + x] = sum / n;
      }
    }
    const vert = new Float32Array(cols * rows);
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        let sum = 0;
        let n = 0;
        for (let dy = -BLUR; dy <= BLUR; dy += 1) {
          const sy = y + dy;
          if (sy < 0 || sy >= rows) continue;
          sum += horiz[sy * cols + x];
          n += 1;
        }
        vert[y * cols + x] = sum / n;
      }
    }
    cur = vert;
  }
  return cur;
}

export function buildHeat(
  overlay: OverlayBundle | null,
  source: HeatSource,
  turns: Episode[] = [],
): HeatField | null {
  if (!overlay || !overlay.frames.length) return null;

  const cols = Math.max(1, Math.ceil(overlay.width / CELL_PX));
  const rows = Math.max(1, Math.ceil(overlay.height / CELL_PX));
  const grid = new Float32Array(cols * rows);
  let total = 0;

  const add = (x: number, y: number, weight: number) => {
    const cx = Math.min(cols - 1, Math.max(0, Math.floor(x / CELL_PX)));
    const cy = Math.min(rows - 1, Math.max(0, Math.floor(y / CELL_PX)));
    grid[cy * cols + cx] += weight;
    total += 1;
  };

  if (source === "turns") {
    for (const e of turns) {
      const frame = overlay.frames.find((f) => f.t === e.peak_ms);
      const person = frame?.p.find((p) => p.i === e.track_id);
      if (!person) continue;
      const [x1, y1, x2, y2] = person.b;
      add((x1 + x2) / 2, (y1 + y2) / 2, 1);
    }
  } else {
    for (const frame of overlay.frames) {
      for (const person of frame.p) {
        if (source === "people") {
          const [x1, y1, x2, y2] = person.b;
          add((x1 + x2) / 2, (y1 + y2) / 2, 1);
          continue;
        }
        for (const o of person.o ?? []) {
          const [x1, y1, x2, y2] = o.b;
          // Confidence as weight, so a wall of 0.3 guesses does not outshine a
          // handful of 0.9 detections purely by being numerous.
          add((x1 + x2) / 2, (y1 + y2) / 2, o.p);
        }
      }
    }
  }

  const smoothed = blur(grid, cols, rows);
  let peak = 0;
  for (const v of smoothed) if (v > peak) peak = v;
  if (peak <= 0) return { cols, rows, cells: smoothed, peak: 0, total };
  const cells = new Float32Array(smoothed.length);
  for (let i = 0; i < smoothed.length; i += 1) cells[i] = smoothed[i] / peak;
  return { cols, rows, cells, peak, total };
}

/**
 * Turquoise -> amber -> red, with alpha rising from nothing.
 *
 * Not the jet colormap. Jet's rainbow puts a hard cyan/green boundary in the
 * middle of the range that reads as a threshold nobody set, and it is
 * unreadable to the ~8% of men with red-green colour deficiency. This ramp is
 * monotonic in both lightness and saturation, so the ordering survives being
 * printed in grey or seen by anyone.
 */
export function heatColor(t: number): [number, number, number, number] {
  const c = Math.min(1, Math.max(0, t));
  // Below this the cell is noise from the blur skirt and is left transparent,
  // otherwise the whole frame acquires a faint wash that hides the picture.
  if (c < 0.08) return [0, 0, 0, 0];
  const stops: [number, [number, number, number]][] = [
    [0.08, [56, 132, 150]],
    [0.4, [214, 168, 74]],
    [0.7, [222, 116, 74]],
    [1, [206, 60, 60]],
  ];
  for (let i = 1; i < stops.length; i += 1) {
    const [t1, c1] = stops[i];
    const [t0, c0] = stops[i - 1];
    if (c <= t1) {
      const k = (c - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * k),
        Math.round(c0[1] + (c1[1] - c0[1]) * k),
        Math.round(c0[2] + (c1[2] - c0[2]) * k),
        Math.round(90 + 130 * c),
      ];
    }
  }
  const last = stops[stops.length - 1][1];
  return [last[0], last[1], last[2], 220];
}

export const HEAT_LABEL: Record<HeatSource, string> = {
  objects: "objects at hands",
  people: "where people were",
  turns: "turns away",
};
