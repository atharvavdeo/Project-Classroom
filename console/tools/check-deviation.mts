/** Run the sideways lane over a real overlay and print what it would show.
 *
 *   npx tsx tools/check-deviation.mts 1509_04_talking
 *
 * This exists because the previous version of this lane reported nothing on
 * the one recording it was built for, and nothing in the UI made that visible
 * -- an empty table reads as "clean hall". Numbers first, then the screen.
 */
import { readFileSync } from "node:fs";
import { deviationEpisodes } from "../src/lib/deviation.js";
import type { OverlayBundle } from "../src/overlay.js";

const id = process.argv[2] ?? "1509_04_talking";
const raw = JSON.parse(
  readFileSync(new URL(`../public/data/${id}.overlay.json`, import.meta.url), "utf8"),
) as OverlayBundle;

const { episodes, baselines, noBaseline } = deviationEpisodes(raw);

console.log(`${id}: ${raw.frames.length} frames, ${raw.width}x${raw.height}`);
console.log(`baselines ${baselines.size}   no baseline ${noBaseline.length}`);
console.log(`episodes  ${episodes.length}`);

const byTrack = new Map<number, number>();
for (const e of episodes) byTrack.set(e.track_id, (byTrack.get(e.track_id) ?? 0) + 1);
console.log(
  "per person:",
  [...byTrack.entries()].sort((a, b) => b[1] - a[1]).map(([t, n]) => `#${t}:${n}`).join(" "),
);

const secs = (ms: number) => (ms / 1000).toFixed(1).padStart(6);
for (const e of episodes.slice(0, 25)) {
  console.log(
    `  #${String(e.track_id).padEnd(3)} ${secs(e.from_ms)}s-${secs(e.to_ms)}s ` +
      `${e.away.padEnd(5)} peak ${e.peak_z.toFixed(1)}z @${secs(e.peak_ms)}s` +
      (e.with_track != null ? `  with #${e.with_track}` : ""),
  );
}
if (episodes.length > 25) console.log(`  ... ${episodes.length - 25} more`);
console.log(`paired: ${episodes.filter((e) => e.with_track != null).length}`);
