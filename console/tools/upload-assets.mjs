/**
 * Push the run artifacts in `public/` to an R2 bucket.
 *
 * 2,076 files and ~105 MB: the recordings, the run bundles and the review
 * crops. They are not in git (see .gitignore) and they are not in the Worker
 * build (see vite.config.ts), so this is the step that puts them where the
 * deployed console can read them.
 *
 * Keys mirror the local paths exactly -- `public/media/x.mp4` becomes
 * `media/x.mp4` -- because the front end asks for `/media/x.mp4` in both
 * places. Any renaming here would be a difference between environments that
 * only shows up in production.
 *
 * Usage:
 *   node tools/upload-assets.mjs                # to the real bucket
 *   node tools/upload-assets.mjs --local        # to the wrangler emulator
 *   node tools/upload-assets.mjs --only 1512    # just one run's files
 *   node tools/upload-assets.mjs --dry-run
 */
import { execFile } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import { join, posix, relative, sep } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);

// The locally installed binary, not `npx`. npx re-resolves the package on
// every invocation, and at 2,076 files that dominated the wall clock.
const WRANGLER = new URL("../node_modules/.bin/wrangler.cmd", import.meta.url)
  .pathname.replace(/^\/([A-Za-z]:)/, "$1");

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const value = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : null;
};

const LOCAL = flag("--local");
const BUCKET = value("--bucket") ?? "drishti-runs";
const DRY = flag("--dry-run");
const ONLY = value("--only");
// Wrangler spawns a process per object. More than a handful in flight on
// Windows costs more in process startup than it wins in parallelism.
//
// Against the local emulator it must be exactly one: every wrangler process
// opens the same miniflare state directory, and concurrent writers collide.
// Measured at 6 workers: 18 of 119 objects failed, and the failures were
// scattered, so a partial upload would have looked like a handful of
// mysteriously missing crops rather than an obvious error.
const CONCURRENCY = LOCAL ? 1 : Number(value("--concurrency") ?? 14);

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const PUBLIC = join(ROOT, "public");

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else out.push(full);
  }
  return out;
}

const files = ["media", "data", "crops"]
  .map((d) => join(PUBLIC, d))
  .filter((d) => {
    try {
      return statSync(d).isDirectory();
    } catch {
      return false;
    }
  })
  .flatMap(walk)
  .map((full) => ({
    full,
    key: relative(PUBLIC, full).split(sep).join(posix.sep),
    bytes: statSync(full).size,
  }))
  .filter((f) => !ONLY || f.key.includes(ONLY));

const total = files.reduce((n, f) => n + f.bytes, 0);
console.log(
  `${files.length} file(s), ${(total / 1048576).toFixed(1)} MB ` +
    `-> r2://${BUCKET}${LOCAL ? " (local emulator)" : ""}`,
);
if (DRY) {
  for (const f of files.slice(0, 10)) console.log(`  ${f.key}`);
  if (files.length > 10) console.log(`  ... and ${files.length - 10} more`);
  process.exit(0);
}

let done = 0;
let failed = 0;
const queue = [...files];

async function worker() {
  for (;;) {
    const file = queue.pop();
    if (!file) return;
    const argv = [
      "r2",
      "object",
      "put",
      `${BUCKET}/${file.key}`,
      "--file",
      file.full,
      ...(LOCAL ? ["--local"] : ["--remote"]),
    ];
    // One retry with a short backoff. Most failures here are contention or a
    // transient network error, and both clear on a second attempt; a put is
    // idempotent so retrying costs nothing but time.
    let error = null;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        await run(WRANGLER, argv, { shell: true, maxBuffer: 1 << 22 });
        error = null;
        break;
      } catch (e) {
        error = e;
        await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
      }
    }
    if (error) {
      failed += 1;
      // Name the file. A count of failures with no names is not actionable,
      // and a partial upload that reports success is how a console ends up
      // showing a broken image for one track and nobody knows which.
      console.error(`  FAILED ${file.key}: ${String(error.message).slice(0, 160)}`);
    }
    done += 1;
    if (done % 50 === 0 || done === files.length) {
      console.log(`  ${done}/${files.length}`);
    }
  }
}

await Promise.all(Array.from({ length: CONCURRENCY }, worker));

console.log(
  failed
    ? `done with ${failed} failure(s) — re-run to retry, puts are idempotent`
    : "done, all objects uploaded",
);
process.exit(failed ? 1 : 0);
