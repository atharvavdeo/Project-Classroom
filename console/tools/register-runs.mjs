/**
 * Register the finished runs in `public/` with a console.
 *
 * The hierarchy is user-authored and deliberately not seeded by the app -- a
 * centre in the sidebar is a claim that the centre exists. But re-typing five
 * recordings after a fresh database is busywork, and busywork is how a
 * developer ends up leaving a console empty and wondering why. This script is
 * the reproducible version of that typing, and it is idempotent: it will not
 * duplicate a recording that is already registered.
 *
 * It targets whichever console is running, so it works against the local Node
 * server and against the deployed Worker without change.
 *
 * Usage:
 *   node tools/register-runs.mjs
 *   node tools/register-runs.mjs --base https://drishti-console.workers.dev
 *   node tools/register-runs.mjs --project "CET Exam 2026" --centre "Dahisar Centre"
 */
const args = process.argv.slice(2);
const value = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};

const BASE = value("--base", "http://localhost:5179").replace(/\/$/, "");
const PROJECT = value("--project", "CET Exam 2026");
const CENTRE = value("--centre", "Dahisar Centre");

/** The runs whose artifacts are in `public/`. Media key, then run key: the
 *  two are different because one recording can be processed more than once,
 *  and the run that produced the crops is the one the console must read. */
const RUNS = [
  ["Hall 3 · Seat 12 · paper handling", "1512_12_paper", "1512_12_paper"],
  ["Hall 1 · Camera 2 · phone", "c01_phone_hall", "1511_01_phone"],
  ["Hall 4 · Camera 1 · mobile usage", "c03_mobile_usage", "1505_03_mobile"],
  ["Hall 5 · Camera 3 · talking", "c04_talking", "1509_04_talking"],
  ["Hall 6 · Camera 1 · phone (last 10 min)", "c06_phone_last10", "1507_06_phone"],
];

async function call(path, body, method = "POST") {
  const res = await fetch(BASE + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const doc = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(doc?.error ?? `${res.status} ${res.statusText}`);
  return doc;
}

const health = await call("/api/health", null, "GET");
console.log(`${BASE} — store: ${health.store}`);

let { projects } = await call("/api/tree", null, "GET");

let project = projects.find((p) => p.name === PROJECT);
if (!project) {
  ({ projects } = await call("/api/projects", { name: PROJECT }));
  project = projects.find((p) => p.name === PROJECT);
  console.log(`  created project ${PROJECT}`);
}

let centre = project.centres.find((c) => c.name === CENTRE);
if (!centre) {
  ({ projects } = await call(`/api/projects/${project.id}/centres`, {
    name: CENTRE,
  }));
  centre = projects
    .find((p) => p.id === project.id)
    .centres.find((c) => c.name === CENTRE);
  console.log(`  created centre ${CENTRE}`);
}

const existing = new Set(centre.videos.map((v) => v.name));
let added = 0;
for (const [name, media, run] of RUNS) {
  if (existing.has(name)) {
    console.log(`  already registered: ${name}`);
    continue;
  }
  await call(`/api/centres/${centre.id}/videos`, {
    name,
    media_url: `/media/${media}.mp4`,
    bundle_url: `/data/${run}.json`,
    overlay_url: `/data/${run}.overlay.json`,
    crops_url: `/crops/${run}`,
  });
  console.log(`  registered: ${name}`);
  added += 1;
}

const after = await call("/api/tree", null, "GET");
const count = after.projects
  .flatMap((p) => p.centres)
  .flatMap((c) => c.videos).length;
console.log(`${added} added; ${count} recording(s) registered in total`);
