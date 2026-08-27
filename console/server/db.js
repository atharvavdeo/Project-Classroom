import Database from "better-sqlite3";
import { readFileSync } from "node:fs";
import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export function openDb(file = process.env.DRISHTI_DB ?? join(here, "drishti.db")) {
  // Resolve before anything else, and say out loud which file was opened.
  //
  // A relative or shell-mangled DRISHTI_DB silently opens a *different*,
  // empty database, and the console then shows an empty project list -- which
  // is indistinguishable from "nobody has created anything yet". That is
  // exactly how a set of real decisions appeared to vanish once. The path is
  // printed so a wrong one is visible immediately rather than inferred later.
  const resolved = resolve(file);
  mkdirSync(dirname(resolved), { recursive: true });
  const db = new Database(resolved);
  db.exec(readFileSync(join(here, "schema.sql"), "utf8"));

  // Durability over speed. These are human decisions, arriving a few per
  // minute at most; there is nothing to gain by letting them sit unflushed,
  // and a hard kill of the process must not be able to lose one.
  db.pragma("synchronous = FULL");

  // Check the WAL back into the main file on a clean exit. Without this the
  // database file can look empty to anything that inspects it directly while
  // the real contents sit in a sidecar -- and a sidecar is easy to delete by
  // accident.
  const close = () => {
    try {
      db.pragma("wal_checkpoint(TRUNCATE)");
      db.close();
    } catch {
      /* already closed */
    }
  };
  for (const signal of ["SIGINT", "SIGTERM", "exit"]) {
    process.once(signal, close);
  }

  console.log(`  database: ${resolved}`);
  return db;
}

export const nowUtc = () => new Date().toISOString();

/** Slug that stays readable in a URL and stays unique without a UUID column. */
export function slug(name, existing) {
  const base =
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 48) || "item";
  if (!existing.has(base)) return base;
  for (let i = 2; ; i += 1) {
    const candidate = `${base}-${i}`;
    if (!existing.has(candidate)) return candidate;
  }
}
