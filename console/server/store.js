/**
 * The store, behind one async interface, with two backends.
 *
 * SQLite is the default and needs nothing: `npm start` on a laptop, or on the
 * GPU box, writes to a file beside the server. Postgres takes over the moment
 * `DATABASE_URL` is set, which is what a hosted deployment uses -- a serverless
 * function has no durable disk, so a SQLite file there would quietly lose every
 * decision between invocations.
 *
 * The interface is async on both paths even though better-sqlite3 is
 * synchronous. A sync-when-local, async-when-hosted API would mean the calling
 * code behaves differently depending on where it runs, and the difference would
 * only show up in production.
 *
 * The two schemas are kept deliberately close (see schema.sql and the
 * `drishti_console_core` migration). Where they differ the difference is
 * pushed down here, not up into the routes.
 */

const isPg = () => Boolean(process.env.DATABASE_URL);

/** `?` positional params for SQLite, `$1..$n` for Postgres. Written once here
 *  so no route has to know which backend it is talking to. */
function toPg(sql) {
  let i = 0;
  return sql.replace(/\?/g, () => `$${++i}`);
}

export async function openStore() {
  return isPg() ? await openPostgres() : await openSqlite();
}

async function openSqlite() {
  const { openDb } = await import("./db.js");
  const db = openDb();
  return {
    kind: "sqlite",
    async all(sql, params = []) {
      return db.prepare(sql).all(...params);
    },
    async get(sql, params = []) {
      return db.prepare(sql).get(...params) ?? null;
    },
    async run(sql, params = []) {
      const info = db.prepare(sql).run(...params);
      return { lastId: info.lastInsertRowid };
    },
    async close() {
      db.close();
    },
  };
}

async function openPostgres() {
  // Imported lazily so a SQLite-only install does not need the driver present.
  const { default: pg } = await import("pg");
  const pool = new pg.Pool({
    connectionString: process.env.DATABASE_URL,
    // Supabase terminates TLS with its own chain; the pooler hostname does not
    // match the certificate CN, which is expected and is why verification is
    // relaxed here rather than the connection being made plaintext.
    ssl: { rejectUnauthorized: false },
    max: Number(process.env.PGPOOL_MAX ?? 4),
  });

  return {
    kind: "postgres",
    async all(sql, params = []) {
      return (await pool.query(toPg(sql), params)).rows;
    },
    async get(sql, params = []) {
      return (await pool.query(toPg(sql), params)).rows[0] ?? null;
    },
    async run(sql, params = []) {
      // Postgres has no lastInsertRowid; ask for the id back instead. Only the
      // decisions insert needs it, and it is the only INSERT that has one.
      const wantsId = /^\s*insert/i.test(sql) && !/returning/i.test(sql);
      const res = await pool.query(
        toPg(wantsId ? `${sql} RETURNING id` : sql),
        params,
      );
      return { lastId: res.rows[0]?.id ?? null };
    },
    async close() {
      await pool.end();
    },
  };
}
