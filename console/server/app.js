/**
 * Node adapter: the shared router, plus the filesystem.
 *
 * One process serves the console, the media and the API, so a self-hosted
 * deployment is a single thing to run. The routes themselves live in
 * `routes.js` and are shared with the Cloudflare Worker.
 */
import express from "express";
import { createWriteStream, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { makeRouter, slug } from "./routes.js";
import { openStore } from "./store.js";
import { describeImage } from "./vision.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const PUBLIC = join(root, "public");
const DIST = join(root, "dist");

/** What is actually on disk, so attaching a recording is a pick, not a typed
 *  path. A path the user could type is a path that can be wrong. */
function localAssets() {
  const list = (dir, test) => {
    const full = join(PUBLIC, dir);
    if (!existsSync(full)) return [];
    return readdirSync(full)
      .filter(test)
      .map((f) => ({
        url: `/${dir}/${f}`,
        name: f,
        bytes: statSync(join(full, f)).size,
      }));
  };
  return {
    media: list("media", (f) => /\.(mp4|webm|mkv)$/i.test(f)),
    bundles: list("data", (f) => f.endsWith(".json") && !f.includes(".overlay")),
    overlays: list("data", (f) => f.endsWith(".overlay.json")),
    crops: existsSync(join(PUBLIC, "crops"))
      ? readdirSync(join(PUBLIC, "crops"))
          .filter((d) => statSync(join(PUBLIC, "crops", d)).isDirectory())
          .map((d) => ({ url: `/crops/${d}`, name: d, bytes: 0 }))
      : [],
  };
}

/**
 * Take an uploaded recording and register it, unprocessed.
 *
 * Streamed to disk rather than buffered: a full hall recording is gigabytes,
 * and holding one in memory to write it out again is the difference between a
 * server that works on the GPU box and one that dies on the first real file.
 *
 * The recording is registered with media only and no run. It will say "not
 * run" in the console until a job finishes, which is the honest state -- an
 * empty dashboard would read as "nothing was found".
 */
function mountUpload(app, store) {
  app.put("/api/uploads/:centreId/:filename", async (req, res) => {
    try {
      const { centreId } = req.params;
      const filename = basename(req.params.filename).replace(/[^\w.\-]/g, "_");
      if (!/\.(mp4|mkv|webm)$/i.test(filename)) {
        return res.status(400).json({ error: "Only mp4, mkv or webm." });
      }
      const centre = await store.get("SELECT id FROM centres WHERE id = ?", [
        centreId,
      ]);
      if (!centre) return res.status(404).json({ error: "No such centre." });

      const dir = join(PUBLIC, "media");
      mkdirSync(dir, { recursive: true });
      const target = join(dir, filename);

      await pipeline(req, createWriteStream(target));
      const bytes = statSync(target).size;
      if (!bytes) {
        return res.status(400).json({ error: "Empty upload." });
      }

      const name = String(req.query.name ?? filename);
      const taken = new Set((await store.all("SELECT id FROM videos")).map((r) => r.id));
      const id = slug(name, taken);
      const at = new Date().toISOString();
      await store.run(
        `INSERT INTO videos (id, centre_id, name, media_url, created_utc)
         VALUES (?,?,?,?,?)`,
        [id, centreId, name, `/media/${filename}`, at],
      );

      res.status(201).json({
        video_id: id,
        media_url: `/media/${filename}`,
        bytes,
        // The caller still has to ask for processing. Uploading and running
        // are separate on purpose: a recording can be attached now and
        // processed when the GPU is free.
        next: `POST /api/videos/${id}/process`,
      });
    } catch (e) {
      console.error(e);
      if (!res.headersSent) res.status(500).json({ error: String(e?.message ?? e) });
    }
  });
}

export async function createApp({ serveStatic = true } = {}) {
  const store = await openStore();
  const route = makeRouter({
    store,
    assets: async () => localAssets(),
    describe: process.env.CEREBRAS_API_KEY
      ? (payload) => describeImage(payload)
      : null,
    agentToken: process.env.AGENT_TOKEN || null,
  });

  const app = express();

  // Before express.json(): an upload is a stream, and the JSON body parser
  // would try to buffer a multi-gigabyte recording into memory first.
  mountUpload(app, store);

  app.use(express.json({ limit: "12mb" }));

  app.use("/api", async (req, res, next) => {
    try {
      const result = await route(req.method, req.originalUrl.split("?")[0], req.body);
      if (!result) return next();
      if (result.text !== undefined) {
        // A route may ask for extra headers -- content-disposition on the CSV
        // export, so a browser saves it rather than rendering it as a page.
        if (result.headers) res.set(result.headers);
        return res.status(result.status).type(result.type).send(result.text);
      }
      res.status(result.status).json(result.body);
    } catch (e) {
      console.error(e);
      if (!res.headersSent) {
        res.status(500).json({ error: String(e?.message ?? e) });
      }
    }
  });

  if (serveStatic) {
    app.use(express.static(PUBLIC, { maxAge: "1h" }));
    if (existsSync(DIST)) {
      app.use(express.static(DIST, { maxAge: "1h" }));
      // Anything that is not an API route is a console route; the SPA resolves
      // it client-side.
      app.get(/^(?!\/api\/).*/, (_req, res) =>
        res.sendFile(join(DIST, "index.html")),
      );
    }
  }

  return app;
}
