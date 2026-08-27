/**
 * Cloudflare Worker: the console, its API, and its media, from one deploy.
 *
 * Three bindings do the work of the Node server's filesystem and SQLite file:
 *
 *   D1     the decisions and the hierarchy. D1 *is* SQLite, so the schema is
 *          the same file the local server uses, minus the PRAGMAs Workers
 *          manages itself.
 *   R2     the recordings, the run bundles and the review crops -- ~105 MB of
 *          derived bytes that have no business in a git repository or a
 *          Worker script. R2 charges nothing for egress, which matters when
 *          the payload is video.
 *   ASSETS the built console.
 *
 * The URL shape is identical to the local server (`/media/...`, `/data/...`,
 * `/crops/...`), so the front end cannot tell the two apart and nothing in it
 * needs a deployment switch.
 *
 * The pipeline does not run here and must not: stages 1-15 are GPU work that
 * would blow the CPU limit on the first frame. This Worker serves what a run
 * already produced.
 */
import { makeRouter, slug } from "../server/routes.js";

/** D1 speaks the same SQL as better-sqlite3; only the call shape differs. */
function d1Store(db) {
  return {
    kind: "d1",
    async all(sql, params = []) {
      return (await db.prepare(sql).bind(...params).all()).results ?? [];
    },
    async get(sql, params = []) {
      return await db.prepare(sql).bind(...params).first();
    },
    async run(sql, params = []) {
      const res = await db.prepare(sql).bind(...params).run();
      return { lastId: res.meta?.last_row_id ?? null };
    },
  };
}

/** What is in the bucket, in the same shape the Node server reports from
 *  disk, so the "attach a recording" picker works identically. */
async function r2Assets(bucket) {
  const pick = async (prefix, test) => {
    const out = [];
    let cursor;
    do {
      const page = await bucket.list({ prefix, cursor, limit: 1000 });
      for (const o of page.objects) {
        const name = o.key.slice(prefix.length);
        if (name && test(name)) {
          out.push({ url: `/${o.key}`, name, bytes: o.size });
        }
      }
      cursor = page.truncated ? page.cursor : undefined;
    } while (cursor);
    return out;
  };

  // Crop folders are prefixes, not objects. Delimited listing returns them as
  // `delimitedPrefixes` without walking every file inside.
  const cropDirs = [];
  {
    let cursor;
    do {
      const page = await bucket.list({
        prefix: "crops/",
        delimiter: "/",
        cursor,
        limit: 1000,
      });
      for (const p of page.delimitedPrefixes ?? []) {
        cropDirs.push({
          url: `/${p.replace(/\/$/, "")}`,
          name: p.slice("crops/".length).replace(/\/$/, ""),
          bytes: 0,
        });
      }
      cursor = page.truncated ? page.cursor : undefined;
    } while (cursor);
  }

  return {
    media: await pick("media/", (n) => /\.(mp4|webm|mkv)$/i.test(n)),
    bundles: await pick(
      "data/",
      (n) => n.endsWith(".json") && !n.includes(".overlay"),
    ),
    overlays: await pick("data/", (n) => n.endsWith(".overlay.json")),
    crops: cropDirs,
  };
}

const SYSTEM_PROMPT = `You are describing a still frame from exam-hall CCTV for a human reviewer.

Rules you must follow exactly:
- Describe ONLY what is visible in the image. Never infer intent.
- You are NOT deciding whether anyone cheated. That is a human's decision and you must not state or imply a verdict.
- If the object is too small, blurred, or occluded to identify, say so plainly. "Cannot tell" is a correct and useful answer.
- Never invent detail that is not in the pixels. No names, no seat numbers, no time of day.
- Prefer the boring, literal reading. Most objects in an exam hall are keyboards, mice, monitors, water bottles, answer sheets and question papers.

Reply as strict JSON, no markdown fence, exactly:
{"title": "<max 6 words, what is visible>", "description": "<1-2 sentences, literal, max 45 words>", "object_guess": "<one of: phone, paper, keyboard, mouse, monitor, bottle, hand_only, cannot_tell>", "confidence": "<one of: clear, unclear, cannot_tell>"}`;

/** The Cerebras proxy. The key is a Worker secret and never reaches a
 *  browser; the prompt is fixed here for the same reason the reason-code
 *  vocabulary is closed. */
async function describeImage(key, { image, question }) {
  if (typeof image !== "string" || !image.startsWith("data:image/")) {
    return { status: 400, body: { error: "image must be a data: URL" } };
  }

  const upstream = await fetch("https://api.cerebras.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      // Cerebras sits behind Cloudflare, which 1010s a default agent string.
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    },
    body: JSON.stringify({
      model: "gemma-4-31b",
      max_tokens: 300,
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: [
            {
              type: "text",
              text:
                typeof question === "string" && question.trim()
                  ? question.slice(0, 300)
                  : "Describe what is visible around this person's hands.",
            },
            { type: "image_url", image_url: { url: image } },
          ],
        },
      ],
    }),
  });

  const text = await upstream.text();
  if (!upstream.ok) {
    return {
      status: upstream.status,
      body: { error: `Cerebras ${upstream.status}`, detail: text.slice(0, 400) },
    };
  }

  const content = JSON.parse(text)?.choices?.[0]?.message?.content ?? "";
  let parsed = null;
  try {
    parsed = JSON.parse(
      content.replace(/^```(?:json)?/i, "").replace(/```$/, "").trim(),
    );
  } catch {
    parsed = null;
  }
  return {
    status: 200,
    body: { model: "gemma-4-31b", parsed, raw: parsed ? null : content },
  };
}

const CACHE = {
  ".mp4": "public, max-age=31536000, immutable",
  ".jpg": "public, max-age=31536000, immutable",
  ".json": "public, max-age=300",
};

const PUBLIC_READ_METHODS = new Set(["GET", "HEAD"]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const { pathname } = url;

    /* ---------------------------------------------------------- landing */

    // The root is the landing page; the console lives under /app.
    //
    // Served by rewriting rather than redirecting, so the marketing page keeps
    // the bare domain. Everything under /app falls through to the SPA, which
    // Workers Assets serves through `not_found_handling: single-page-application`
    // -- the console's own asset URLs are absolute (/assets/...), so it does not
    // care what path it was reached at.
    if (request.method === "GET" && (pathname === "/" || pathname === "/index.html")) {
      const page = await env.ASSETS.fetch(new URL("/landing.html", url));
      if (page.ok) {
        // A fresh copy of the headers: the Response from ASSETS is immutable.
        const headers = new Headers(page.headers);
        headers.set("content-type", "text/html; charset=utf-8");
        return new Response(page.body, { status: 200, headers });
      }
      // If the landing file is missing from the build, the console is a better
      // answer than a 404 on the front door.
    }

    // The public deployment is intentionally a viewer. Keep the existing
    // D1/R2-backed evidence readable, but reject every state-changing API
    // request before it can reach the upload handler or shared router.
    if (pathname.startsWith("/api/") && !PUBLIC_READ_METHODS.has(request.method)) {
      return Response.json(
        {
          error: "Public showcase mode",
          detail:
            "Uploads, reviewer changes, job processing, and AI requests are disabled. Existing project data remains available to view.",
        },
        {
          status: 403,
          headers: { "cache-control": "no-store", allow: "GET, HEAD" },
        },
      );
    }

    /* ---------------------------------------------------------- uploads */

    // Handled before the shared router, for the same reason the Node server
    // mounts it before its JSON body parser: the body is a recording, and the
    // router's job is to work with parsed JSON. Streamed straight into R2 --
    // a Worker has ~128 MB of memory and a hall recording is larger than that,
    // so buffering it would fail on exactly the files that matter most.
    if (request.method === "PUT" && pathname.startsWith("/api/uploads/")) {
      if (!env.ASSETS_BUCKET) {
        return Response.json(
          { error: "No R2 bucket is bound; uploads have nowhere to go." },
          { status: 503 },
        );
      }
      const parts = pathname.split("/").filter(Boolean); // api uploads centre file
      const centreId = parts[2];
      const filename = (parts[3] ?? "").replace(/[^\w.\-]/g, "_");
      if (!centreId || !/\.(mp4|mkv|webm)$/i.test(filename)) {
        return Response.json(
          { error: "Only mp4, mkv or webm, filed under a centre." },
          { status: 400 },
        );
      }

      const store = d1Store(env.DB);
      const centre = await store.get("SELECT id FROM centres WHERE id = ?", [
        centreId,
      ]);
      if (!centre) {
        return Response.json({ error: "No such centre." }, { status: 404 });
      }
      if (!request.body) {
        return Response.json({ error: "Empty upload." }, { status: 400 });
      }

      const object = await env.ASSETS_BUCKET.put(`media/${filename}`, request.body, {
        httpMetadata: {
          contentType: request.headers.get("content-type") || "video/mp4",
        },
      });

      const name = url.searchParams.get("name") || filename;
      const taken = new Set(
        (await store.all("SELECT id FROM videos")).map((r) => r.id),
      );
      const id = slug(name, taken);
      await store.run(
        `INSERT INTO videos (id, centre_id, name, media_url, created_utc)
         VALUES (?,?,?,?,?)`,
        [id, centreId, name, `/media/${filename}`, new Date().toISOString()],
      );

      return Response.json(
        {
          video_id: id,
          media_url: `/media/${filename}`,
          bytes: object?.size ?? 0,
          // Uploading and running are separate on purpose: a recording can be
          // attached now and processed when the GPU is free.
          next: `POST /api/videos/${id}/process`,
        },
        { status: 201 },
      );
    }

    /* ------------------------------------------------------------- API */

    if (pathname.startsWith("/api/")) {
      const route = makeRouter({
        store: d1Store(env.DB),
        assets: () => r2Assets(env.ASSETS_BUCKET),
        describe: env.CEREBRAS_API_KEY
          ? (payload) => describeImage(env.CEREBRAS_API_KEY, payload)
          : null,
        agentToken: env.AGENT_TOKEN || null,
      });

      let body;
      if (request.method === "POST") {
        body = await request.json().catch(() => ({}));
      }

      try {
        const out = await route(request.method, pathname, body);
        if (out?.text !== undefined) {
          return new Response(out.text, {
            status: out.status,
            // Extra headers from the route, e.g. content-disposition on the
            // CSV export. Spread after content-type so a route can override it.
            headers: { "content-type": out.type, ...(out.headers ?? {}) },
          });
        }
        return Response.json(out.body, { status: out.status });
      } catch (e) {
        return Response.json({ error: String(e?.message ?? e) }, { status: 500 });
      }
    }

    /* ------------------------------------------------------- run assets */

    if (/^\/(media|data|crops)\//.test(pathname)) {
      const key = pathname.slice(1);

      // R2 has to be switched on for the account before a bucket can exist,
      // and that is a dashboard action. Say which step is missing rather than
      // 404ing: a missing recording and a missing bucket look identical from
      // the browser, and only one of them is fixed by re-uploading.
      if (!env.ASSETS_BUCKET) {
        return Response.json(
          {
            error: "Run artifacts are not available: no R2 bucket is bound.",
            fix: "Enable R2 in the Cloudflare dashboard, then `wrangler r2 bucket create drishti-runs`, uncomment the r2_buckets block in wrangler.toml, redeploy, and run `node tools/upload-assets.mjs`.",
          },
          { status: 503 },
        );
      }

      // The Range header is parsed here rather than handed to R2 as
      // `range: request.headers`. That form is documented but returned the
      // whole object with a 200 under local emulation -- measured: a request
      // for bytes 1000000-1000999 came back as 4,349,309 bytes. A video the
      // browser cannot range-request is a video it must download to the end
      // before it can scrub, which for a full recording is unusable, and the
      // failure is silent. An explicit offset/length cannot degrade quietly.
      const rangeHeader = request.headers.get("range");
      const match = rangeHeader?.match(/^bytes=(\d*)-(\d*)$/);

      const options = {};
      if (match && (match[1] !== "" || match[2] !== "")) {
        if (match[1] === "") {
          options.range = { suffix: Number(match[2]) };
        } else {
          options.range = { offset: Number(match[1]) };
          if (match[2] !== "") {
            options.range.length = Number(match[2]) - Number(match[1]) + 1;
          }
        }
      } else {
        // Conditional requests only matter for a full fetch; combining them
        // with a range asks R2 two questions at once.
        options.onlyIf = request.headers;
      }

      const object = await env.ASSETS_BUCKET.get(key, options);
      if (!object) {
        return new Response("Not found", {
          status: match ? 416 : 404,
        });
      }

      const headers = new Headers();
      object.writeHttpMetadata(headers);
      headers.set("etag", object.httpEtag);
      headers.set("accept-ranges", "bytes");
      const ext = pathname.slice(pathname.lastIndexOf("."));
      if (CACHE[ext]) headers.set("cache-control", CACHE[ext]);

      if (!object.body) {
        // `onlyIf` matched: the client already has this byte-for-byte.
        return new Response(null, { status: 304, headers });
      }

      if (options.range) {
        const offset = object.range?.offset ?? Number(match[1] || 0);
        const length = object.range?.length ?? object.size - offset;
        headers.set(
          "content-range",
          `bytes ${offset}-${offset + length - 1}/${object.size}`,
        );
        headers.set("content-length", String(length));
        return new Response(object.body, { status: 206, headers });
      }

      return new Response(object.body, { headers });
    }

    /* -------------------------------------------------------- the console */

    // Anything else is a console route; the SPA resolves it client-side, so a
    // miss falls back to index.html rather than 404ing a deep link.
    return env.ASSETS.fetch(request);
  },
};
