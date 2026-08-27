import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The API lives in `server/`, not here.
 *
 * It was a Vite middleware while the only endpoint was the Cerebras proxy.
 * Once decisions had to persist, a dev-server plugin was the wrong home: it
 * does not exist in a production build, so anything written through it would
 * have worked only on the machine that authored it. Vite now proxies to the
 * same process that serves the built console.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5178,
    strictPort: true,
    proxy: {
      "/api": {
        target: `http://localhost:${process.env.API_PORT ?? 5179}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    // Do not copy `public/` into the build.
    //
    // `public/` holds the run artifacts -- the recordings, the overlay
    // bundles and the review crops, ~105 MB of derived bytes. Vite's default
    // is to copy all of it into `dist/`, which produced a 106 MB build. Two
    // things broke: it is far past what Cloudflare will accept as Worker
    // static assets, and because Workers Assets is consulted before the
    // Worker runs, `/media/...` was answered from that copy instead of from
    // R2 -- so the Worker's range-request handling never executed and video
    // seeking silently fell back to whole-file downloads.
    //
    // The artifacts are served directly from `public/` by the Node server,
    // and from R2 by the Worker. Neither path wants them duplicated here.
    copyPublicDir: false,
  },
});
