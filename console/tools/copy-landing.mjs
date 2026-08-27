/** Put the landing page into the build output.
 *
 * `landing.html` is not a Vite entry point on purpose. It is a single
 * self-contained file with inline CSS and JS and no imports, so running it
 * through the bundler would gain nothing and would rewrite its asset paths --
 * and its one image reference points at R2 (`/crops/...`), which the Worker
 * serves and the bundler must not try to resolve at build time.
 *
 * `vite.config.ts` sets `copyPublicDir: false` because `public/` is ~105 MB of
 * run artifacts, so there is no automatic path for a static file either. Hence
 * this: one copy, run after the build.
 */
import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../landing.html");
const dest = resolve(here, "../dist/landing.html");

if (!existsSync(src)) {
  console.error(`landing.html is missing at ${src}`);
  process.exit(1);
}
mkdirSync(dirname(dest), { recursive: true });
copyFileSync(src, dest);
console.log(`landing.html -> dist/ (${(statSync(dest).size / 1024).toFixed(1)} kB)`);
