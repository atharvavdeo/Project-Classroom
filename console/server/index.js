/** Self-hosted entry point: one process serving the console, the media and
 *  the API. See `server/app.js` for the routes. */
import { createApp } from "./app.js";

const port = Number(process.env.PORT ?? 5179);

const app = await createApp();
app.listen(port, () => {
  console.log(`drishti console on http://localhost:${port}`);
  console.log(
    `  store: ${process.env.DATABASE_URL ? "postgres (DATABASE_URL)" : "sqlite"}`,
  );
  if (!process.env.CEREBRAS_API_KEY) {
    console.log("  CEREBRAS_API_KEY unset — image description will 503");
  }
});
