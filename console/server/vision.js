/**
 * Server-side proxy for Cerebras inference.
 *
 * The key is read from the environment and never reaches the browser. A key in
 * a bundle is a key that has to be rotated.
 *
 * The prompt is fixed here for the same reason the reason-code vocabulary is
 * closed: a caption from a model that was free to say anything can assert
 * things the pipeline never measured. This one describes and refuses to judge.
 */
const SYSTEM_PROMPT = `You are describing a still frame from exam-hall CCTV for a human reviewer.

Rules you must follow exactly:
- Describe ONLY what is visible in the image. Never infer intent.
- You are NOT deciding whether anyone cheated. That is a human's decision and you must not state or imply a verdict.
- If the object is too small, blurred, or occluded to identify, say so plainly. "Cannot tell" is a correct and useful answer.
- Never invent detail that is not in the pixels. No names, no seat numbers, no time of day.
- Prefer the boring, literal reading. Most objects in an exam hall are keyboards, mice, monitors, water bottles, answer sheets and question papers.

Reply as strict JSON, no markdown fence, exactly:
{"title": "<max 6 words, what is visible>", "description": "<1-2 sentences, literal, max 45 words>", "object_guess": "<one of: phone, paper, keyboard, mouse, monitor, bottle, hand_only, cannot_tell>", "confidence": "<one of: clear, unclear, cannot_tell>"}`;

const MODEL = "gemma-4-31b";

export async function describeImage({ image, question }) {
  const key = process.env.CEREBRAS_API_KEY;
  if (!key) {
    return {
      status: 503,
      body: {
        error:
          "CEREBRAS_API_KEY is not set on the server. Export it and restart.",
      },
    };
  }
  if (typeof image !== "string" || !image.startsWith("data:image/")) {
    return { status: 400, body: { error: "image must be a data: URL" } };
  }

  const upstream = await fetch("https://api.cerebras.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      // Cerebras sits behind Cloudflare, which 1010s a default undici agent.
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    },
    body: JSON.stringify({
      model: MODEL,
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

  const payload = JSON.parse(text);
  const content = payload?.choices?.[0]?.message?.content ?? "";

  // The model is told to return bare JSON, but a fence still shows up
  // occasionally. Strip it; if it still will not parse, hand back the raw text
  // rather than pretending it was structured.
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
    body: {
      model: MODEL,
      parsed,
      raw: parsed ? null : content,
      usage: payload?.usage ?? null,
    },
  };
}
