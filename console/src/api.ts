/** Thin client over the console API. Nothing here is seeded or invented:
 *  if the tree is empty it is because nobody has created a project yet, and
 *  the UI says exactly that instead of showing example centres. */
import type { ProjectRef, VideoRef } from "./review";

/** One overturned proposal: the human's answer, plus what the machine was
 *  claiming when they gave it. Both halves are needed -- a bare "no" is not a
 *  usable label. */
export interface RejectedRow {
  id: number;
  video_id: string;
  video_name: string;
  centre_name: string | null;
  track_id: number;
  decision: string;
  machine_state: string | null;
  key_class: string | null;
  key_verdict: string | null;
  key_pts_ms: number | null;
  note: string | null;
  reviewer: string;
  decided_utc: string;
}

/** A stretch of a recording a human marked by hand. The only claim in the
 *  system that no model produced. */
export interface ReviewInterval {
  id: number;
  video_id: string;
  from_ms: number;
  to_ms: number;
  kind: "violation" | "review" | "note";
  track_id: number | null;
  note: string | null;
  reviewer: string;
  created_utc: string;
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  const doc = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(doc?.error ?? `${res.status} ${res.statusText}`);
  return doc as T;
}

export interface Asset {
  url: string;
  name: string;
  bytes: number;
}
export interface Assets {
  media: Asset[];
  bundles: Asset[];
  overlays: Asset[];
  crops: Asset[];
}

export interface DecisionRow {
  id: number;
  video_id: string;
  track_id: number;
  decision: "human_confirmed" | "human_dismissed" | "needs_better_view";
  machine_state: string | null;
  key_class: string | null;
  key_verdict: string | null;
  key_pts_ms: number | null;
  note: string | null;
  reviewer: string;
  decided_utc: string;
}

export const api = {
  tree: () => call<{ projects: ProjectRef[] }>("/api/tree"),
  assets: () => call<Assets>("/api/assets"),

  createProject: (name: string) =>
    call<{ projects: ProjectRef[] }>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  createCentre: (projectId: string, name: string) =>
    call<{ projects: ProjectRef[] }>(`/api/projects/${projectId}/centres`, {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  createVideo: (centreId: string, body: Partial<VideoRef> & { name: string }) =>
    call<{ projects: ProjectRef[] }>(`/api/centres/${centreId}/videos`, {
      method: "POST",
      body: JSON.stringify({
        name: body.name,
        media_url: body.video,
        bundle_url: body.bundle,
        overlay_url: body.overlay,
        crops_url: body.crops,
      }),
    }),

  remove: (kind: "projects" | "centres" | "videos", id: string) =>
    call<{ projects: ProjectRef[] }>(`/api/${kind}/${id}`, { method: "DELETE" }),

  decisions: (videoId: string) =>
    call<{ current: Record<string, DecisionRow>; revisions: number }>(
      `/api/videos/${videoId}/decisions`,
    ),

  decide: (
    videoId: string,
    body: {
      track_id: number;
      decision: DecisionRow["decision"];
      machine_state?: string | null;
      key_class?: string | null;
      key_verdict?: string | null;
      key_pts_ms?: number | null;
      note?: string | null;
    },
  ) =>
    call<DecisionRow>(`/api/videos/${videoId}/decisions`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /**
   * Send a recording.
   *
   * XMLHttpRequest rather than fetch, for one reason: upload progress. A hall
   * recording is gigabytes, and a control that says nothing for ten minutes is
   * indistinguishable from one that has silently failed. `fetch` still has no
   * portable way to observe request-body progress.
   *
   * The body is the file itself, not multipart — the server streams it
   * straight to storage, so wrapping it in a form encoding would only add a
   * parse step over several gigabytes.
   */
  upload(
    centreId: string,
    file: File,
    name: string,
    onProgress: (fraction: number) => void,
  ): Promise<{ video_id: string; media_url: string; bytes: number }> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const safe = file.name.replace(/[^\w.\-]/g, "_");
      xhr.open(
        "PUT",
        `/api/uploads/${centreId}/${safe}?name=${encodeURIComponent(name)}`,
      );
      xhr.setRequestHeader("Content-Type", file.type || "video/mp4");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let doc: Record<string, unknown> = {};
        try {
          doc = JSON.parse(xhr.responseText);
        } catch {
          /* fall through to the status check */
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(doc as never);
        } else {
          reject(new Error(String(doc.error ?? `HTTP ${xhr.status}`)));
        }
      };
      xhr.onerror = () => reject(new Error("Upload failed — network error."));
      xhr.send(file);
    });
  },

  /** Queue a run. Returns as soon as the job row exists; the pipeline itself
   *  takes minutes and reports through the job's state. */
  process: (videoId: string) =>
    call<{ job: { id: number; state: string }; note?: string }>(
      `/api/videos/${videoId}/process`,
      { method: "POST", body: JSON.stringify({}) },
    ),

  job: (id: number) =>
    call<{ job: { id: number; state: string; stage: string | null; progress: number; error: string | null; run_key: string | null } }>(
      `/api/jobs/${id}`,
    ),

  /** Every overturned proposal, across every recording. */
  dismissed: () => call<{ rows: RejectedRow[] }>("/api/dismissed"),

  /** Stretches a human marked by hand on the timeline. */
  intervals: (videoId: string) =>
    call<{ rows: ReviewInterval[] }>(`/api/videos/${videoId}/intervals`),

  markInterval: (
    videoId: string,
    body: {
      from_ms: number;
      to_ms: number;
      kind?: "violation" | "review" | "note";
      track_id?: number | null;
      note?: string | null;
    },
  ) =>
    call<{ ok: true }>(`/api/videos/${videoId}/intervals`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  describe: (image: string) =>
    call<{
      parsed: {
        title?: string;
        description?: string;
        object_guess?: string;
        confidence?: string;
      } | null;
      raw: string | null;
    }>("/api/vision", {
      method: "POST",
      body: JSON.stringify({ image }),
    }),
};
