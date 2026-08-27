import type { ProjectRef } from "./review";

/**
 * Helpers over the tree the API returns.
 *
 * There is deliberately no `PROJECTS` constant here any more. The hierarchy
 * used to ship with a CET exam, a Dahisar centre and a Thane centre baked in.
 * Those were placeholders that read as facts: a centre in the sidebar is a
 * claim that the centre exists and has recordings, and nobody had said so.
 * Every project, centre and recording is now created by the person using the
 * console, and an empty console says it is empty.
 */

export function findVideo(projects: ProjectRef[], videoId: string) {
  for (const p of projects) {
    for (const c of p.centres) {
      const v = c.videos.find((x) => x.id === videoId);
      if (v) return { project: p, centre: c, video: v };
    }
  }
  return null;
}

export function firstProcessed(projects: ProjectRef[]) {
  for (const p of projects) {
    for (const c of p.centres) {
      const v = c.videos.find((x) => x.processed);
      if (v) return { project: p, centre: c, video: v };
    }
  }
  return null;
}

/** Any recording at all, processed or not, so a fresh console can still land
 *  somewhere rather than showing a blank shell. */
export function firstVideo(projects: ProjectRef[]) {
  for (const p of projects) {
    for (const c of p.centres) {
      if (c.videos.length) return { project: p, centre: c, video: c.videos[0] };
    }
  }
  return null;
}
