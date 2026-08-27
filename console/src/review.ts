/** Shapes from tools/build_review_crops.py, plus the reviewer's own decision. */

/** The reviewer's answer, in the contract's own vocabulary.
 *
 *  Not a private UI enum: these are the exact three values the store keeps and
 *  the two that `fusion.assert_machine_state()` refuses to write, so a row can
 *  be replayed into the pipeline without translation. `needs_better_view` is
 *  the reviewer abstaining -- a third answer, never a quiet dismissal. */
export type Decision =
  | "human_confirmed"
  | "human_dismissed"
  | "needs_better_view";

export interface StripFrame {
  pts_ms: number;
  file: string;
  /** True when SAM 3 backed an object in this frame. */
  supported: boolean;
  n: number;
}

export interface CropEntry {
  track_id: number;
  key_pts_ms: number;
  key_class: string | null;
  key_confidence: number | null;
  key_verdict: string | null;
  key_supported: boolean;
  wrist_distance_norm: number | null;
  nearest_wrist: string | null;
  supported_frames: number;
  proposals: number;
  sightings: number;
  key?: string;
  /** The same crop with nothing drawn on it, for the vision model. */
  raw?: string;
  frame?: string;
  /** SAM 3's own annotated crop for the key instant, when it was adjudicated. */
  sam3_crop?: string;
  strip: StripFrame[];
}

export type CropManifest = Record<string, CropEntry>;

/** A project is an exam; a centre is a venue; a recording is one camera.
 *  Every one of them is created by the person using the console -- see
 *  `projects.ts` for why none of this is seeded. */
export interface VideoRef {
  id: string;
  name: string;
  /** Present only when a run bundle exists for it. */
  bundle?: string;
  overlay?: string;
  video?: string;
  crops?: string;
  processed: boolean;
  /** How many people in this recording a human has already answered for. */
  decided?: number;
}

export interface CentreRef {
  id: string;
  name: string;
  videos: VideoRef[];
}

export interface ProjectRef {
  id: string;
  name: string;
  centres: CentreRef[];
}
