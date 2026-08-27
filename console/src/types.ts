/** Shapes emitted by tools/build_console_bundle.py (schema 1).
 *
 * These mirror the run artifacts rather than restating them. If a field is
 * optional here it is because the pipeline can genuinely omit it -- absence
 * means "not measured", and the UI must render that differently from zero.
 */

export type MachineState =
  | "no_action"
  | "context_observation"
  | "needs_better_view"
  | "review_candidate";

export type HumanState = "human_confirmed" | "human_dismissed";
export type State = MachineState | HumanState;

export type Lane = "object" | "orientation" | "hands" | "presence" | "quality";
export type Severity = "info" | "low" | "medium" | "high";

/** One of the five fusion conditions, with the arithmetic that produced it.
 *  `detail` is never optional: a condition that cannot show its working has
 *  no business on a card. */
export interface Condition {
  name: string;
  passed: boolean;
  detail: string;
}

export interface Flag {
  flag: string;
  lane: Lane;
  severity: Severity;
  evidence: string;
  why: string;
}

export interface Span {
  from_ms: number;
  to_ms: number;
}

export interface OrientationLabel {
  label: string;
  classification:
    | "observed_only"
    | "review_context"
    | "context_requires_second_modality";
  dashboard_action:
    | "display_only"
    | "show_review_context"
    | "show_only_with_second_modality";
  guardrail: string;
  samples?: number;
  events?: number;
}

export interface OrientationEvent extends Span, OrientationLabel {
  duration_ms: number;
  direction?: "left" | "right";
}

export interface Track {
  track_id: number;
  state: State;
  priority: number | null;
  seat: string | null;
  seat_state: string | null;
  modalities: string[];
  reason_codes: string[];
  conditions: Condition[];
  funnel: unknown;

  samples: number | null;
  resolvable_samples: number | null;
  first_seen_ms: number | null;
  last_seen_ms: number | null;
  coverage_of_recording: number | null;
  median_head_scale_px: number | null;

  baseline_yaw: number | null;
  baseline_pitch: number | null;
  baseline_yaw_relative_to_torso: number | null;

  flags: Flag[];
  orientation_labels: OrientationLabel[];
  orientation_events: OrientationEvent[];
  absences: Span[];
  episodes: Span[];
  near_hand_object_counts: Record<string, number>;
}

export interface Stage {
  stage: string;
  state: "complete" | "pending" | "blocked" | "skipped_unavailable" | string;
  reason: string | null;
  covered_ms: number | null;
  total_ms: number | null;
}

export interface Bundle {
  schema: number;
  run: {
    run_id?: string;
    code_revision?: string;
    code_dirty?: boolean;
    calibration_version?: string;
    seed?: number;
    created_utc?: string;
    models?: unknown;
    sources?: unknown;
  };
  video: string;
  duration_ms: number;
  lanes: Lane[];
  counts: {
    tracks: number;
    states: Partial<Record<State, number>>;
    samples: number | null;
    people_with_a_yaw_baseline: number | null;
  };
  coverage_of_recording: { min?: number; median?: number; max?: number };
  coverage_caveat: string;
  note: string;
  stages: Stage[];
  sam3: Record<string, unknown>;
  chit: Record<string, unknown>;
  contract: Record<string, unknown>;
  tracks: Track[];
}

export const STATE_ORDER: State[] = [
  "review_candidate",
  "needs_better_view",
  "context_observation",
  "no_action",
  "human_confirmed",
  "human_dismissed",
];

export const LANES: Lane[] = [
  "object",
  "orientation",
  "hands",
  "presence",
  "quality",
];

/** Which lanes may route a person to review on their own.
 *  Mirrors fusion.py: object evidence routes alone, behaviour does not. */
export const LANE_ROUTES_ALONE: Record<Lane, boolean> = {
  object: true,
  orientation: false,
  hands: false,
  presence: false,
  quality: false,
};
