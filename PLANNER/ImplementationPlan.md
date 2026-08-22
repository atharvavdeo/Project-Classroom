# Implementation Plan

## Phase 1: Evidence foundation

### Slice 1: Run health and model-route status

**What this delivers:** An operator can open the existing console and see whether the selected run, AlphaPose, each Roboflow model route, and SAM3 are available, unavailable, or blocked with a concrete reason before reviewing any evidence.

**Depends on:** none

**Touches:** `pipeline/run_manifest.py` — model/config records and run status; `pipeline/artifacts.py` — declared per-model artifact namespaces; `configs/models/alphapose_crowd_baseline.yaml`, `configs/models/roboflow_phone.yaml` (new), `configs/models/roboflow_chit.yaml` (new), `configs/models/sam3_verifier.yaml` (new) — immutable route, checkpoint/API, prompt and licence configuration; `tools/preflight_rtx3090.py` — launch probes that fail closed; `classroom/api.py` — `GET /api/health`; `classroom/static/console.html` — run-health banner; `tests/test_preflight.py` — missing credential, weights, licence and SAM3 configuration cases.

### Slice 2: Canonical evidence and review-state contract

**What this delivers:** Every observation has an immutable source timestamp, source-frame geometry, model/version provenance, camera/quality state, and one explicit non-verdict disposition that is safe to render in the review website.

**Depends on:** Slice 1

**Touches:** `classroom/schema.sql` — versioned `detection`, `pose_observation`, `event`, `human_review` and audit records; `classroom/db.py` — additive migration and append-only helpers; `pipeline/evidence.py` — event-package contract; `pipeline/labels.py` — allowed states `not_assessable`, `context_observation`, `signal`, `review_candidate`, `reviewer_confirmed_policy_violation`, `reviewer_dismissed` and `need_better_view`; `tests/test_observational_language.py` and `tests/test_evidence.py` — no verdict vocabulary, no erased evidence and no unstated model route.

### Slice 3: Independent reference and hard-negative workbench

**What this delivers:** Annotators can create a frozen, source-frame reference pack containing known phones/chits, explicitly reviewed empty frames, and the project's real confusers without exposing it to inference.

**Depends on:** Slice 2

**Touches:** `tools/annotate_boxes.py`, `tools/annotate_timeline.py` and `tools/export_labels.py` — phone, paper/chit, mouse, keyboard, monitor/bezel, placard, bottle, paper and occlusion annotations; `classroom/schema.sql` — `annotation`, `box_annotation`, `reviewed_frame` and `annotated_span`; `pipeline/reference_manifest.py` and `pipeline/config_validation.py` — source-hash/PTS/class validation and inference isolation; `tests/test_reference_isolation.py` and `tests/test_evaluate.py` — reference leakage, explicit negative and incomplete-coverage cases.

## Phase 2: Roboflow and SAM3 evidence refinement

### Slice 4: Separate Roboflow phone and paper/chit routes

**What this delivers:** Each person/desk context crop receives separately versioned Roboflow phone and paper/chit observations, with API failure recorded as unavailable evidence rather than an empty detection.

**Depends on:** Slices 1–3

**Touches:** `pipeline/chit_detector.py` — generalized Roboflow client and coordinate projection; `pipeline/object_detection.py` — source-coordinate object records without COCO class substitution; `tools/attach_chit_detections.py` and `tools/run_person_timeline.py` — independent phone/chit invocation and artifact paths; `configs/models/roboflow_phone.yaml` and `configs/models/roboflow_chit.yaml` (new) — exact workspace/model/version/class-map configuration; `tests/test_detect_real.py` and `tests/test_crop_projection.py` — timeout, rate-limit, malformed response, source-projection and one-route-unavailable behavior.

### Slice 5: SAM3 prompted mask and track verifier

**What this delivers:** A Roboflow proposal can be passed to SAM3 as a boxed exemplar in the same source crop/video span, producing source-projected masks, temporal object tracks, prompt provenance and an explicit disagreement/absence state without allowing SAM3 to invent a violation.

**Depends on:** Slice 4

**Touches:** `pipeline/sam3_verifier.py` (new) — local SAM3 adapter, box/exemplar prompt, mask projection, track continuity and failure handling; `pipeline/artifacts.py` — `09_object/sam3_verifier/` masks, prompts and tracks; `tools/run_sam3_verifier.py` (new) — bounded batch execution over Roboflow proposals only; `configs/models/sam3_verifier.yaml` — checkpoint, precision, prompt templates and sampling; `tests/test_sam3_verifier.py` (new) — multiple instances, prompt drift, lost/reappearing target, mask outside crop, unavailable model and no-proposal behavior.

### Slice 6: Detector–segmenter fusion with retained disagreement

**What this delivers:** The pipeline records a phone/chit proposal, SAM3 spatial-temporal support, negative context and uncertainty as separate facts, then creates only `signal`, `context_observation` or `not_assessable`—never a machine cheating verdict.

**Depends on:** Slices 2, 4 and 5

**Touches:** `pipeline/temporal_confirmation.py` — class/geometry/time grouping extended with SAM3 tracks; `pipeline/evidence.py` — fused evidence packet and disagreement reason; `pipeline/gates.py` — non-destructive route states; `pipeline/ranking.py` — transparent evidence chips rather than a probability; `tests/test_temporal_confirmation.py`, `tests/test_gate_audit.py` and `tests/test_evidence.py` — Roboflow-only, SAM3-only, conflicting, intermittent, duplicate and unavailable evidence.

### Slice 7: Paper/chit context and movement safeguards

**What this delivers:** Paper/chit signals are assessed against the desk, keyboard, mouse, monitor, wrist visibility and object movement so a paper-looking region under a working hand is preserved as context or uncertainty rather than promoted by a universal size/confidence rule.

**Depends on:** Slices 4–6

**Touches:** `pipeline/chit_evidence.py` — camera-normalized geometry, negative-context overlap, desk-relative motion and abstention facts; `pipeline/person_timeline.py` — paper object association across samples without treating persistence as guilt; `pipeline/pose.py` — wrist confidence/visibility transfer; `tools/score_chit_evidence.py` — per-track evidence explanations and rendered rejected examples; `tests/test_chit_evidence.py` (new) — keyboard/mouse under hand, monitor image, white shirt, partition, answer sheet, water bottle, stationary desk paper, occluded wrist and camera-scale changes.

### Slice 8: Phone false-positive safeguards

**What this delivers:** Phone evidence is evaluated separately from chit evidence and can distinguish supported, conflicting, context-bound and not-assessable observations while retaining placards, mice, keyboards and bezels as reviewable hard negatives.

**Depends on:** Slices 3–6

**Touches:** `pipeline/object_detection.py` — Roboflow phone taxonomy and native-footprint abstention; `pipeline/temporal_confirmation.py` — phone track continuity and object-context association; `pipeline/evaluate.py` — phone-only outcome categories; `tools/render_phones.py` and `tools/render_detections.py` — source overlays/contact sheets for false positives; `tests/test_phone_evidence.py` (new) — placard, mouse, keyboard, bezel, monitor, hand-held phone, partially hidden phone, tiny phone and API/SAM3 disagreement.

## Phase 3: Pose, occupancy and behaviour policy

### Slice 9: AlphaPose observability and direction episodes

**What this delivers:** AlphaPose remains the sole pose source for wrist, head/neck and body-direction rules, and produces baseline-relative direction episodes with explicit low-resolution/occlusion abstentions instead of a universal head-angle rule.

**Depends on:** Slices 1 and 2

**Touches:** `pipeline/pose.py` and `pipeline/head_pose.py` — head scale, yaw/pitch confidence, personal baseline and episode records; `pipeline/person_timeline.py` — direction episode summaries rather than only away-fractions; `tools/run_head_analysis.py` — overlays and per-track timeline; `tests/test_head_pose.py` and `tests/test_pose_manifest_equivalence.py` — permanently turned subject, short glance, VFR gap, intermittent occlusion, low-confidence keypoint and track split cases.

### Slice 10: Hand activity, hidden-hand and object-interaction states

**What this delivers:** Each candidate time span records visible hand activity, below-desk activity, hand visibility duration and object interaction separately, so a hand hidden for a long time becomes `not_assessable` unless independent visible evidence exists.

**Depends on:** Slices 6, 7 and 9

**Touches:** `pipeline/person_timeline.py` — hand visibility, wrist/desk transitions and object interaction timeline; `pipeline/pose.py` — wrist confidence and desk/lap context; `pipeline/evidence.py` — combined packet facts; `tools/run_person_timeline.py` and `tools/render_annotated_video.py` — visible overlays; `tests/test_hand_activity.py` (new) — typing, mouse use, writing, drinking, temporary self-occlusion, prolonged occlusion, hand under desk with no object, and hand/object temporal mismatch.

### Slice 11: Seat occupancy, staff exclusion and movement transitions

**What this delivers:** The system can show seat occupied, empty, candidate-like, staff/aisle-like or not-assessable over time, logging stand/leave/return transitions without claiming a track is a person or that a seat change is impersonation.

**Depends on:** Slices 2 and 9

**Touches:** `pipeline/calibration.py` and `pipeline/seat_association.py` — seat, aisle and invigilator-path geometry; `pipeline/tracking.py` — tracker continuity caveats; `pipeline/person_timeline.py` — seat-level occupancy observations; `tools/calibrate.py` and `tools/propose_calibration.py` — reviewed aisle/staff masks; `tests/test_seat_association.py` and `tests/test_occupancy.py` (new) — standing invigilator, passing candidate, empty seat, seated occlusion, simultaneous adjacent movement and calibration epoch change.

### Slice 12: Cross-signal review-candidate policy

**What this delivers:** A machine can place a time-bounded event into the human queue only when independent, quality-adequate signals overlap; ordinary mouse/keyboard work, one detector hit, an unresolvable hand or a single head turn remain non-escalating observations.

**Depends on:** Slices 6–11

**Touches:** `pipeline/ranking.py` — evidence-source independence, quality penalties and candidate state; `pipeline/evidence.py` — policy explanation and competing evidence; `pipeline/gates.py` — `not_assessable`, context and uncertainty routing; `tools/generate_final_report.py` — no verdict-language assertion; `tests/test_review_policy.py` (new) — one-modality-only, repeated same-modality, overlapping independent signals, staff path, permitted paper, no baseline, low coverage and quality interruption.

## Phase 4: Review dashboard and human decisions

### Slice 13: Append-only reviewer workflow

**What this delivers:** A reviewer can confirm a policy violation, dismiss it with a factual reason, or request a better view; each decision is append-only, attributable and cannot alter detector/pose evidence or model rank.

**Depends on:** Slices 2 and 12

**Touches:** `classroom/schema.sql` — expanded `human_review` decision/state/reason fields and audit history; `classroom/db.py` — append-only review writes; `classroom/api.py` — review-state endpoints and validation; `tests/test_review_api.py` (new) — invalid transition, overwrite attempt, evidence immutability, reviewer attribution and no verdict-language violation.

### Slice 14: Evidence-first dashboard

**What this delivers:** The existing console becomes a usable review website with a queue, seat timeline, occupancy/staff state, source clip, masks/boxes/skeletons, quality limits, evidence chips and human actions, while clearly separating `not_assessable`, context and review candidates.

**Depends on:** Slices 11 and 13

**Touches:** `classroom/api.py` — queue, event detail, seat timeline, occupancy and artifact endpoints; `classroom/static/console.html` — accessible queue, filters, clip panel, evidence provenance, reviewer controls and abstention explanation; `pipeline/evidence.py` — stable dashboard packet schema; `tests/test_console_contract.py` (new) — empty run, no candidate, unavailable artifact, event with conflicting models, not-assessable evidence and reviewer-decision rendering.

### Slice 15: Honest dashboard metrics and model comparison

**What this delivers:** Operators can see per-class/model reference count, TP/FP/FN, precision/recall/F1, false-positive category, abstention, quality/occlusion, candidate-routing, stage-loss, review outcome and review-load measures only where exhaustive annotations support them; every dashboard value links back to its denominator and artifact set, while undefined values stay explicitly undefined.

**Depends on:** Slices 3, 8, 12 and 14

**Touches:** `pipeline/evaluate.py`, `pipeline/reference_manifest.py` and `pipeline/gate_audit.py` — phone/chit-separated reference evaluation, coverage and loss attribution; `tools/evaluate_run.py` and `tools/compare_configurations.py` — model-selection and failure-cluster reports; `classroom/api.py` — metrics endpoints with completeness state; `classroom/static/console.html` — metrics views that suppress undefined values rather than displaying invented zeros; `tests/test_metrics_zero_detections.py`, `tests/test_evaluation_artifacts.py` and `tests/test_dashboard_metrics.py` (new) — incomplete reference set, empty detections, abstention-only run, model unavailable and phone/chit separation.

## Phase 5: Release evidence

### Slice 16: Full-run regression and release packet

**What this delivers:** One documented command creates a new immutable run, executes the chosen AlphaPose + Roboflow + SAM3 routes sequentially, renders all known hard cases, runs reference evaluation, and publishes a dashboard-ready release packet without a cheating verdict.

**Depends on:** Slices 1–15

**Touches:** `tools/run_product_zero.py` and `tools/run_person_timeline.py` — deterministic orchestration and resource release; `pipeline/run_manifest.py` and `pipeline/artifacts.py` — parent run, model hashes, status and seal; `tools/generate_final_report.py` — release report with coverage/limitations; `README.md`, `docs/FINAL_LOCK_IN_AUDIT.md` and `docs/CLASSIFICATION_AND_OPEN_QUESTIONS.md` — updated operating contract; `tests/test_endtoend.py`, `tests/test_realworld_pipeline.py` and `tests/test_regressions.py` — environment preflight, known false positives, VFR, calibration and review-state regression coverage.
