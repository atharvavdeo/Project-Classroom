# Project Classroom — Product Zero Implementation PRD

**Status:** Implementation handoff  
**Target:** Offline one-fixed-camera examination-hall analysis on an NVIDIA RTX 3090 (24 GB VRAM)  
**Coverage:** Complete recorded video; 30–60 people; human review is final authority

## 1. Product boundary

Project Classroom converts a full poor-quality CCTV recording into an auditable timeline of seat-linked motion, pose, and object evidence. It identifies clips worth human review and records where the camera/source quality prevents a reliable conclusion.

It is not a cheating classifier. It must not infer intent, issue a verdict, use face/name/biometric identity, or convert model confidence into a disciplinary conclusion. Permitted language is observational: `phone-like object detected`, `secondary-paper-like object detected`, `hand entered lap zone`, `ambiguous_seat`, or `insufficient_visual_evidence`.

The pipeline must process the full decodable duration before evaluation. Known phone/chit frames are an independently maintained **reference manifest**, read only by the post-run evaluator. They must not influence motion candidates, ROI/crop selection, model routing, thresholds, or inference inputs.

Product Zero is inference and measurement only. Training/fine-tuning is explicitly deferred until this pipeline reports error clusters and source limitations.

## 2. Scope

### Included

- Full-video ingest, decoded-PTS inventory, health monitoring, and integrity accounting.
- Preprocessing/mitigation of blackout/whiteout, frozen frames, PTS/decode gaps, blur, exposure changes, camera vibration/repositioning, compression noise, shadows, and recurring environmental motion.
- Reviewed dense-hall spatial calibration: seat, desk, upper-body, lap/bag, monitor/exclusion/environment masks and anonymous seat identifiers.
- Whole-video motion estimation, global-motion compensation when reliable, residual local ROI motion, heatmaps, robust per-seat temporal baseline, segmentation, and ranked candidate events.
- Candidate-only person tracking and matching A/B pose comparison: RTMPose, RTMO, AlphaPose.
- Hand/wrist-guided context crops when pose is reliable; desk/lap/native fallback crops when it is not.
- Matching A/B object comparison: RF-DETR vs D-FINE, unsliced native crop vs selective SAHI slicing.
- Temporal object confirmation, bounded Gemma VLM inspection of flagged evidence packets, and human-reviewable reports.
- A single deterministic artifact directory per run containing all intermediate and final images, overlays, JSON/JSONL records, metrics, logs, and failures.

### Excluded

- Processing only known phone/chit frames or using their annotations before the evaluation stage.
- Any training/fine-tuning, pseudo-labeling, active learning, or trained-model accuracy claim.
- SelfPose3D in Product Zero; it is reserved for a future calibrated multi-camera experiment.
- Multiple timestamp-synchronised cameras, cross-camera tracking/identity, 3D reconstruction, real-time operation, audio, face recognition, eye-tracking verdicts, and alerts.
- `cheating` labels, cheating probability, or automated escalation.
- Silent deletion by a quality/confidence/motion gate.

### Future branch

Only when overlapping cameras are timestamp synchronised and geometrically calibrated, test SelfPose3D against exactly the same evaluation framework. Do not add its dependencies or assumptions now.

## 3. Blocking inputs

Implementation must fail clearly instead of guessing when these are absent.

| Input | Required form | If absent/invalid |
|---|---|---|
| Videos | Original source file(s), immutable SHA-256, readable streams. | Block run. |
| Calibration | Reviewed profile per camera geometry/epoch. | Permit health diagnostic only; block seat-attributed evidence. |
| Reference manifest | Schema in §14, independent phone/chit sightings. | Run inference but mark object evaluation incomplete; make no quality claim. |
| Taxonomy | Product-owner-approved mapping of target and hard-negative objects. | Block detector selection/evaluation. |
| RTX environment | CUDA runtime, weights, license status, writable capacity, launchable models. | Mark named configuration unavailable; never silently substitute. |
| Privacy policy | Access, retention/export/delete rules for CCTV and crops. | Block distribution/export beyond execution owner. |

“Evaluate flagged frames against jammu” is interpreted as **Gemma**, consistent with the agreed scope. If Jammu is a different model/tool, the owner must provide its repository, weights, license, exact runtime, and output contract first.

## 4. Pipeline and execution order

```text
Complete source video
  → preflight + immutable run manifest
  → ingest/PTS/decode inventory
  → video-health + global-disturbance observations
  → calibration validation + camera alignment epoching
  → whole-video motion / residual ROI scan
  → temporal segmentation + candidate manifest
  → candidate-only person detection / tracking / seat association
  → candidate-only pose A/B
  → hand/desk/lap/natural-context crop manifest
  → detector + SAHI A/B
  → temporal evidence confirmation
  → marked Gemma evidence review
  → ranking, artifact index, post-hoc reference evaluation, final report
```

Stages through segmentation operate across all decodable source timestamps. Later stages may consume only the full-video candidate manifest and retained low-priority/uncertain candidates. No model is allowed to inspect the reference manifest.

### Comparison-first order

1. Build and test ingest, quality, calibration, motion, ROI, and segmentation without neural-model dependency.
2. Freeze `candidate_manifest.jsonl`: video hash, PTS/frame, calibration version, event ID, health state, and source crop geometry.
3. Run tracking and RTMPose/RTMO/AlphaPose on the identical pose manifest. Do not select a pose model yet.
4. Freeze `object_crop_manifest.jsonl` from the same events. Run RF-DETR/D-FINE native inference, then controlled SAHI variants on the same crop rows.
5. Apply temporal confirmation; call Gemma only for flagged/ambiguous packet rows.
6. Read the reference manifest only now; evaluate all configurations, generate a selection report, retain all challenger artifacts.

This is mandatory: otherwise an input/crop/threshold change can be incorrectly credited to a model.

## 5. RTX 3090 runtime plan

### Resource rules

The 3090 has 24 GB VRAM. Heavy models must run sequentially, never all resident together:

1. person detection + tracker;
2. one pose configuration at a time;
3. one object configuration at a time;
4. Gemma after CV evidence generation.

Each configuration records GPU name, driver/CUDA/runtime versions, checkpoint SHA-256, input size, precision, batch size, thresholds, sampling rate, seed, wall time, median/p95 latency, peak allocated/reserved VRAM, host RAM, and disk use. Release model resources between phases. If a model cannot load while retaining a configured 2 GB VRAM safety margin, mark `resource_unavailable`; do not silently lower a resolution or replace it.

### Mandatory configuration families

| Family | Named configurations | Required role |
|---|---|---|
| Pose | `rtmpose_baseline`, `rtmo_baseline`, `alphapose_crowd_baseline`; `alphapose_wholebody` only if exact checkpoint passes preflight. | Pose challenger evidence, never verdicts. |
| Tracker | `tracker_primary`, `tracker_challenger`, `seat_only_no_track`. | Compare continuity; seat remains stable context. |
| Object | `rf_detr_native`, `dfine_native`, `rf_detr_sahi`, `dfine_sahi`. | Compare detector and slicing effects. |
| Crop | `seat_desk_context`, `lap_bag_context`, `pose_hand_context`, `native_source_context_fallback`. | Preserve source-scale context. |
| VLM | `gemma_marked_evidence`. | Observational review only. |

No configuration is called “best” until all runnable mandatory configurations have a matched report. A failed configuration stays in the comparison table as unavailable.

## 6. Video-health and preprocessing requirements

Source imagery is immutable. Every enabled preprocessing step stores raw input, transformed output, difference/diagnostic image, configuration ID, time/PTS, and metrics. Any transform is evaluated enabled **and** disabled on known references. Preprocessing must never replace source pixels in final evaluation.

### 6.1 Ingest and timestamp truth

- Decode the complete stream and derive timing from decoded PTS, not container FPS alone; support VFR.
- Record dimensions, codec, declared FPS, PTS span, frame count, keyframes, bitrate, colour metadata, rotation, errors, repeated PTS, and gaps.
- Produce regular contact sheets and contact sheets for every integrity incident.
- Each artifact retains source PTS/frame mapping. No generated image lacks source provenance.
- Undecodable spans are `decode_failure`; never interpolate an event through them.

### 6.2 Required health conditions

| Label | Detection evidence | Required action |
|---|---|---|
| `blackout` | Sustained near-black fraction/luma distribution. | Retain and mark unavailable; do not learn baseline. |
| `whiteout` | Sustained saturation/near-white evidence. | Retain; pause baseline adaptation. |
| `frozen_video` | Temporal image hash/SSIM repeats while PTS advances. | Retain; prevent false motion. |
| `decode_gap` | PTS/decode discontinuity. | Keep visible gap; do not bridge events. |
| `blur` | Sharpness relative to per-camera baseline. | Reduce object/pose reliability, preserve motion. |
| `exposure_change` | Scene-wide luma/histogram/chromaticity shift. | Flag recovery period; no baseline update. |
| `camera_motion` | Dominant global transform/motion field with reliable static-background support. | Compensate only on success; otherwise no local attribution. |
| `camera_repositioned` | Sustained new geometry / failed alignment. | New camera epoch, calibration review required. |
| `compression_noise` | Isolated low-area, nonpersistent residual activity. | Retain diagnostic, de-prioritise. |
| `environmental_motion` | Marked persistent region or periodic recurring pattern. | Retain named observation, penalise ranking. |

### 6.3 Alignment, shadows, lighting, noise

1. Build static-background support excluding seats, people, monitor changes, aisles, reflections, and environment masks.
2. Estimate global transform on static support using robust feature/flow fitting; ECC alignment may be a measured challenger only with convergence/quality score. Identity/no-compensation is a baseline.
3. Save raw, support mask, inlier/features, transform, confidence, compensated frame, difference image, and failure reason.
4. Compensate only if inlier coverage/residual/confidence pass immutable configuration thresholds. Otherwise label `camera_motion_uncertain` and retain raw evidence.
5. A sustained geometry change begins a new camera epoch and blocks seat attribution until calibration approval.
6. Main motion scoring combines residual magnitude, motion area ratio, directional coherence, persistence, and health state. Raw luma differencing alone is prohibited.
7. CLAHE, denoising, sharpening, or enhancement are named comparisons only; never unconditional full-video rewrites.
8. Baselines are per seat/zone, robust (median/MAD or equivalent), use relative and absolute spread floors, and require moving **area fraction** plus persistence. Absolute moving-block count alone is forbidden.
9. Baseline learning excludes quality-affected spans and records included/excluded windows. Rare activity must not be silently absorbed as normal.

## 7. Dense spatial calibration and attribution

### Principle

Tracker ID is not student identity in an occluded 30–60-person room. Anonymous seat/desk context is the stable identifier. A track supports a seat only while association is sufficiently clear.

### Calibration profile

Each camera-epoch profile contains:

- camera ID, dimensions, reference PTS/hash, operator, reviewer, version, approval status;
- anonymous seat IDs and reviewed masks for `seat_context`, `desk_work_surface`, `upper_body`, `lap_bag`, optional aisle/standing zones;
- monitor exclusion, environment/reflection, invigilator-path, and static-background masks;
- adjacency graph and explicit ambiguous boundaries;
- confidence notes and review images under crowded, sparse, changed-light, and aisle/reflection conditions.

Automatic polygon proposal may assist, but never approves calibration. Approval must reject zero-area, out-of-frame, unresolved overlap, missing background support, and unlabelled boundaries. Modifying a calibration creates a new version and requires reprocessing/reprojection; old events are never silently rewritten.

### Attribution procedure

1. Map raw and compensated motion to each intersecting zone.
2. Associate person detections/tracks using overlap, torso/bottom geometry, temporal continuity, and calibration constraints.
3. Associate pose only to the person crop that produced it.
4. Associate hand crops only to reliable wrist/pose source and candidate seats.
5. Assign `seat_id` only after configured evidence passes; otherwise output `ambiguous_seat` or `unattributed` and retain all candidate seat scores.

Never assign by nearest-box centre alone. Never force a seat across shared desk/aisle boundaries.

Required overlays: calibration masks, raw/residual motion masks, tracks and association lines/scores, pose confidence, source-projected crop rectangles.

## 8. Whole-video motion, ROI, segmentation, and non-destructive gates

### Motion/ROI features

Prefer codec motion vectors when present; validate with a pixel/optical-flow challenger on selected samples. Record availability. Use PTS-based windows.

Persist per seat/zone/window: raw/compensated motion distribution; area fraction; directional coherence; residual energy; desk/upper-body/lap contribution; neighbour/full-scene correlation; health flags; baseline median/spread/deviation/eligibility; periodicity; raw score/components.

ROI output is an uncertainty-aware set of active subregions **plus** broad seat context. Tiny object detection needs desk/hand context; a narrow motion blob alone is insufficient.

### Segmentation

Implement `QUIET → CANDIDATE → ACTIVE → COOLING → CLOSED`, with immutable start threshold, lower stop threshold, minimum duration, candidate bridge, merge gap, and maximum duration.

- Start requires residual motion and meaningful area fraction, not z-score alone.
- Keep short/subtle candidates even if they do not activate; mark lower priority.
- Bridge only actual intermittent activity. Rolling-max smoothing that invents duration from a spike is prohibited.
- Do not bridge across decode gap, blackout, uncertain camera movement, epoch, or calibration boundary.
- Split long events at sustained troughs, quality interruptions, epoch changes, or maximum duration; retain parent/child linkage.
- Store every transition reason, timestamp, scores, thresholds, and feature values.

### Gate protocol

Gates route evidence; they do not erase it.

| State | Meaning | Effect |
|---|---|---|
| `pass` | Adequate reliability. | Normal route/rank. |
| `deprioritise` | Weak or environmental but usable evidence. | Retain with penalty. |
| `uncertain` | Conflicting/insufficient condition classification. | Retain and expose for review/reprocess. |
| `suppress_from_auto_ranking` | Known integrity issue makes auto-rank unreliable. | Retain/searchable; excluded only from automatic priority. |
| `failed` | Data unavailable/corrupt. | Preserve gap; no claimed analysis. |

Each record has condition, PTS range, inputs/scores, thresholds/config ID, state, reason, artifacts. Gate audit must measure correct/false pass, correct/false suppression, uncertain retention, and failed coverage on references plus sampled normal/disturbance intervals.

## 9. Tracking and pose A/B requirements

Heavy pose cannot run on every person/frame. Full coverage comes from motion. For candidates, use configured low-rate person detection, labelled tracker interpolation, pose only on relevant/overlap people, and high-cost whole-body only on candidate frames/clips.

Freeze a `pose_manifest` with video hash, PTS/index, person-source/crop, seat candidates, health state, and deterministic ID. Run RTMPose, RTMO, AlphaPose against exactly the same rows.

For every configuration retain keypoints/confidences, full/crop overlay, persons proposed/returned, processing time, VRAM, and pose-seat state. Report by room region/crowding/health state: coverage, continuity, seat ambiguity, head/elbow/wrist visibility, reviewed joint swaps, throughput, resource use.

Pose may report only observable temporal context: baseline-relative head orientation/dwell, wrist entering desk/lap/bag context, repeated reach, torso/shoulder movement, cross-seat ambiguity. A single neck turn or low-confidence wrist must never create a high-priority event. Output must distinguish visible, occluded, not detected, and not resolvable at source scale.

## 10. Hand-object refinement and detector A/B

Hand-first is refinement, not primary detection. A hand may be hidden or mislocalised. Required sequence:

1. seat/motion event;
2. person/seat association;
3. pose wrist/elbow only when confidence supports it;
4. generous `pose_hand_context` crop containing hand, forearm direction, held-object area, desk/lap context;
5. if unreliable, use `seat_desk_context`, `lap_bag_context`, or `native_source_context_fallback`;
6. detector observations require temporal association before final evidence.

Every crop stores source hash/PTS/dimensions, source-coordinate rectangle, resize method, calibration, crop reason, seat candidates, source pose/track, and native pixel size. Upscaling/tile/VLM cannot invent detail; output `insufficient_visual_evidence` where resolution does not support the object.

### RF-DETR / D-FINE / SAHI matrix

Run `rf_detr_native`, `dfine_native`, `rf_detr_sahi`, `dfine_sahi` on identical crop manifests.

- Native: one explicit resize policy on original crop.
- SAHI: only on the same **candidate crop**, never unconditional 3×3/10×10 full-video grids; explicit slice/overlap/edge/threshold/NMS configuration.
- Persist crop, resized input, every slice, raw detections, merge/NMS output, source projection, config, time/VRAM/failure.
- Preserve raw and merged results. Do not let one detector overwrite another.

Provisional taxonomy requiring owner confirmation: `phone`, `secondary_paper_chit`, `answer_sheet`, `calculator`, `mouse`, `keyboard`, `hand`, `bag`, `monitor_or_bezel`, `stationery`, `unknown_object`, `insufficient_visual_evidence`. Hard negatives are mandatory; a detector calling mouse/keyboard/paper edge a phone is a failure cluster, not a mere false score.

Temporal confirmation groups class/geometry/seat/track/time. Store support-frame count, duration, spatial consistency, score distribution, crop provenance. Final status is `temporally_supported_object_evidence`, `single_frame_object_candidate`, `conflicting_object_evidence`, or `insufficient_visual_evidence`. A singleton is retained, not silently deleted.

## 11. Gemma flagged-evidence evaluation

Gemma is called only after flagged/ambiguous motion/pose/object evidence exists. It cannot create candidates, alter boundaries, alter rank, identify unmarked people, or state intent.

Each call must include a contact sheet of 4–8 chronological source frames with marked/labeled subject seat, magnified native subject crops, used hand/desk/lap crops, object boxes/config/scores, health state, and PTS on every tile. Whole-frame-only packets are prohibited.

The constrained JSON contains: `subject_marker_visible`, `evidence_quality` (`sufficient|limited|insufficient`), bounded `observable_actions`, `object_assessment`, `object_certainty` (`not_visible|possible|supported`), `quality_limitations`, `needs_human_review`, `reasoning_note`. It forbids verdict/identity/intent/unmarked-person language. Retain contact sheet, prompt version, raw output, parsed output, validation result, checkpoint hash, and latency. Schema-invalid output stays invalid, never repaired into evidence.

## 12. Ranking and reviewer output

Priority is transparent ranking, not cheating probability. It may combine residual motion/area, duration/repetition, seat association quality, pose context, temporally supported object evidence, quality penalty, environment/global penalty, and uncertainty penalty. Persist every component/coefficient/configuration ID.

An event package must include PTS interval; pre/post-roll clip; source overlays/crops; calibration/epoch; health/gate states; motion transitions; track/pose/object/VLM observations; configuration hashes; artifact pointers; human-review status. With zero high-priority events, report analysed/failed/affected duration, retained low-priority/uncertain counts, configs, and reference results—never “nothing happened.”

## 13. Common immutable run folder

```text
artifacts/runs/<run_id>/
├── RUN_MANIFEST.json
├── RUN_STATUS.json
├── source/{video_inventory.json,timestamp_index.jsonl,contact_sheets/}
├── 01_ingest/{decoded_frames_index.jsonl,anomalies/}
├── 02_health/{observations.jsonl,timeline.png,<pts>_<condition>/}
├── 03_alignment/<pts>/{raw.png,static_mask.png,features.png,compensated.png,metrics.json}
├── 04_calibration/{calibration_snapshot.json,overlays/,masks/}
├── 05_motion_roi/{window_metrics.jsonl,heatmaps/,overlays/,candidate_manifest.jsonl}
├── 06_segmentation/{event_state_transitions.jsonl,events.jsonl,timeline.png}
├── 07_tracking/<tracker_config>/{observations.jsonl,overlays/}
├── 08_pose/{pose_manifest.jsonl,rtmpose_baseline/,rtmo_baseline/,alphapose_crowd_baseline/,comparison.json}
├── 09_object/{object_crop_manifest.jsonl,rf_detr_native/,dfine_native/,rf_detr_sahi/,dfine_sahi/,comparison.json}
├── 10_evidence/{temporal_confirmation.jsonl,events/<event_id>/{clip.mp4,source_overlay.png,crops/,evidence.json},ranked_events.json}
├── 11_gemma/<event_id>/{contact_sheet.png,prompt.json,raw_response.txt,parsed.json,validation.json}
├── 12_evaluation/{reference_manifest_copy.jsonl,matches.jsonl,misses/,false_positives/,gate_audit.json,metrics.json}
├── 13_reports/{final_report.md,model_selection_report.md,quality_limitations.md,resource_report.json}
└── logs/{commands.log,stdout.log,stderr.log}
```

Artifacts use sortable PTS prefixes and native-source metadata. Completed runs are immutable; reprocessing creates a new run ID linked to parent.

## 14. Data contracts

### Reference manifest

JSONL/CSV must validate these fields:

| Field | Requirement |
|---|---|
| `reference_id` | Unique immutable ID. |
| `source_video_sha256` | Equals source hash. |
| `source_pts_ms` | Exact decoded timestamp. |
| `frame_index` | Optional, required when available; maps to PTS. |
| `class` | Approved mapped target class. |
| `bbox_xyxy_source` | Required when localisation is known; source pixels. |
| `seat_id` | Optional approved calibration seat ID. |
| `visibility` | `visible|partially_occluded|heavily_occluded|too_small|blurred|unknown`. |
| `annotation_source` | Independent identification provenance. |
| `notes` | Optional factual text. |

Invalid source hash/PTS/class/out-of-bounds box is written to `validation_errors.json`; evaluation is incomplete, not silently skipped.

### Event and observation records

Event records include source/camera epoch/calibration, PTS frames, parent split, seat candidate scores/state, health/gate state, motion values/thresholds, model observation IDs, priority components, artifact links, and review history. Model observations include model/config/checkpoint hash, input artifact hash, source projection, timestamp, output, score, warning/error, timing/VRAM, artifact pointer. One persistence action must not erase other configurations.

## 15. Evaluation protocol

### Isolation

Reference manifest becomes readable only in stage 12. Add an automated test that fails if a candidate/crop row includes `reference_id` or reference annotations.

### Matching

For each reference and configuration, match source hash and configured PTS tolerance; require class and source-coordinate overlap when reference box exists. Render source/crop/raw/merged/final images for every result. Classify `TP`, `FP`, `FN`, `ambiguous`, `insufficient_visual_evidence`, `quality_unavailable`, or `not_routed_to_candidate`. Attribute loss stage: motion/ROI, gate, crop, tracker/seat, pose/hand, detector, temporal confirmer, VLM disagreement.

### Required metrics

Per class and configuration report reference count, TP/FP/FN, precision/recall/F1, FP/hour, candidate-routing recall, recall conditional on routing, full-pipeline recall, abstention rate, quality breakdown, stage-loss waterfall, event/hour, candidate duration/video duration, rank of every reference, latency/VRAM/RAM/disk, and gate confusion measures. Phone and chit results must be separate. Zero detections must yield valid metrics rather than crash.

`model_selection_report.md` lists every configuration/availability/input/model metrics, failure clusters, runtime, license status, and an honest recommendation. No forced winner.

## 16. Ordered implementation files

Implementation must create or map these functions/files:

### Phase 0 — reproducibility/preflight

```text
README.md
requirements-rtx3090.lock or environment-rtx3090.yml
configs/models/*.yaml
configs/experiments/*.yaml
tools/preflight_rtx3090.py
pipeline/run_manifest.py
pipeline/artifacts.py
pipeline/config_validation.py
tests/test_preflight.py
```

Preflight checks launchability (not just weight existence), capacity, GPU/runtime, hashes, and license recording.

### Phase 1 — ingest and health

```text
pipeline/ingest.py
pipeline/timestamps.py
pipeline/health.py
tools/validate_video.py
tests/test_ingest_vfr.py
tests/test_health_conditions.py
```

### Phase 2 — calibration/alignment

```text
pipeline/calibration.py
pipeline/alignment.py
tools/calibrate.py
tools/review_calibration.py
tests/test_calibration_validation.py
tests/test_alignment.py
```

### Phase 3 — motion/ROI/segmentation/gates

```text
pipeline/motion.py
pipeline/roi.py
pipeline/baseline.py
pipeline/segmentation.py
pipeline/gates.py
tools/run_motion_scan.py
tests/test_motion_noise.py
tests/test_baseline_guards.py
tests/test_segmentation.py
tests/test_gate_audit.py
```

Test static zones, low-area compression noise, sustained subtle motion, impulse, exposure, vibration, environment periodicity, adjacent seats, maximum-duration split, and epoch break.

### Phase 4 — tracking/pose harness

```text
pipeline/person_detection.py
pipeline/tracking.py
pipeline/seat_association.py
pipeline/pose.py
tools/run_pose_ab.py
tools/render_pose_comparison.py
tests/test_seat_association.py
tests/test_pose_manifest_equivalence.py
```

### Phase 5 — crop/object/SAHI harness

```text
pipeline/crops.py
pipeline/object_detection.py
pipeline/sahi_adapter.py
pipeline/temporal_confirmation.py
tools/run_detector_ab.py
tools/render_detector_comparison.py
tests/test_crop_projection.py
tests/test_sahi_merge.py
tests/test_temporal_confirmation.py
```

### Phase 6 — evidence/Gemma/report

```text
pipeline/evidence.py
pipeline/gemma_review.py
pipeline/ranking.py
tools/render_evidence_packet.py
tools/generate_final_report.py
tests/test_gemma_packet_subject_marker.py
tests/test_gemma_schema.py
tests/test_observational_language.py
```

### Phase 7 — reference evaluation

```text
pipeline/reference_manifest.py
pipeline/evaluate.py
pipeline/gate_audit.py
tools/evaluate_run.py
tools/compare_configurations.py
tests/test_reference_isolation.py
tests/test_metrics_zero_detections.py
tests/test_evaluation_artifacts.py
```

## 17. Non-negotiable implementation rules

1. Full video first; reference frames only after inference.
2. Preserve original frames and before/after provenance for every transform.
3. Retain every gate result; no silent discard.
4. No single model is truth; all selection requires matched evidence.
5. Seat is context, not a forced tracker identity.
6. No universal head-angle rule; use baseline-relative temporal dwell and confidence.
7. Gemma is bounded evidence review, never candidate discovery/priority control.
8. Cropping/SAHI/VLM cannot invent native pixels; abstain explicitly.
9. Model configurations cannot overwrite one another’s artifacts/data.
10. Version and hash every threshold/input/slice/model/seed.
11. Baselines cannot learn through bad-quality intervals or silently absorb rare event activity.
12. Every reported item needs PTS-native overlay/crop provenance.
13. Integration tests may not silently skip missing models/videos as a pass.
14. No user-facing verdict vocabulary.

## 18. Acceptance criteria

- [ ] Complete decodable PTS duration is processed or every missing range is explicitly `failed`.
- [ ] Each run records source/config/calibration/model hashes, GPU/runtime metadata, resources, stage coverage, and artifacts.
- [ ] Health fixtures show retained labels for black/white/freeze/gap/blur/exposure/vibration/reposition/noise/environment motion.
- [ ] Gates have five-state records, visual artifacts, and false-pass/false-suppression audit.
- [ ] Invalid/unapproved calibration blocks seat attribution; boundary cases produce `ambiguous_seat`.
- [ ] Motion state transitions retain reasons and prevent spike-duration fabrication.
- [ ] Pose competitors receive identical manifest rows and persist independent outputs/availability metrics.
- [ ] Object competitors and SAHI variants receive identical crop rows; slicing is candidate-crop-only with source projection/merge provenance.
- [ ] Low-confidence wrists produce fallback crop/uncertainty rather than fabricated hand crops.
- [ ] Gemma evidence packets contain marked+magnified subject; whole-frame-only packet test fails.
- [ ] Candidate stage is demonstrably reference-manifest-isolated.
- [ ] Every reference row produces rendered TP/FP/FN/abstention/quality/not-routed output; phone/chit metrics are separate.
- [ ] Final report lists source limitations and a measured configuration recommendation without verdict language.

## 19. Research basis

- SAHI is included only as a measured crop-level comparison: slicing can aid small-object detection but cannot restore detail missing from the CCTV source. [SAHI paper](https://arxiv.org/abs/2202.06934)
- AlphaPose is a reasonable crowd/whole-body challenger because it supports CrowdPose and PoseFlow, but it has not proven dense far-hand accuracy on this exact hall; measure it. [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
- Global alignment is quality-gated because ECC/image registration can support alignment but may fail under poor texture/exposure/vibration; compare it to identity/no-compensation. [OpenCV ECC](https://docs.opencv.org/doc/doxygen/html/dc/d6b/group__video__track.html)

## 20. Definition of done

The work is complete only when the RTX 3090 owner can run one documented command against a full source video and obtain the immutable run folder, complete-duration accounting, artifacts for every preprocessing/model stage, A/B comparison reports, gate audit, reference-frame evaluation, and an honest recommendation.

The final report must choose one measured result: (1) a configuration has sufficient evidence quality to build on; (2) motion/ROI triage is useful but object detection is unreliable at this source scale; or (3) source quality/calibration/reference data prevents a claim and exact missing camera/data/training requirements are documented.
