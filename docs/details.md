# Project Classroom — Processing Details

Every stage, every threshold, every label, and the rules each model runs under.
**All values below are read from the code**, not from memory — see the config
dataclass named beside each table.

Companion to `docs/ARCHITECTURE.md`, which covers *why* the stack was chosen.
This document covers *what actually happens to a video*.

---

## 1. The environment this runs on

The reference frames show a **computer-based-test centre**, not a paper exam
hall: individual cubicles with frosted partitions, a monitor/keyboard/mouse at
every station, numbered seat placards on the wall, an oblique ceiling camera,
and a burnt-in timestamp in the corner. That fixes the threat model — phone
use, looking at a neighbour's screen, talking, leaving the seat, swapping seats
— and it is why chit detection is deprioritised (§9).

| Property | Measured |
|---|---|
| Resolution | 1280×720; **video 04 is 640×480** |
| Frame rate | 25 fps; **video 04 is 8 fps**; VFR on 4 of 6 files |
| Compression | ~0.08 bits/pixel |
| Median torso (shoulder) width | 64 px (v01) · 84 px (v03) · 35 px (paper) |
| Median interocular distance | 9.3 px (v01) · 10.1 px (v03) · 4.7 px (paper) |
| Phone footprint, near seat | ~30 × 20 px |

---

## 2. How a video is processed, stage by stage

### 2.1 Ingest — `pipeline/ingest.py`

One forward decode pass with PyAV, `flags2=+export_mvs` so codec motion
vectors come out with the frames. Every decoded frame's PTS is written to
`01_ingest/decoded_frames_index_<video>.jsonl`.

**Rule: PTS is the only clock.** Frame indices are never used for timing —
4 of 6 files are variable-frame-rate, so index × nominal-fps is wrong by a
growing amount. Every artifact filename is a zero-padded millisecond PTS, so
lexical sort equals temporal sort.

### 2.2 Preprocessing A/B — `pipeline/preprocessing.py` (PRD 6.3.7)

Four named transforms on 12 frames sampled across the recording. **Nothing
downstream consumes any of it.** Preprocessing may never replace source pixels
in evaluation, so this stage measures and stops.

| Config | What it does |
|---|---|
| `identity_no_preprocessing` | Named baseline. Not "no configuration". |
| `clahe` | Contrast-limited adaptive histogram equalisation |
| `denoise_nlmeans` | Non-local means |
| `unsharp_mask` | Unsharp masking |

Statistics per transform: `mean_luma`, `std_luma`, `sharpness` (Laplacian
variance), `blocking_ratio` (on-DCT-grid ÷ off-grid gradient energy),
`saturated_high`, `saturated_low`.

**Decision: measurement path is locked to `identity`.** Measured deltas:

| Transform | v01 Δsharp / Δblocking | v03 Δsharp / Δblocking |
|---|---|---|
| clahe | +203.18 / −0.0122 | +42.14 / **+0.0463** |
| denoise_nlmeans | −30.66 / −0.0091 | −9.18 / **−0.2414** |
| unsharp_mask | +878.47 / −0.0090 | +223.88 / **+0.2315** |

The two videos **disagree in sign**. On the more compressed source (v03,
baseline blocking 1.4632) CLAHE and unsharp raise blocking while raising
sharpness — at a 30×20 px object scale that is amplifying compression
structure, which is indistinguishable from amplifying the object. A transform
that helps the clean source and hurts the dirty one does not get locked in.
`clahe` is used **for the reviewer's display panel only**, labelled.

### 2.3 Health — `pipeline/health.py` (`HealthConfig`)

| Condition | Rule |
|---|---|
| `blackout` | ≥90% near-black for ≥500 ms |
| `whiteout` | ≥90% near-white for ≥500 ms |
| `frozen_video` | frame delta ≤0.005 for ≥1000 ms |
| `blur` | sharpness ≤0.4× local reference for ≥300 ms |
| `exposure_change` | luma step ≥8.0, recovery window 2000 ms |
| `compression_noise` | blocking over ≤5% area, ≤500 ms |
| `environmental_motion` | persistence ≥0.5 |

Health states never delete evidence. They attach to the event and become a
**multiplicative rank penalty** (§7).

### 2.4 Camera alignment — `pipeline/alignment.py` (`AlignmentConfig`)

Three named configurations fitted on **static background support only** —
pixels excluding seats, people, monitors, aisles and reflections. Fitting on
pixels containing people measures the people.

| Config | Method |
|---|---|
| `identity` | No compensation. A named baseline, not an absence. |
| `features_ransac` | ORB features + RANSAC |
| `ecc` | OpenCV ECC, 60 iterations, ε=1e-4, min correlation 0.90 |

**Gate before compensation:** inlier ratio ≥0.60, ≥12 inliers, residual ≤2.0 px,
support coverage ≥2%. A translation >120 px is rejected as implausible. Fail
the gate → the span is labelled `camera_motion_uncertain`, raw evidence is
retained, and rank is multiplied by 0.4.

**Epoch break:** ≥8 px translation sustained ≥2000 ms ⇒ a new camera epoch, and
seat attribution is blocked past it until calibration is re-approved.

Measured on v03 (18.4% static support): ECC 6/15 compensated at median residual
0.2318; features_ransac 0/15 at 0.2699; identity 0/15 at **6.5655**. Identity's
residual proves there is real motion to correct.

### 2.5 Motion scan — `pipeline/motion.py` (`MotionConfig`)

Codec motion vectors, whole video, **no neural model**, measured at 49–111×
realtime. 500 ms windows; a block counts as moving at magnitude ≥2.0.

**Global-motion gate:** if ≥55% of blocks move together at magnitude ≥1.5, the
window is global camera motion, not a person, and is compensated rather than
reported. Exposure steps ≥6.0 luma are handled the same way.

### 2.6 ROI — `pipeline/roi.py` (`RoiConfig`)

Moving blocks are linked (`link_distance=1`) into regions of ≥4 blocks, grown
by 3 context blocks. A region covering ≥60% of a seat zone is reported as the
whole zone rather than as a box, because a box around the whole zone implies a
localisation the data does not support.

### 2.7 Segmentation — `pipeline/segmentation.py` (`SegmentConfig`)

A **hysteresis state machine** per seat, in that seat's own z-units:

- **start** at z ≥ 4.0 with ≥5% zone area moving and ≥1 moving block
- **stop** at z ≤ 2.0
- minimum duration 1200 ms; sub-threshold gaps bridged up to 1500 ms; adjacent
  events merged across 2500 ms
- maximum duration 90 s, then a **trough split**: a dip to z ≤ 0.5 lasting
  ≥4000 ms splits one long event into two
- 3000 ms pre-roll and post-roll so a reviewer sees the lead-in

Two thresholds, not one, is what stops an event flickering on and off at a
single boundary.

---

## 3. Person detection and tracking

### 3.1 Person detection — `pipeline/person_detection.py`

D-FINE runs on the frame; each person box becomes one row. **The sample plan is
written before decoding** and `row_id` is keyed on `(video_sha, pts_ms)` — a
row is a *frame*, not an event, so a frame serving three events appears once
with three `event_ids`, not three times.

Rows are `DETECTED` or `INTERPOLATED`, never silently mixed.

### 3.2 Tracking — `pipeline/tracking.py`

Three configurations: `tracker_primary` (IoU + velocity), `tracker_challenger`
(ByteTrack), `seat_only_no_track` (the null baseline).

**Rule: reports "seat-inconsistent transitions", never "ID switches".** There
is no identity ground truth in this corpus, and a metric named for something
unmeasured is a fabricated metric.

### 3.3 Seat association — `pipeline/seat_association.py` (`AssociationConfig`)

Association is by **footprint**, never box centre — a person leaning across an
aisle has a centre over the neighbour's desk and feet under their own.
Per-track majority vote: ≥0.60 of ≥3 resolved boxes must agree.

**Continuity may resolve an ambiguity but never overrule a contradiction.**

**Current state: blocked.** 6 of 7 calibration profiles are `draft`, so seat
attribution is refused (PRD 3) and every row is `unattributed`.

---

## 4. Pose — what it predicts and what it emits

### 4.1 Models

Top-down: one forward pass **per person**, on frozen identical manifest rows.

| Config | Model |
|---|---|
| `alphapose_crowd_baseline` | **AlphaPose FastPose_DUC res152** — locked |
| `rtmpose_baseline` | RTMPose (challenger) |
| `rtmo_baseline` | RTMO, one-stage (challenger) |

### 4.2 Predictions

13 named COCO keypoints: `nose`, `left_eye`, `right_eye`, `left_ear`,
`right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`,
`left_wrist`, `right_wrist`, `left_hip`, `right_hip`.

### 4.3 Labels emitted — four observability states

| State | Meaning |
|---|---|
| `visible` | resolved above `keypoint_conf_min = 0.30` |
| `occluded` | something blocked it |
| `not_detected` | the model returned nothing |
| `not_resolvable_at_source_scale` | too few source pixels — the model was never entitled to answer |

The last two are deliberately distinct. Collapsing them would let a resolution
limit read as a model failure.

### 4.4 Rules

- **Person floor:** `min_person_px = 60.0`. A shorter box cannot support a wrist
  keypoint, so it is excluded rather than guessed.
- **Lap entry:** a wrist below the desk line by `lap_entry_margin_px = 8.0` sets
  `wrist_in_lap_context`.
- **Sustained requires support:** `min_support_frames = 2`. A single neck turn
  cannot set `sustained`, which is PRD 9's rule made structural.
- **Head dwell is baseline-relative:** `head_dwell_mad = 3.0` MADs above *that
  seat's own* median, never a universal angle. The camera looks down at the
  front row and across at the back.
- **Joint swaps are reported, never auto-corrected** (`suspected_joint_swaps`).

---

## 5. Head and neck — `pipeline/head_pose.py` (`HeadPoseConfig`)

### 5.1 What it predicts

Normalised **projective proxies**, monotonic in the true angle and comparable
within one subject over time. They are not protractor angles: recovering those
needs camera intrinsics and a face model, and at 4.7–10.1 px interocular this
corpus supports neither.

| Measure | Derivation |
|---|---|
| `yaw` | nose offset from ear midpoint ÷ half ear separation = `sin(yaw)` for a spherical head |
| `yaw` (one ear only) | saturated ±1.0; the occluded ear **is** the measurement |
| `pitch` | eye midpoint − ear midpoint, ÷ head scale |
| `roll` | eye-line tilt − shoulder-line tilt |
| `yaw_relative_to_torso` | yaw × shoulder squareness — separates neck turn from body rotation |

### 5.2 Labels emitted

`frontal` · `turned_left` · `turned_right` · `downward` ·
`not_resolvable_at_source_scale`

### 5.3 Rules

| Rule | Value | Why |
|---|---|---|
| Keypoint floor | 0.35 | AlphaPose median facial confidence is 0.84–0.93 |
| Head-scale floor | 6.0 px | below this the yaw denominator is jitter, not geometry |
| Turned threshold | 0.35 | ≈20° under the spherical assumption |
| Excursion threshold | median + 3.0 MAD | per subject, never universal |
| Minimum event | 1500 ms | a glance is not an event |
| Merge gap | **2.5 × the observed sample interval** | fixed 800 ms was shorter than the 1 Hz sample period, so every run split into singletons — measured, **1 event across 379 observations** |
| Repetition window | 60 000 ms | repeated turns the same way |

### 5.4 Temporal events

`look_down`, `yaw_left`, `yaw_right` — each with start, end, duration, peak,
frame count, and the baseline and threshold that produced it.

`neighbour_bias` asks whether a turn points at an **occupied adjacent seat**
rather than an aisle or a wall, and counts `repeated_same_direction`.

---

## 6. Objects — what each detector predicts

### 6.1 Crop selection — `pipeline/crops.py` (`CropConfig`)

Fallback chain, with the reason recorded on every crop:

`pose_hand_context` (radius = 1.0 × torso, wrist confidence ≥0.40)
→ `lap_bag_context` → `seat_desk_context` → `native_source_context_fallback`

**Resolution floor:** `abstain_below_native_px = 32`. A crop whose short side is
under 32 px is refused before the model runs, so a detector never sees a crop
it cannot answer for.

### 6.2 Detector configurations

| Config | Description |
|---|---|
| `dfine_native` | one resize of the crop |
| `dfine_sahi` | **2×2 slices, 0.25 overlap**, NMS IoU 0.45, full crop also included |
| `rf_detr_native` / `rf_detr_sahi` | challengers |

SAHI pads each interior side by `step × overlap ÷ 2`, and edge detection is
**one-sided** — a box crossing a boundary must be tested against that boundary,
not both.

### 6.3 Taxonomy emitted

`phone` · `secondary_paper_chit` · `answer_sheet` · `calculator` · `mouse` ·
`keyboard` · `hand` · `bag` · `monitor_or_bezel` · `stationery` ·
`book_or_notebook_like_object` · `unknown_object` ·
`insufficient_visual_evidence`

**Target classes** (the only ones that corroborate an event): `phone`,
`secondary_paper_chit`, `book_or_notebook_like_object`.
**Hard negatives** (tracked as confusions): `mouse`, `keyboard`, `calculator`,
`answer_sheet`, `monitor_or_bezel`, `stationery`.

### 6.4 Rules

- Score floor `0.25` — deliberately low, because PRD 10 wants singletons
  retained and judged by temporal confirmation, not thresholded away.
- **Object footprint floor `120 px²`.** An object whose own footprint is tiny is
  an abstention even inside a large crop.
- Runners return **source-frame coordinates**, never crop coordinates. A box
  left in crop space is wrong by the crop offset and still draws a
  plausible-looking overlay.

### 6.5 Temporal confirmation — `ConfirmConfig`

Group by class and position (IoU ≥0.10) across ≤3000 ms gaps; ≥2 supporting
frames and ≥0.60 class majority.

| Status | Meaning |
|---|---|
| `temporally_supported_object_evidence` | same object, several frames |
| `single_frame_object_candidate` | one frame — **retained**, a singleton is a status |
| `conflicting_object_evidence` | frames disagreed — reported, never resolved to majority |
| `insufficient_visual_evidence` | below the floor |

**Nothing is deleted.** `deleted: 0` is reported explicitly.

---

## 7. Ranking and dispositions — `pipeline/ranking.py`

Weights: `motion_deviation` 1.0 · `motion_area` 0.6 · `duration` 0.3 ·
`repetition` 0.5 · `seat_quality` 0.8 · `pose_sustained` 1.2 ·
**`object_supported` 2.0** · `object_single_frame` 0.4.

Penalties are **multiplicative**: `quality_affected` 0.6 · `environmental` 0.5 ·
`uncertain_seat` 0.7 · `camera_uncompensated` 0.4. A fixed subtraction would be
negligible on a strong event, which is exactly where an integrity problem
matters most.

**Only target-class evidence scores.** On v03, temporal confirmation produced
334 `unknown_object`, 103 `monitor_or_bezel` and 76 `keyboard` against 36
`phone`. Counting furniture put 34 of 46 events in the top band; filtering to
target classes dropped it to **1**.

| Disposition | Rule |
|---|---|
| `high_potential_corroborated` | target-class object confirmed **AND** sustained pose **AND** resolved seat **AND** no penalty |
| `flagged_for_review` | partial corroboration, or top 25% on motion alone |
| `retained_low_priority` | passed the gate, nothing corroborates it |
| `rejected_not_offered` | a gate withheld it — **never "cleared"** |

Bands cannot be absolute scores: the top event on v01 scores 85.63 and on v03
9.92, because the dominant term is deviation in that seat's own units.

---

## 8. Labels for the UI — `pipeline/labels.py`

**49 labels in 7 groups**, exported by `tools/export_labels.py` to
`artifacts/labels.json`, each with key, display name, group, severity, colour
and an observational description.

| Group | Count |
|---|---|
| object | 13 |
| head_event | 11 |
| health | 8 |
| facing | 5 |
| observability / disposition / object_status | 4 / 4 / 4 |

**`severity` orders reviewer attention. It is not a probability of misconduct**,
and nothing in this system estimates one. A UI must not render it as a guilt
score.

---

## 9. Paper, notebooks and chits — the honest state

**Detected correctly, not yet distinguished.** The detector's paper class fires
on real paper. Measured on v03, 200 instances: median linear extent 0.63
shoulder widths, median paper-like fraction 0.599, median +83.8 luma over the
surrounding desk. Rendering them showed **answer sheets, question papers and
printed packaging** — real paper, wrongly named "chit".

**The colour idea has not been validated.** Pooling 522 measured instances over
two cameras, the best single threshold on `luma_over_surround` reaches
**F1 0.682** (recall 0.756, precision 0.622 at +20). That is a useful *gate*
and not a classifier. It is recorded as measured, not adopted.

**A relative-size ruler works better than absolute pixels**: normalising by
shoulder width cut the cross-camera median ratio for `stationery` from 2.45×
to 1.10×. Absolute area cannot transfer — the same phone measures 485 px on v01
and 883 px on v03.

**Chit vs notebook remains unresolved**, and a classical paper-region scanner
built for it was withdrawn after it fired on shirt collars, chair highlights
and desk edges. Closing this needs labelled examples, which is a data problem,
not an algorithm problem.

---

## 10. The open questions, answered

### 10.1 Too many people in one frame

Person detection has **no cap**; each person becomes one pose row and top-down
pose costs **one forward pass per person**, so pose time grows *linearly* with
crowd size. Measured: v01 median 2 people/frame (max 5); the paper video 2081
rows across 12 seats.

The failure mode at 30+ people is cost, not correctness — and it is the one
case where **RTMO is the right model**, because one-stage inference is constant
in person count. That is why it is retained as an available configuration
rather than deleted.

Crowding also degrades IoU tracking (more ambiguous matches) and raises
occlusion, both of which show up as track fragmentation (§10.2).

### 10.2 Someone walks past and hides a candidate

Three things happen, and only two are handled well.

**Handled:** occluded keypoints are labelled `occluded`, not guessed. The
distinction from `not_detected` is preserved.

**Handled:** seat association votes over the *whole track* (≥0.60 majority of
≥3 boxes), so a short occlusion does not move a person to a different seat.

**Not handled:** the track breaks and a new id is issued. There is **no
re-identification**. Measured on v01: **29 tracks for ~2 people**, 10 lasting a
single frame, the longest covering 55% of frames. This is the single biggest
accuracy problem in the system today and it degrades every per-person metric
that depends on continuity.

### 10.3 Camera distortion

**Lens distortion is not corrected** — there are no camera intrinsics. The
practical consequence is small: seat polygons are drawn on the same distorted
image the detector sees, so attribution stays self-consistent *within one
camera*. It would matter for cross-camera geometry, which we do not do.

**Camera movement is handled** (§2.4): ECC compensation behind a quality gate,
with `camera_motion_uncertain` and a 0.4× penalty when the gate fails, and an
epoch break at ≥8 px sustained ≥2 s that blocks seat attribution past it.

### 10.4 Sudden flashes and lighting changes

Three independent defences:
1. **Health** labels `exposure_change` at a luma step ≥8.0 with a 2000 ms
   recovery window; `whiteout` at ≥90% near-white for ≥500 ms.
2. **Motion** compensates exposure steps ≥6.0 and applies the global-motion
   gate: ≥55% of blocks moving together at ≥1.5 is the scene, not a person.
3. **Ranking** multiplies the score by 0.6 for `quality_affected`.

A flash therefore raises no event on its own, and any event overlapping one
carries the flag into the reviewer's view.

### 10.5 The person exchanges seats

**Not detected today, and the reason is specific.** `associate_track` assigns
**one** seat per track by majority vote — so a track that genuinely changes
seats gets flattened to whichever seat it spent longer in, and the change is
hidden by the very mechanism that makes association robust.

The fix uses what already exists: per-box associations are already computed
before `_majority` collapses them. Emitting the *transition* — a persistent
change in the per-box seat sequence — gives the `seat_change` event, which is
already declared in `labels.py`. It also needs an **approved calibration**,
which is blocked on 6 of 7 cameras.

### 10.6 The person gets up and comes back — flag as "got up" or as cheating?

**As "got up". Never as cheating.** The labels are `left_seat` (medium) and
`returned_to_seat` (info), and the system does not assert misconduct anywhere —
PS #1 requires alerts that "highlight specific observed behaviors rather than
making direct accusations".

Standing up is ordinary: submitting a paper, a toilet break, an invigilator
calling someone over. What makes an absence worth a reviewer's time is
**duration** and **what co-occurs**: object evidence in the vacated seat, or
another track entering it. Under the disposition rules that combination reaches
`high_potential_corroborated` on its own merits; the absence alone does not.

### 10.7 Phone under the desk, not directly visible

If the desk physically occludes the phone, **no detector can see it and none
should claim to**. What the system does instead is report the behaviour:

- the crop chain falls back `pose_hand_context` → **`lap_bag_context`** →
  `seat_desk_context`, with `fallback_reason` recorded, so a hand below the
  desk still produces a crop;
- `wrist_in_lap_context` fires when a wrist crosses the desk line by ≥8 px;
- sustained lap dwell sets `pose_sustained`, which carries weight 1.2 into rank.

So the output is *"the candidate's hand was below the desk line for N seconds"*
— a reviewable observation — and not *"there is a phone"*. Any event where a
confirmed phone **and** sustained lap dwell coincide reaches the top band.

### 10.8 A mouse detected as a phone

This is the system's most common false positive and it is tracked explicitly
rather than hidden.

`mouse`, `keyboard`, `calculator`, `answer_sheet`, `monitor_or_bezel` and
`stationery` are declared **hard negatives**, and `failure_clusters()` groups
confusions by `(predicted, actual)` so `phone<-mouse` is counted as its own
number. Measured on v03 `dfine_sahi`: 199 mouse, 458 keyboard, 405 monitor
against 159 phone — in a CBT centre every desk has a mouse and a dark
rectangular monitor bezel, which is the exact shape a phone detector confuses.

Three mitigations are in place:
1. **Only target classes corroborate.** A confirmed mouse cannot carry an event
   into any band.
2. **Temporal confirmation** requires ≥2 frames and a 0.60 class majority, so a
   one-frame mouse→phone flip becomes `single_frame_object_candidate`, not
   supported evidence.
3. **RF-DETR as a second opinion**: it produces 53 hard negatives where D-FINE
   produces 735, so agreement between the two families is materially stronger
   than either alone.

The real fix is fine-tuning with these hard negatives — `tools/finetune_dfine.py`
and `NEGATIVE_CLASSES` exist for exactly this and need labelled data.

---

## 11. What is not measured, stated plainly

- **Eye gaze and iris direction.** Impossible at 4.7–10.1 px interocular; an eye
  is 2–3 px and there is no iris. Every published proctoring stack that does
  this is webcam-based with a face 10–20× larger.
- **Face identity.** Never computed, by design (PS #1 privacy requirement).
- **Chit vs notebook.** §9.
- **Seat changes, absences, re-identification.** §10.2, §10.5.
- **Any assertion that a candidate cheated.** Not a capability of this system.

---

## 12. Compute

| Component | Runtime |
|---|---|
| Motion / ROI / segmentation | CPU, 49–111× realtime, no neural model |
| D-FINE | `onnxruntime`, prefers `CUDAExecutionProvider` automatically |
| AlphaPose | torch, `--pose-device cuda`, falls back to CPU with a printed reason |
| SAHI | 4 inferences per crop — 0.24 crops/s on CPU vs 1.58 native |

Pose configurations can be restricted with `--pose-configs` once the A/B is
decided; omitted ones are recorded `not_selected` with the reason, never
silently dropped.
