# Final report — run `1446/04_talking`

## Measured result (PRD 20)

**(3) Source quality, calibration or reference data prevents a claim.** Specifically, reference data: this run completed motion, ROI, segmentation, tracking, pose and object inference over the full decodable duration and produced auditable artifacts for each, but no reference manifest was evaluated, so no accuracy statement of any kind is supportable. This is a statement about what is missing, not about what the pipeline found.

## What this run surfaced

38 event(s) at normal priority over 143.4 s analysed; 0 retained at low priority and 0 excluded from automatic ordering. Priority is a reading order, not a probability that anything occurred.

| Measure                          | Value   |
|----------------------------------|---------|
| Events ranked                    | 38      |
| Normal priority                  | 38      |
| Retained at low priority         | 0       |
| Excluded from automatic ordering | 0       |
| Need human review                | 38      |
| Analysed                         | 143.4 s |
| Failed                           | 0.0 s   |
| Quality affected                 | 0.0 s   |

Priority is a reading order over what this configuration surfaced. It is not a probability that anything occurred, and no row here is a finding about any person.

## Stage coverage

| Stage           | State               | Reason                                                                 |
|-----------------|---------------------|------------------------------------------------------------------------|
| source          | complete            |                                                                        |
| 01_ingest       | complete            | one complete forward decode pass; motion vectors present               |
| 02_health       | complete            |                                                                        |
| 03_alignment    | skipped_unavailable | no camera_motion or camera_repositioned condition was observed, so no  |
| 04_calibration  | complete            | 04/e1/v1 (draft)                                                       |
| 05_motion_roi   | complete            |                                                                        |
| 06_segmentation | complete            |                                                                        |
| 07_tracking     | complete            |                                                                        |
| 08_pose         | complete            |                                                                        |
| 09_object       | complete            |                                                                        |
| 10_evidence     | complete            |                                                                        |
| 11_gemma        | skipped_unavailable | no VLM review was performed; every event needs human review            |
| 12_evaluation   | blocked             | no reference manifest; evaluation incomplete and no quality claim may  |
| 13_reports      | pending             |                                                                        |

## Source limitations

- **164 of 3542 crops** (4.6%) fall below the native-resolution floor. No detector can answer for these.
- Median crop magnification **5.28x**, maximum **15.42x**. Magnification is interpolation, not detail.
- Crop policy mix: {'native_source_context_fallback': 1567, 'pose_hand_context': 1975}.

- Crops fell back from `pose_hand_context` for: {'no_pose_observation': 660, 'wrist_confidence_below_floor': 896, 'wrist_not_resolvable_at_source_scale': 11}.

## Evidence packets

38 of 38 packets meet PRD 11's composition requirements. an invalid packet is never sent to a model. PRD 11 prohibits whole-frame-only packets because a frame showing thirty candidates and marking none invites a confident description of the wrong person.

## Bounded VLM review

| Measure                       | Value |
|-------------------------------|-------|
| Reviews                       | 38    |
| Attempted                     | 0     |
| Not attempted                 | 38    |
| Schema valid                  | 0     |
| Schema invalid                | 0     |
| Repaired                      | 0     |
| Packets blocked before review | 0     |
| Need human review             | 38    |

no invalid response was repaired. A JSON object assembled by the pipeline from a malformed response is not the model's assessment, and nothing downstream could tell (PRD 11).

## What is missing

1. **Product-owner approval of the taxonomy.** PRD 3 blocks detector selection and evaluation until the target and hard-negative class list is approved.
2. **rf_detr_native** — skipped by --no-rfdetr; it stays in the comparison as unavailable and is never substituted by D-FINE (PRD 5).
3. **rf_detr_sahi** — skipped by --no-rfdetr; it stays in the comparison as unavailable and is never substituted by D-FINE (PRD 5).

## Scope of any claim

This system produces observational evidence for human review. It is not a cheating classifier, it does not infer intent, and no output here is a verdict about any person. The permitted vocabulary is observational throughout: `phone-like object detected`, `hand entered lap zone`, `ambiguous_seat`, `insufficient_visual_evidence`.

The validated corpus is 1-17 people. The 30-60 person figure in the PRD is a capacity and design target, not a result this corpus supports.
