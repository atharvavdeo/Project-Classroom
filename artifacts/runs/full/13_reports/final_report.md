# Final report — run `full`

## Measured result (PRD 20)

**(3) Source quality prevents a claim.** No configuration produced a single true positive against 3 reference row(s), at a best precision of 0.000, and motion routed none of them to a candidate, so the loss is not attributable to any single stage.

## What this run surfaced

23 event(s) at normal priority over 282.1 s analysed; 0 retained at low priority and 0 excluded from automatic ordering. Priority is a reading order, not a probability that anything occurred.

| Measure                          | Value   |
|----------------------------------|---------|
| Events ranked                    | 23      |
| Normal priority                  | 23      |
| Retained at low priority         | 0       |
| Excluded from automatic ordering | 0       |
| Need human review                | 23      |
| Analysed                         | 282.1 s |
| Failed                           | 0.0 s   |
| Quality affected                 | 0.0 s   |

Priority is a reading order over what this configuration surfaced. It is not a probability that anything occurred, and no row here is a finding about any person.

## Stage coverage

| Stage           | State               | Reason                                                      |
|-----------------|---------------------|-------------------------------------------------------------|
| source          | pending             |                                                             |
| 01_ingest       | pending             |                                                             |
| 02_health       | pending             |                                                             |
| 03_alignment    | pending             |                                                             |
| 04_calibration  | pending             |                                                             |
| 05_motion_roi   | complete            |                                                             |
| 06_segmentation | complete            |                                                             |
| 07_tracking     | complete            |                                                             |
| 08_pose         | complete            |                                                             |
| 09_object       | complete            |                                                             |
| 10_evidence     | complete            |                                                             |
| 11_gemma        | skipped_unavailable | no VLM review was performed; every event needs human review |
| 12_evaluation   | complete            |                                                             |
| 13_reports      | pending             |                                                             |

## Source limitations

- **0 of 416 crops** (0.0%) fall below the native-resolution floor. No detector can answer for these.
- Median crop magnification **2.95x**, maximum **13.33x**. Magnification is interpolation, not detail.
- Crop policy mix: {'lap_bag_context': 137, 'native_source_context_fallback': 43, 'pose_hand_context': 236}.

- Crops fell back from `pose_hand_context` for: {'no_pose_observation': 139, 'wrist_confidence_below_floor': 41}.

## Evidence packets

23 of 23 packets meet PRD 11's composition requirements. an invalid packet is never sent to a model. PRD 11 prohibits whole-frame-only packets because a frame showing thirty candidates and marking none invites a confident description of the wrong person.

## Bounded VLM review

| Measure                       | Value |
|-------------------------------|-------|
| Reviews                       | 23    |
| Attempted                     | 0     |
| Not attempted                 | 23    |
| Schema valid                  | 0     |
| Schema invalid                | 0     |
| Repaired                      | 0     |
| Packets blocked before review | 0     |
| Need human review             | 23    |

no invalid response was repaired. A JSON object assembled by the pipeline from a malformed response is not the model's assessment, and nothing downstream could tell (PRD 11).

## What is missing

1. **Product-owner approval of the taxonomy.** PRD 3 blocks detector selection and evaluation until the target and hard-negative class list is approved.
2. **rf_detr_native** — rfdetr is not installed in this environment, so RF-DETR did not run. It stays in the comparison as unavailable and is never substituted by D-FINE (PRD 5).
3. **rf_detr_sahi** — rfdetr is not installed in this environment, so RF-DETR did not run. It stays in the comparison as unavailable and is never substituted by D-FINE (PRD 5).

## Scope of any claim

This system produces observational evidence for human review. It is not a cheating classifier, it does not infer intent, and no output here is a verdict about any person. The permitted vocabulary is observational throughout: `phone-like object detected`, `hand entered lap zone`, `ambiguous_seat`, `insufficient_visual_evidence`.

The validated corpus is 1-17 people. The 30-60 person figure in the PRD is a capacity and design target, not a result this corpus supports.
