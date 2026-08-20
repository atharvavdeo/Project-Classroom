# Model selection report — run `1446/12_paper`

PRD 5: no configuration is called best until every runnable mandatory configuration has a matched report. PRD 15 places selection at stage 12, against the reference manifest.

## Configuration availability

| Configuration            | Family           | State           | Licence                     | Reason                                                       |
|--------------------------|------------------|-----------------|-----------------------------|--------------------------------------------------------------|
| person_detection_primary | person_detection | available       | Apache-2.0                  |                                                              |
| tracker_primary          | tracker          | available       | MIT                         |                                                              |
| seat_only_no_track       | tracker          | available       | n/a-internal                |                                                              |
| rtmpose_baseline         | pose             | available       | Apache-2.0                  |                                                              |
| rtmo_baseline            | pose             | available       | Apache-2.0                  |                                                              |
| rtmo_large               | pose             | available       | Apache-2.0                  |                                                              |
| alphapose_crowd_baseline | pose             | available       | SJTU-NonCommercial-Research |                                                              |
| dfine_native             | object           | available       | Apache-2.0                  |                                                              |
| rf_detr_native           | object           | available       | Apache-2.0                  |                                                              |
| dfine_sahi               | object           | available       | Apache-2.0                  |                                                              |
| rf_detr_sahi             | object           | available       | Apache-2.0                  |                                                              |
| gemma_marked_evidence    | vlm              | missing_weights | Gemma-Terms-of-Use          | checkpoint not found: C:\DrishtiAI\Project-Classroom\models\ |

## Pose comparison

Manifest equivalence: **verified** (5881 rows, digest `3df16113796accf0`)

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

Selection: none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

## Detector and slicing comparison

Crop equivalence: **verified** (5881 crops, digest `654ca2ddec3aa28a`)

| Configuration  | State                | Merged | Phone | Chit | Hard negatives |
|----------------|----------------------|--------|-------|------|----------------|
| rf_detr_native | resource_unavailable | —      | —     | —    | —              |
| dfine_native   | available            | 32203  | 1648  | 0    | 14262          |
| rf_detr_sahi   | resource_unavailable | —      | —     | —    | —              |
| dfine_sahi     | available            | 47888  | 2910  | 0    | 15577          |

### Failure clusters

PRD 10 treats a phone called on a mouse, keyboard or paper edge as a failure cluster to fix, not a false positive to average away. Without reference labels the class mix below is the shape of the output, not its accuracy.

**dfine_native** — unlabelled: {'bag': 2495, 'book_or_notebook_like_object': 1093, 'insufficient_visual_evidence': 233, 'keyboard': 2611, 'monitor_or_bezel': 9784, 'mouse': 1758, 'phone': 1648, 'stationery': 109, 'unknown_object': 12472}

**dfine_sahi** — unlabelled: {'bag': 4695, 'book_or_notebook_like_object': 3092, 'insufficient_visual_evidence': 233, 'keyboard': 3438, 'monitor_or_bezel': 9864, 'mouse': 1925, 'phone': 2910, 'stationery': 350, 'unknown_object': 21381}

Selection: none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 38     |
| recurrent_scene_object               | 33417  |
| single_frame_object_candidate        | 4563   |
| temporally_supported_object_evidence | 1601   |

4563 singletons retained, 0 deleted. no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Recommendation

No forced winner. See the evaluation metrics in `12_evaluation/`.
