# Model selection report — run `1446/04_talking`

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

Manifest equivalence: **verified** (3542 rows, digest `fd5679cc30d38d47`)

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

Selection: none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

## Detector and slicing comparison

Crop equivalence: **verified** (3542 crops, digest `9a30aca9fbbd5aef`)

| Configuration  | State                | Merged | Phone | Chit | Hard negatives |
|----------------|----------------------|--------|-------|------|----------------|
| rf_detr_native | resource_unavailable | —      | —     | —    | —              |
| dfine_native   | available            | 24597  | 1571  | 0    | 11957          |
| rf_detr_sahi   | resource_unavailable | —      | —     | —    | —              |
| dfine_sahi     | available            | 40157  | 4188  | 0    | 18039          |

### Failure clusters

PRD 10 treats a phone called on a mouse, keyboard or paper edge as a failure cluster to fix, not a false positive to average away. Without reference labels the class mix below is the shape of the output, not its accuracy.

**dfine_native** — unlabelled: {'bag': 1402, 'book_or_notebook_like_object': 176, 'insufficient_visual_evidence': 164, 'keyboard': 4599, 'monitor_or_bezel': 5474, 'mouse': 1572, 'phone': 1571, 'stationery': 312, 'unknown_object': 9327}

**dfine_sahi** — unlabelled: {'bag': 3280, 'book_or_notebook_like_object': 257, 'insufficient_visual_evidence': 164, 'keyboard': 6930, 'monitor_or_bezel': 7177, 'mouse': 2778, 'phone': 4188, 'stationery': 1154, 'unknown_object': 14229}

Selection: none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 40     |
| recurrent_scene_object               | 35131  |
| single_frame_object_candidate        | 916    |
| temporally_supported_object_evidence | 245    |

916 singletons retained, 0 deleted. no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Recommendation

No forced winner. See the evaluation metrics in `12_evaluation/`.
