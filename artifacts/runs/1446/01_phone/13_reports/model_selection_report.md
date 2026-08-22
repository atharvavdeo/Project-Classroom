# Model selection report — run `1446/01_phone`

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

Manifest equivalence: **verified** (1076 rows, digest `c1b2f98d2bb45bb5`)

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

Selection: none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

## Detector and slicing comparison

Crop equivalence: **verified** (1076 crops, digest `5e2aea52b3c598b6`)

| Configuration  | State                | Merged | Phone | Chit | Hard negatives |
|----------------|----------------------|--------|-------|------|----------------|
| rf_detr_native | resource_unavailable | —      | —     | —    | —              |
| dfine_native   | available            | 6529   | 237   | 0    | 597            |
| rf_detr_sahi   | resource_unavailable | —      | —     | —    | —              |
| dfine_sahi     | available            | 10339  | 416   | 0    | 1236           |

### Failure clusters

PRD 10 treats a phone called on a mouse, keyboard or paper edge as a failure cluster to fix, not a false positive to average away. Without reference labels the class mix below is the shape of the output, not its accuracy.

**dfine_native** — unlabelled: {'bag': 439, 'book_or_notebook_like_object': 75, 'insufficient_visual_evidence': 7, 'keyboard': 68, 'monitor_or_bezel': 272, 'mouse': 122, 'phone': 237, 'stationery': 135, 'unknown_object': 5174}

**dfine_sahi** — unlabelled: {'bag': 656, 'book_or_notebook_like_object': 111, 'insufficient_visual_evidence': 7, 'keyboard': 116, 'monitor_or_bezel': 510, 'mouse': 232, 'phone': 416, 'stationery': 378, 'unknown_object': 7913}

Selection: none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 6      |
| single_frame_object_candidate        | 7506   |
| temporally_supported_object_evidence | 3147   |

7506 singletons retained, 0 deleted. no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Recommendation

No forced winner. See the evaluation metrics in `12_evaluation/`.
