# Model selection report — run `corpus_01`

PRD 5: no configuration is called best until every runnable mandatory configuration has a matched report. PRD 15 places selection at stage 12, against the reference manifest.

## Configuration availability

| Configuration            | Family           | State     | Licence                     | Reason |
|--------------------------|------------------|-----------|-----------------------------|--------|
| person_detection_primary | person_detection | available | Apache-2.0                  |        |
| tracker_primary          | tracker          | available | MIT                         |        |
| seat_only_no_track       | tracker          | available | n/a-internal                |        |
| rtmpose_baseline         | pose             | available | Apache-2.0                  |        |
| rtmo_baseline            | pose             | available | Apache-2.0                  |        |
| rtmo_large               | pose             | available | Apache-2.0                  |        |
| alphapose_crowd_baseline | pose             | available | SJTU-NonCommercial-Research |        |
| dfine_native             | object           | available | Apache-2.0                  |        |
| rf_detr_native           | object           | available | Apache-2.0                  |        |
| dfine_sahi               | object           | available | Apache-2.0                  |        |
| rf_detr_sahi             | object           | available | Apache-2.0                  |        |
| gemma_marked_evidence    | vlm              | available | Gemma-Terms-of-Use          |        |

## Pose comparison

Manifest equivalence: **verified** (379 rows, digest `67489132465ccb8b`)

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

Selection: none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

## Detector and slicing comparison

Crop equivalence: **verified** (379 crops, digest `165261f983377ae1`)

| Configuration  | State     | Merged | Phone | Chit | Hard negatives |
|----------------|-----------|--------|-------|------|----------------|
| rf_detr_native | available | 594    | 11    | 0    | 111            |
| dfine_native   | available | 2325   | 92    | 0    | 206            |
| rf_detr_sahi   | available | 1285   | 34    | 0    | 243            |
| dfine_sahi     | available | 3709   | 158   | 0    | 452            |

### Failure clusters

PRD 10 treats a phone called on a mouse, keyboard or paper edge as a failure cluster to fix, not a false positive to average away. Without reference labels the class mix below is the shape of the output, not its accuracy.

**rf_detr_native** — unlabelled: {'bag': 55, 'insufficient_visual_evidence': 4, 'keyboard': 4, 'monitor_or_bezel': 66, 'mouse': 19, 'phone': 11, 'stationery': 22, 'unknown_object': 413}

**dfine_native** — unlabelled: {'bag': 151, 'insufficient_visual_evidence': 4, 'keyboard': 20, 'monitor_or_bezel': 107, 'mouse': 40, 'phone': 92, 'stationery': 39, 'unknown_object': 1872}

**rf_detr_sahi** — unlabelled: {'bag': 99, 'insufficient_visual_evidence': 4, 'keyboard': 15, 'monitor_or_bezel': 154, 'mouse': 28, 'phone': 34, 'stationery': 46, 'unknown_object': 905}

**dfine_sahi** — unlabelled: {'bag': 231, 'insufficient_visual_evidence': 4, 'keyboard': 45, 'monitor_or_bezel': 180, 'mouse': 81, 'phone': 158, 'stationery': 146, 'unknown_object': 2864}

Selection: none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 12     |
| single_frame_object_candidate        | 3822   |
| temporally_supported_object_evidence | 1522   |

3822 singletons retained, 0 deleted. no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Recommendation

No forced winner. See the evaluation metrics in `12_evaluation/`.
