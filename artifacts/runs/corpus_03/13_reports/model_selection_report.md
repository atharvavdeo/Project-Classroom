# Model selection report — run `corpus_03`

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

Manifest equivalence: **verified** (217 rows, digest `1a8b6e0ebc502fc6`)

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

Selection: none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

## Detector and slicing comparison

Crop equivalence: **verified** (217 crops, digest `ac7227b7dc3e6daa`)

| Configuration  | State     | Merged | Phone | Chit | Hard negatives |
|----------------|-----------|--------|-------|------|----------------|
| rf_detr_native | available | 617    | 10    | 0    | 53             |
| dfine_native   | available | 1648   | 108   | 0    | 735            |
| rf_detr_sahi   | available | 919    | 19    | 0    | 94             |
| dfine_sahi     | available | 2767   | 159   | 0    | 1165           |

### Failure clusters

PRD 10 treats a phone called on a mouse, keyboard or paper edge as a failure cluster to fix, not a false positive to average away. Without reference labels the class mix below is the shape of the output, not its accuracy.

**rf_detr_native** — unlabelled: {'bag': 16, 'keyboard': 4, 'monitor_or_bezel': 21, 'mouse': 25, 'phone': 10, 'stationery': 3, 'unknown_object': 538}

**dfine_native** — unlabelled: {'bag': 66, 'keyboard': 151, 'monitor_or_bezel': 303, 'mouse': 200, 'phone': 108, 'stationery': 81, 'unknown_object': 739}

**rf_detr_sahi** — unlabelled: {'bag': 19, 'keyboard': 7, 'monitor_or_bezel': 47, 'mouse': 23, 'phone': 19, 'stationery': 17, 'unknown_object': 787}

**dfine_sahi** — unlabelled: {'bag': 82, 'keyboard': 458, 'monitor_or_bezel': 405, 'mouse': 199, 'phone': 159, 'stationery': 103, 'unknown_object': 1361}

Selection: none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 0      |
| single_frame_object_candidate        | 2818   |
| temporally_supported_object_evidence | 1122   |

2818 singletons retained, 0 deleted. no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Recommendation

No forced winner. See the evaluation metrics in `12_evaluation/`.
