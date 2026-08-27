# Model selection report — run `full`

PRD 5: no configuration is called best until every runnable mandatory configuration has a matched report. PRD 15 places selection at stage 12, against the reference manifest.

## Configuration availability

| Configuration            | Family           | State           | Licence            | Reason                                                       |
|--------------------------|------------------|-----------------|--------------------|--------------------------------------------------------------|
| person_detection_primary | person_detection | available       | Apache-2.0         |                                                              |
| tracker_primary          | tracker          | available       | MIT                |                                                              |
| seat_only_no_track       | tracker          | available       | n/a-internal       |                                                              |
| rtmpose_baseline         | pose             | available       | Apache-2.0         |                                                              |
| rtmo_baseline            | pose             | available       | Apache-2.0         |                                                              |
| rtmo_large               | pose             | available       | Apache-2.0         |                                                              |
| alphapose_crowd_baseline | pose             | licence_blocked | UNCONFIRMED        | licence is unconfirmed; PRD 3 requires licence status record |
| dfine_native             | object           | available       | Apache-2.0         |                                                              |
| rf_detr_native           | object           | launch_failed   | Apache-2.0         | import rfdetr failed: ModuleNotFoundError: No module named ' |
| dfine_sahi               | object           | available       | Apache-2.0         |                                                              |
| rf_detr_sahi             | object           | launch_failed   | Apache-2.0         | import rfdetr failed: ModuleNotFoundError: No module named ' |
| gemma_marked_evidence    | vlm              | available       | Gemma-Terms-of-Use |                                                              |

## Pose comparison

Manifest equivalence: **verified** (416 rows, digest `e740670cced453b2`)

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

Selection: none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

## Detector and slicing comparison

Crop equivalence: **verified** (416 crops, digest `4aa2ee325248e243`)

| Configuration  | State                | Merged | Phone | Chit | Hard negatives |
|----------------|----------------------|--------|-------|------|----------------|
| rf_detr_native | resource_unavailable | —      | —     | —    | —              |
| dfine_native   | available            | 3179   | 225   | 232  | 1420           |
| rf_detr_sahi   | resource_unavailable | —      | —     | —    | —              |
| dfine_sahi     | available            | 5432   | 318   | 136  | 2245           |

### Failure clusters

PRD 10 treats a phone called on a mouse, keyboard or paper edge as a failure cluster to fix, not a false positive to average away. Without reference labels the class mix below is the shape of the output, not its accuracy.

**dfine_native** — unlabelled: {'bag': 115, 'keyboard': 290, 'monitor_or_bezel': 580, 'mouse': 385, 'phone': 225, 'secondary_paper_chit': 232, 'stationery': 165, 'unknown_object': 1187}

**dfine_sahi** — unlabelled: {'bag': 151, 'keyboard': 877, 'monitor_or_bezel': 783, 'mouse': 381, 'phone': 318, 'secondary_paper_chit': 136, 'stationery': 204, 'unknown_object': 2582}

Selection: none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 0      |
| single_frame_object_candidate        | 3482   |
| temporally_supported_object_evidence | 1619   |

3482 singletons retained, 0 deleted. no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Recommendation

No forced winner. See the evaluation metrics in `12_evaluation/`.
