# Configuration comparison — run `full`

**Reference isolation: verified** across 7 pre-evaluation stages.

PRD 15 makes the reference manifest readable only at stage 12. An evaluation of a pipeline that already saw the answers measures nothing.

## References

| Measure        | Value |
|----------------|-------|
| Accepted rows  | 3     |
| Rejected rows  | 1     |
| Complete       | False |
| Localised rows | 2     |

1 row(s) could not be validated. Every metric derived from this manifest is INCOMPLETE; the rejected rows were recorded, not skipped (PRD 14).

## Overall, per configuration

An undefined ratio prints as `—`. A precision of 0.000 means the detector was wrong; `—` means it was never asked.

| Configuration | TP | FP  | FN | Not routed | Insufficient | Quality unavail. | Precision | Recall | F1    | FP/hour  |
|---------------|----|-----|----|------------|--------------|------------------|-----------|--------|-------|----------|
| dfine_native  | 0  | 449 | 0  | 3          | 0            | 0                | 0.000     | 0.000  | 0.000 | 5729.883 |
| dfine_sahi    | 0  | 443 | 0  | 3          | 0            | 0                | 0.000     | 0.000  | 0.000 | 5653.314 |

## Phone and chit, separately (PRD 15)

### `phone`

| Configuration | Refs | TP | FP  | FN | Precision | Recall | Routing recall | Recall | routed | Full-pipeline recall |
|---------------|------|----|-----|----|-----------|--------|----------------|-----------------|----------------------|
| dfine_native  | 3    | 0  | 217 | 0  | 0.000     | 0.000  | 0.000          | —               | 0.000                |
| dfine_sahi    | 3    | 0  | 307 | 0  | 0.000     | 0.000  | 0.000          | —               | 0.000                |

### `secondary_paper_chit`

| Configuration | Refs | TP | FP  | FN | Precision | Recall | Routing recall | Recall | routed | Full-pipeline recall |
|---------------|------|----|-----|----|-----------|--------|----------------|-----------------|----------------------|
| dfine_native  | 0    | 0  | 232 | 0  | 0.000     | —      | —              | —               | —                    |
| dfine_sahi    | 0    | 0  | 136 | 0  | 0.000     | —      | —              | —               | —                    |

**Routing recall** is the share of references motion routed to a candidate at all. **Recall | routed** is what the rest of the pipeline then found among those. Quoting only one of these is how a pipeline gets optimised in the wrong direction: a recall figure that blames the detector for the motion gate leads directly to loosening the gate until everything is a candidate.

## Stage-loss waterfall

The earliest stage that could have carried a reference and did not. A reference lost at `motion_roi` was never shown to a detector, so it is not a detector error.

| Configuration | motion_roi | gate | crop | tracker_seat | pose_hand | detector | temporal_confirmer | vlm_disagreement |
|---------------|------------|------|------|--------------|-----------|----------|--------------------|------------------|
| dfine_native  | 3          | 0    | 0    | 0            | 0         | 0        | 0                  | 0                |
| dfine_sahi    | 3          | 0    | 0    | 0            | 0         | 0        | 0                  | 0                |

## Outcome breakdown by visibility

### `dfine_native`

| Visibility         | TP | FP | FN | ambiguous | insufficient_visual_evidence | quality_unavailable | not_routed_to_candidate |
|--------------------|----|----|----|-----------|------------------------------|---------------------|-------------------------|
| heavily_occluded   | 0  | 0  | 0  | 0         | 0                            | 0                   | 1                       |
| partially_occluded | 0  | 0  | 0  | 0         | 0                            | 0                   | 1                       |
| too_small          | 0  | 0  | 0  | 0         | 0                            | 0                   | 1                       |

### `dfine_sahi`

| Visibility         | TP | FP | FN | ambiguous | insufficient_visual_evidence | quality_unavailable | not_routed_to_candidate |
|--------------------|----|----|----|-----------|------------------------------|---------------------|-------------------------|
| heavily_occluded   | 0  | 0  | 0  | 0         | 0                            | 0                   | 1                       |
| partially_occluded | 0  | 0  | 0  | 0         | 0                            | 0                   | 1                       |
| too_small          | 0  | 0  | 0  | 0         | 0                            | 0                   | 1                       |

## Gate audit

audited on references plus sampled normal and disturbance intervals (PRD 8). References alone cannot expose a gate that passes everything: on that population it scores a perfect false-suppression rate.

> **only 0 of 40 normal intervals could be drawn in 2000 attempts. Disturbance spans and reference neighbourhoods cover most of this recording, so there is little undisturbed footage to sample. The gate's behaviour on quiet footage is therefore UNMEASURED here, not measured as good.**

| Population          | Intervals | Relevant | Correct pass | False suppression | False pass | Correct suppression | False-supp. rate |
|---------------------|-----------|----------|--------------|-------------------|------------|---------------------|------------------|
| sampled_disturbance | 20        | 2        | 2            | 0                 | 18         | 0                   | 0.000            |
| reference           | 3         | 3        | 3            | 0                 | 0          | 0                   | 0.000            |

## Configuration availability

| Configuration            | State           | Reason                                                                 |
|--------------------------|-----------------|------------------------------------------------------------------------|
| person_detection_primary | available       |                                                                        |
| tracker_primary          | available       |                                                                        |
| seat_only_no_track       | available       |                                                                        |
| rtmpose_baseline         | available       |                                                                        |
| rtmo_baseline            | available       |                                                                        |
| rtmo_large               | available       |                                                                        |
| alphapose_crowd_baseline | licence_blocked | licence is unconfirmed; PRD 3 requires licence status recorded before  |
| dfine_native             | available       |                                                                        |
| rf_detr_native           | launch_failed   | import rfdetr failed: ModuleNotFoundError: No module named 'rfdetr'    |
| dfine_sahi               | available       |                                                                        |
| rf_detr_sahi             | launch_failed   | import rfdetr failed: ModuleNotFoundError: No module named 'rfdetr'    |
| gemma_marked_evidence    | available       |                                                                        |

A configuration that could not run stays in this table as unavailable and is never substituted (PRD 5).

## Recommendation

None. PRD 3 blocks detector selection until the product owner approves the taxonomy. The comparison below is recorded for review, not acted on.

no forced winner. A configuration that could not run stays in the table as unavailable (PRD 5).
