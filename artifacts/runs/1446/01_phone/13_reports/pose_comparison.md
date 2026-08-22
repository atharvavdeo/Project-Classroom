# Pose configuration comparison

**Manifest equivalence: verified** — 1076 rows, digest `c1b2f98d2bb45bb5`

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

## Rows processed

| Seat state   | Rows |
|--------------|------|
| unattributed | 1076 |

| Health state | Rows |
|--------------|------|
| none         | 1076 |

## Configurations

| Configuration            | State     | Rows | Joints visible | Wrist visible | Wrist occluded | Wrist unresolvable | Swaps flagged | Rows/s |
|--------------------------|-----------|------|----------------|---------------|----------------|--------------------|---------------|--------|
| rtmpose_baseline         | available | 1076 | 92.5%          | 1464          | 446            | 0                  | 44            | 21.65  |
| rtmo_baseline            | available | 1076 | 77.9%          | 831           | 259            | 0                  | 19            | 29.9   |
| alphapose_crowd_baseline | available | 1076 | 86.3%          | 1541          | 603            | 8                  | 38            | 13.01  |

## Selection

none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

### What the wrist columns mean

- **visible** — the estimator returned a wrist above the confidence floor on a person large enough to resolve one.
- **occluded** — a wrist the estimator could not see, most often because the desk is between it and the camera. This is the interesting case, not a gap.
- **unresolvable** — the person box is below the source-scale floor. This is a statement about the camera, not the person, and no amount of model improvement changes it.
