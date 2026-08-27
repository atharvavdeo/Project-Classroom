# Pose configuration comparison

**Manifest equivalence: verified** — 379 rows, digest `67489132465ccb8b`

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

## Rows processed

| Seat state   | Rows |
|--------------|------|
| unattributed | 379  |

| Health state | Rows |
|--------------|------|
| none         | 379  |

## Configurations

| Configuration            | State     | Rows | Joints visible | Wrist visible | Wrist occluded | Wrist unresolvable | Swaps flagged | Rows/s |
|--------------------------|-----------|------|----------------|---------------|----------------|--------------------|---------------|--------|
| rtmpose_baseline         | available | 379  | 92.1%          | 513           | 161            | 0                  | 11            | 1.3    |
| rtmo_baseline            | available | 379  | 77.4%          | 303           | 95             | 0                  | 7             | 1.68   |
| alphapose_crowd_baseline | available | 379  | 86.6%          | 549           | 203            | 6                  | 15            | 2.46   |

## Selection

none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

### What the wrist columns mean

- **visible** — the estimator returned a wrist above the confidence floor on a person large enough to resolve one.
- **occluded** — a wrist the estimator could not see, most often because the desk is between it and the camera. This is the interesting case, not a gap.
- **unresolvable** — the person box is below the source-scale floor. This is a statement about the camera, not the person, and no amount of model improvement changes it.
