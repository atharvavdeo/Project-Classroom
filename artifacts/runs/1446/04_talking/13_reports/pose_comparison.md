# Pose configuration comparison

**Manifest equivalence: verified** — 3542 rows, digest `fd5679cc30d38d47`

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

## Rows processed

| Seat state   | Rows |
|--------------|------|
| unattributed | 3542 |

| Health state | Rows |
|--------------|------|
| none         | 3542 |

## Configurations

| Configuration            | State     | Rows | Joints visible | Wrist visible | Wrist occluded | Wrist unresolvable | Swaps flagged | Rows/s |
|--------------------------|-----------|------|----------------|---------------|----------------|--------------------|---------------|--------|
| rtmpose_baseline         | available | 3542 | 85.9%          | 3776          | 1966           | 22                 | 95            | 37.24  |
| rtmo_baseline            | available | 3542 | 78.5%          | 681           | 205            | 0                  | 9             | 78.36  |
| alphapose_crowd_baseline | available | 3542 | 72.3%          | 4020          | 2800           | 264                | 108           | 21.23  |

## Selection

none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

### What the wrist columns mean

- **visible** — the estimator returned a wrist above the confidence floor on a person large enough to resolve one.
- **occluded** — a wrist the estimator could not see, most often because the desk is between it and the camera. This is the interesting case, not a gap.
- **unresolvable** — the person box is below the source-scale floor. This is a statement about the camera, not the person, and no amount of model improvement changes it.
