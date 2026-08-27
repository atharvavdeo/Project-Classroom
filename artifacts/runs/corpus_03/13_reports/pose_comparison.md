# Pose configuration comparison

**Manifest equivalence: verified** — 217 rows, digest `1a8b6e0ebc502fc6`

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

## Rows processed

| Seat state   | Rows |
|--------------|------|
| seated       | 195  |
| unattributed | 22   |

| Health state | Rows |
|--------------|------|
| none         | 217  |

## Configurations

| Configuration            | State     | Rows | Joints visible | Wrist visible | Wrist occluded | Wrist unresolvable | Swaps flagged | Rows/s |
|--------------------------|-----------|------|----------------|---------------|----------------|--------------------|---------------|--------|
| rtmpose_baseline         | available | 217  | 96.0%          | 265           | 31             | 0                  | 6             | 1.79   |
| rtmo_baseline            | available | 217  | 85.4%          | 224           | 0              | 0                  | 8             | 5.46   |
| alphapose_crowd_baseline | available | 217  | 70.5%          | 285           | 135            | 14                 | 9             | 2.14   |

## Selection

none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

### What the wrist columns mean

- **visible** — the estimator returned a wrist above the confidence floor on a person large enough to resolve one.
- **occluded** — a wrist the estimator could not see, most often because the desk is between it and the camera. This is the interesting case, not a gap.
- **unresolvable** — the person box is below the source-scale floor. This is a statement about the camera, not the person, and no amount of model improvement changes it.
