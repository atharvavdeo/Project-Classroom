# Pose configuration comparison

**Manifest equivalence: verified** — 5881 rows, digest `3df16113796accf0`

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

## Rows processed

| Seat state   | Rows |
|--------------|------|
| unattributed | 5881 |

| Health state | Rows |
|--------------|------|
| none         | 5881 |

## Configurations

| Configuration            | State     | Rows | Joints visible | Wrist visible | Wrist occluded | Wrist unresolvable | Swaps flagged | Rows/s |
|--------------------------|-----------|------|----------------|---------------|----------------|--------------------|---------------|--------|
| rtmpose_baseline         | available | 5881 | 81.0%          | 5088          | 2852           | 548                | 1187          | 101.42 |
| rtmo_baseline            | available | 5881 | 78.7%          | 2906          | 644            | 0                  | 752           | 365.49 |
| alphapose_crowd_baseline | available | 5881 | 62.2%          | 5934          | 3950           | 1878               | 1197          | 28.68  |

## Selection

none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

### What the wrist columns mean

- **visible** — the estimator returned a wrist above the confidence floor on a person large enough to resolve one.
- **occluded** — a wrist the estimator could not see, most often because the desk is between it and the camera. This is the interesting case, not a gap.
- **unresolvable** — the person box is below the source-scale floor. This is a statement about the camera, not the person, and no amount of model improvement changes it.
