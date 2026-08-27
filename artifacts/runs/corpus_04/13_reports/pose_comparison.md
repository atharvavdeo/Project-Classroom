# Pose configuration comparison

**Manifest equivalence: verified** — 1242 rows, digest `79250e33df9ad4e5`

every configuration below was given byte-identical manifest rows; a difference in their metrics is therefore attributable to the model (PRD 4)

## Rows processed

| Seat state   | Rows |
|--------------|------|
| unattributed | 1242 |

| Health state | Rows |
|--------------|------|
| none         | 1242 |

## Configurations

| Configuration            | State     | Rows | Joints visible | Wrist visible | Wrist occluded | Wrist unresolvable | Swaps flagged | Rows/s |
|--------------------------|-----------|------|----------------|---------------|----------------|--------------------|---------------|--------|
| rtmpose_baseline         | available | 1242 | 85.7%          | 1315          | 693            | 8                  | 31            | 3.13   |
| rtmo_baseline            | available | 1242 | 79.0%          | 257           | 73             | 0                  | 3             | 3.8    |
| alphapose_crowd_baseline | available | 1242 | 72.2%          | 1410          | 976            | 98                 | 36            | 3.4    |

## Selection

none. PRD 5 forbids naming a configuration best until every runnable mandatory configuration has a matched report; selection happens in stage 12 against references.

### What the wrist columns mean

- **visible** — the estimator returned a wrist above the confidence floor on a person large enough to resolve one.
- **occluded** — a wrist the estimator could not see, most often because the desk is between it and the camera. This is the interesting case, not a gap.
- **unresolvable** — the person box is below the source-scale floor. This is a statement about the camera, not the person, and no amount of model improvement changes it.
