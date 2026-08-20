# Detector and slicing comparison

**Crop equivalence: verified** — 5881 crops, digest `654ca2ddec3aa28a`

every configuration below processed byte-identical crop rows, so a difference in their output is attributable to the detector and its slicing (PRD 4)

## The crops every configuration received

| Crop policy                    | Crops |
|--------------------------------|-------|
| native_source_context_fallback | 3499  |
| pose_hand_context              | 2382  |

Why crops fell back from `pose_hand_context`:

| Reason                               | Crops |
|--------------------------------------|-------|
| no_pose_observation                  | 1637  |
| wrist_confidence_below_floor         | 1588  |
| wrist_not_resolvable_at_source_scale | 274   |

- **233 of 5881 crops (4.0%)** are below the native-resolution floor. No detector can answer for these, and that is a statement about the camera.
- Median magnification **7.34x**, max **21.46x**. magnification is interpolation, not detail. A crop at 20x is 640 px of upscaled evidence over a 32 px source footprint, and the abstention rule is judged on the footprint (PRD 10).

## Configurations

| Configuration  | State                | Crops | Abstained | Slices | Merged | Phone | Chit | Hard negatives | Crops/s |
|----------------|----------------------|-------|-----------|--------|--------|-------|------|----------------|---------|
| rf_detr_native | resource_unavailable | —     | —         | —      | —      | —     | —    | —              | —       |
| dfine_native   | available            | 5881  | 4.0%      | —      | 32203  | 1648  | 0    | 14262          | 17.24   |
| rf_detr_sahi   | resource_unavailable | —     | —         | —      | —      | —     | —    | —              | —       |
| dfine_sahi     | available            | 5881  | 4.0%      | 21940  | 47888  | 2910  | 0    | 15577          | 3.36    |

Phone and chit are reported separately throughout, per PRD 15. **Hard negatives** counts calls on mouse, keyboard, calculator, answer sheet, monitor bezel and stationery — the objects that look like a phone at 30x20 px. PRD 10 treats a phone called on one of these as a failure cluster to fix, not a false positive to average away.

### Unavailable configurations

Retained per PRD 5; never substituted by an available one.

| Configuration  | State                | Reason                                                                                                        |
|----------------|----------------------|---------------------------------------------------------------------------------------------------------------|
| rf_detr_native | resource_unavailable | skipped by --no-rfdetr; it stays in the comparison as unavailable and is never substituted by D-FINE (PRD 5). |
| rf_detr_sahi   | resource_unavailable | skipped by --no-rfdetr; it stays in the comparison as unavailable and is never substituted by D-FINE (PRD 5). |

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 38     |
| recurrent_scene_object               | 33417  |
| single_frame_object_candidate        | 4563   |
| temporally_supported_object_evidence | 1601   |

**4563 singletons retained, 0 deleted.** no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Selection

none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

Taxonomy approved: **False**
