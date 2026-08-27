# Detector and slicing comparison

**Crop equivalence: verified** — 379 crops, digest `165261f983377ae1`

every configuration below processed byte-identical crop rows, so a difference in their output is attributable to the detector and its slicing (PRD 4)

## The crops every configuration received

| Crop policy                    | Crops |
|--------------------------------|-------|
| native_source_context_fallback | 111   |
| pose_hand_context              | 268   |

Why crops fell back from `pose_hand_context`:

| Reason                       | Crops |
|------------------------------|-------|
| no_pose_observation          | 42    |
| wrist_confidence_below_floor | 69    |

- **4 of 379 crops (1.1%)** are below the native-resolution floor. No detector can answer for these, and that is a statement about the camera.
- Median magnification **3.74x**, max **13.33x**. magnification is interpolation, not detail. A crop at 20x is 640 px of upscaled evidence over a 32 px source footprint, and the abstention rule is judged on the footprint (PRD 10).

## Configurations

| Configuration  | State     | Crops | Abstained | Slices | Merged | Phone | Chit | Hard negatives | Crops/s |
|----------------|-----------|-------|-----------|--------|--------|-------|------|----------------|---------|
| rf_detr_native | available | 379   | 1.1%      | —      | 594    | 11    | 0    | 111            | 3.62    |
| dfine_native   | available | 379   | 1.1%      | —      | 2325   | 92    | 0    | 206            | 1.68    |
| rf_detr_sahi   | available | 379   | 1.1%      | 1663   | 1285   | 34    | 0    | 243            | 0.87    |
| dfine_sahi     | available | 379   | 1.1%      | 1663   | 3709   | 158   | 0    | 452            | 0.2     |

Phone and chit are reported separately throughout, per PRD 15. **Hard negatives** counts calls on mouse, keyboard, calculator, answer sheet, monitor bezel and stationery — the objects that look like a phone at 30x20 px. PRD 10 treats a phone called on one of these as a failure cluster to fix, not a false positive to average away.

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 12     |
| single_frame_object_candidate        | 3822   |
| temporally_supported_object_evidence | 1522   |

**3822 singletons retained, 0 deleted.** no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Selection

none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

Taxonomy approved: **False**
