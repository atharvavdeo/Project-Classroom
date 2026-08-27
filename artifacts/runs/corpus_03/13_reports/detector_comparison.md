# Detector and slicing comparison

**Crop equivalence: verified** — 217 crops, digest `ac7227b7dc3e6daa`

every configuration below processed byte-identical crop rows, so a difference in their output is attributable to the detector and its slicing (PRD 4)

## The crops every configuration received

| Crop policy                    | Crops |
|--------------------------------|-------|
| lap_bag_context                | 71    |
| native_source_context_fallback | 21    |
| pose_hand_context              | 125   |

Why crops fell back from `pose_hand_context`:

| Reason                       | Crops |
|------------------------------|-------|
| no_pose_observation          | 69    |
| wrist_confidence_below_floor | 23    |

- **0 of 217 crops (0.0%)** are below the native-resolution floor. No detector can answer for these, and that is a statement about the camera.
- Median magnification **2.96x**, max **13.33x**. magnification is interpolation, not detail. A crop at 20x is 640 px of upscaled evidence over a 32 px source footprint, and the abstention rule is judged on the footprint (PRD 10).

## Configurations

| Configuration  | State     | Crops | Abstained | Slices | Merged | Phone | Chit | Hard negatives | Crops/s |
|----------------|-----------|-------|-----------|--------|--------|-------|------|----------------|---------|
| rf_detr_native | available | 217   | 0.0%      | —      | 617    | 10    | 0    | 53             | 5.06    |
| dfine_native   | available | 217   | 0.0%      | —      | 1648   | 108   | 0    | 735            | 1.58    |
| rf_detr_sahi   | available | 217   | 0.0%      | 1065   | 919    | 19    | 0    | 94             | 0.79    |
| dfine_sahi     | available | 217   | 0.0%      | 1065   | 2767   | 159   | 0    | 1165           | 0.24    |

Phone and chit are reported separately throughout, per PRD 15. **Hard negatives** counts calls on mouse, keyboard, calculator, answer sheet, monitor bezel and stationery — the objects that look like a phone at 30x20 px. PRD 10 treats a phone called on one of these as a failure cluster to fix, not a false positive to average away.

## Temporal confirmation

| Status                               | Groups |
|--------------------------------------|--------|
| conflicting_object_evidence          | 0      |
| insufficient_visual_evidence         | 0      |
| single_frame_object_candidate        | 2818   |
| temporally_supported_object_evidence | 1122   |

**2818 singletons retained, 0 deleted.** no group was deleted. A singleton is a status, not a reason to discard, and the count above is retained evidence (PRD 10).

## Selection

none. PRD 3 blocks detector selection and evaluation until the product owner approves the taxonomy, and PRD 15 places selection in stage 12 against the reference manifest.

Taxonomy approved: **False**
