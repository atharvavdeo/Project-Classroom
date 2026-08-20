# Architecture audit — run 1446

An independent read of the code against `ARCHITECTURE.md` and `details.md`,
followed by direct verification of every P0 finding. Findings marked
**VERIFIED** were re-checked against the code and the run artifacts here, not
taken on report.

**Verdict: not safe to lock in.** The measurements are sound. The machinery
that turns them into a ranked reviewer queue is not.

---

## P0 — Documented invariants that do not hold

### 1. VERIFIED — "byte-identical rows verified by digest" can never fail

`pipeline/pose.py:493` sets `manifest_digest=manifest_digest(rows)`.
`pipeline/pose.py:615` computes `expected = manifest_digest(rows)` — from the
**same in-memory list**. The comparison is `x != x`. Identical structure in
`pipeline/object_detection.py:344`, where one `crop_digest` variable is threaded
from `tools/run_detector_ab.py:120` into every runner and then compared with
itself.

`"manifest_equivalence": "verified"` and `"crop_equivalence": "verified"` are
printed on every run and quoted in `tools/write_findings.py` and
`generate_final_report.py`. They are decoration. A real check would digest the
rows each runner actually consumed.

**Fix:** have each runner hash the rows it iterated and return that; compare
against the frozen manifest's digest.

### 2. VERIFIED — three of eight ranking components are structurally zero

Measured over all 113 ranked events in this run:

| video | events | components that ever fire |
|---|--:|---|
| 01_phone | 46 | motion_deviation 46, motion_area 46, duration 46, object_supported **3**, object_single_frame 2 |
| 04_talking | 38 | motion_deviation, motion_area, duration |
| 12_paper | 29 | motion_deviation, motion_area, duration |

`pose_sustained` (weight 1.2), `seat_quality` (0.8) and `repetition` (0.5) are
**0 on every event of every video**.

- `pose_sustained`: `tools/render_evidence_packet.py:172` calls
  `ps.evidence(...)` without `baseline_pitch`/`baseline_mad`, so the guard at
  `pose.py:579` never runs and `head_dwell_frames` stays 0. `sustained` then
  reduces to `wrist_lap_frames >= 2`, which needs `desk_line_y` from an
  **approved** seat — unavailable while calibrations are DRAFT.
- `repetition` / `seat_quality`: `ranking.py:159-160` reads `event["repetition"]`
  and `event["seat_confidence"]`. Neither key exists in
  `06_segmentation/events.jsonl`.

ARCHITECTURE §7.3 announces this defect as **fixed**. Passing the argument was
not sufficient; the value is still zero.

### 3. VERIFIED — the four-band disposition is two bands

`high_potential_corroborated` = **0** and `rejected_not_offered` = **0** across
all 113 events.

- `ranking.rank()` is called from `render_evidence_packet.py:335` without
  `gate_state_of`, so every event defaults to `"pass"` and REJECTED is
  unreachable. The `gates.DEPRIORITISE` routing in `motion.py` reaches nothing.
- Events carry no `health_state`, so `_penalties()` sees an empty set and **no
  multiplicative penalty ever fires**. The 0.6 / 0.5 / 0.7 / 0.4 table is inert.
- The top band needs `len(modalities) == 2`, which requires `pose_sustained`
  (see #2). Unreachable by construction.

ARCHITECTURE §7.2's headline numbers ("1 high-potential at seat S-61";
"46 events → 13") match no run in this repository. Actual for 01_phone: 46 → 15,
zero high-potential.

### 4. VERIFIED — "resolved seat" is a polygon name, not a person→seat association

`ranking.py:307` tests `event.seat_state not in ("", None, "unattributed",
"ambiguous_seat")`, and `seat_state` comes from the motion event's zone id. The
module that actually refuses attribution under DRAFT calibration,
`pipeline/seat_association.py`, never reaches this test. The top band's "four
independent conditions" are three.

---

## P1 — Statistical soundness

### 5. VERIFIED — `look_down` cannot fire; my own fix closed it

`head_pose.py:415` applies `min_departure_from_baseline = 0.30` to **all three**
excursion kinds. That constant is justified in its own docstring entirely from
yaw's saturation behaviour. Pitch is on a different scale:

```
pitch  n=796   p10 -0.164   median -0.000   p90  0.283
yaw    n=813   p10 -1.163   median -0.110   p90  1.500
```

A 0.30 departure above a ~0.0 median sits **beyond the 90th percentile of
pitch**. Result: **0 `look_down` events across 796 pitch samples**, while the
same rows classify 43 frames `downward`.

**Fix:** separate departure thresholds per measure, derived from each one's own
observed spread.

### 6. VERIFIED, but the reported magnitude was wrong — one-ear yaw sign

`head_pose.py:237` asserts direction from which ear survived, with no image-
handedness correction, while the two-ear branch derives `direction` per
observation. The convention is plausibly inverted.

The audit reported this affects 383 of 813 samples (47%). **Measured: 48 of 813
(5.9%)** — 765 come from the two-ear branch. Real bug, one-twentieth of the
claimed scale, and it cannot explain video 04's `yaw_right` skew.

### 7. VERIFIED — `neighbour_bias()` never attaches

Two independent faults. `head_pose.py:472` assumes "the subject's left is
toward smaller x", the opposite of `estimate()`'s documented convention — and
that premise is unstable anyway: over 1007 two-ear observations,
`left_ear.x >= right_ear.x` holds only 56.4% of the time, because an oblique
ceiling camera sees as many backs of heads as fronts. Separately,
`summarise():564` looks up the composite key `"unattributed/track_12"` while
`neighbour_bias()` keys on raw `"unattributed"`, so the lookup always misses.

ARCHITECTURE §10 lists neighbour bias under "measured and defensible". It is
neither computed nor attached.

### 8. VERIFIED — ranking components are on incommensurable scales

`motion_deviation` is a raw robust z (observed to 84.94); `object_supported` is
a raw count; the other five are bounded to [0,1]. The linear sum gives effective
influence ~85 : 2, not the published 1.0 : 2.0. The top event of 01_phone
(score 85.63) is **99.2% motion deviation**.

---

## P2 — Absolute pixel thresholds that do not transfer

The corpus mixes 1280×720@25 with 640×480@8. `details.md` §9 states the
governing principle — *absolute area cannot transfer* — and then violates it in
at least six places.

| constant | file | why it breaks |
|---|---|---|
| `abstain_below_object_px = 120.0` | `object_detection.py:97` | the same phone measures 485 px² on v01 and 883 px² on v03; on 640×480 it lands **at** the floor. A per-camera on/off switch on phone evidence. |
| `min_person_px = 60.0` | `pose.py:92` (comment says "at 720p") | rows dropped: **01_phone 0.4%, 04_talking 3.7%, 12_paper 16.0%**. `evidence():562` *drops* them rather than recording NOT_RESOLVABLE as documented. |
| `abstain_below_native_px = 32`, `min_native_px = 24`, `min_slice_native_px = 32` | `object_detection.py:93`, `crops.py:64,69`, `sahi_adapter.py:69` | abstention rates are not comparable between cameras, yet `compare()` publishes them side by side. |
| `implausible_translation_px = 120.0`, `epoch_break_translation_px = 8.0` | `alignment.py:81,91` | 9.4% of width at 1280, 18.8% at 640 — 2× more permissive on video 04. |
| `block_mag_threshold = 2.0` | `motion.py:68` | binarised **before** per-seat baselining, so the z-score cannot correct it. Video 04 is half the resolution and 8 fps, giving ~1.5× larger MV magnitudes for identical motion. Plausible mechanical cause of the 96.5% over-segmentation. |
| `lap_entry_margin_px = 8.0`, `min_head_scale_px = 6.0` | `pose.py:97`, `head_pose.py:71` | `lap_entry_margin_px` is the only live path to `sustained`. `min_head_scale_px` sits between the corpus's measured head scales, making head orientation a near-binary per-camera switch. |

**Correctly relative, for the record:** `recurrence_grid_fraction`,
`max_wrist_distance_torso`, `hand_context_torso`, all IoU thresholds, all
z-unit thresholds in `segmentation.py` and `baseline.py`.

---

## P3 — Silent failures

| finding | file | effect |
|---|---|---|
| `min_wrist_coverage` declared with a careful justification and **never read** | `temporal_confirmation.py:103` | `flag_handheld` samples only 2 instants and accepts `len(distances) >= 1`, so a 40-frame group can be demoted on one wrist observation — the opposite of the docstring's promise. |
| `assert_isolated` skips stages 01–04 and swallows `JSONDecodeError` | `reference_manifest.py:258,311` | reference data attached during calibration or alignment passes the check; a corrupt artifact reads identically to a clean one. |
| AlphaPose CPU fallback printed, never persisted | `pose.py:770` | `PoseResult` has no `device` field, so `rows_per_second` is published with no record of what it ran on — and `--pose-device` defaults to `cpu`. ARCHITECTURE §3.1 uses rows/s as a locked selection criterion. |
| unreadable frame returns `[]` | `object_detection.py:452,507` | indistinguishable from "looked and found nothing"; no `crops_abstained` increment, no trace. |
| missing pose observations return `{}` | `render_evidence_packet.py:162` | degrades silently to "no sustained pose anywhere" — the same signature as #2. |

---

## P4 — Dead code and dead config

`gates.py` `PENALTY` table / `GateRecord.penalty` / `GateLog.multiplier()`;
`ConfirmConfig.min_wrist_coverage`; `HealthConfig.compression_noise_max_area`,
`compression_noise_max_duration_ms` (the live check in `motion.classify_noise()`
has **no duration test at all**, contradicting details §2.3);
`HealthConfig.environmental_min_persistence` (duplicated live in
`BaselineConfig`); `TrackConfig.min_track_ms` (so flicker tracks are never
marked); `roi.heat()`; `health.hamming()`; `ingest.stats_series()`;
`RankedEvent.config_ids` / `artifact_paths` (serialised empty on every row).

**VERIFIED — labels are 49 declared, 47 unique.** `insufficient_visual_evidence`
and `not_resolvable_at_source_scale` are each defined twice and `BY_KEY` is a
dict comprehension, so the second silently wins:
`describe("not_resolvable_at_source_scale")` can never return the `facing`
label. Six labels (`lean_toward_neighbour`, `left_seat`, `returned_to_seat`,
`seat_change`, `look_up`, `over_shoulder`) are declared and never emitted.

---

## Cross-cutting: the head/neck stage is a reporting cul-de-sac

**VERIFIED.** `grep -rn "head_pose\|excursions.jsonl\|08_pose/head"` across
`ranking.py`, `evidence.py` and `render_evidence_packet.py` returns **nothing**.

The ARCHITECTURE §8 diagram draws `POSE → HP → EXC → NB → PKT → RNK → DISP`.
That edge does not exist. Head orientation is computed, written to JSON, and has
no effect on what a reviewer is shown or in what order. `pose.evidence()` — the
function that *does* feed rank — still uses the old `head_pitch` vertical-offset
proxy that §5.1 identifies as the thing that was broken.

PS #1's named objective (repeated sideward glances, excessive head turning) is
measured and then discarded.

---

## Also: `book_or_notebook_like_object` is a corroborating TARGET class

`object_detection.py:67`. ARCHITECTURE §6 states every "chit" arrived as COCO
`book` and that rendering showed answer sheets and packaging, and declares chit
detection out of scope. Marking this class as corroborating re-admits that
signal through the ranking path. Inert today only because `pose_sustained` is 0
— it activates the moment that is fixed.

---

## Fix order

1. **Make the digest check real** (#1) — otherwise every A/B claim in the repo
   is unsupported.
2. **Wire `gate_state` and `health_state` into ranking** (#3) — restores two of
   four bands and the entire penalty design.
3. **Separate the pitch departure threshold from yaw's** (#5) — restores
   `look_down`.
4. **Connect head excursions to evidence and ranking** (cul-de-sac) — otherwise
   the head work has no product effect.
5. **Normalise the six absolute pixel constants** to torso width or frame
   fraction (#P2) — otherwise nothing transfers to a new camera.
6. Correct the label duplicates and the docs' 49/47 and group-count claims.
