# Run 1501 — continuous per-person pipeline with object evidence and flags

Two recordings, everything on GPU. This run exists because run 1446 sampled
only 20-39% of each video: it built its sample plan from the motion candidate
manifest, so **60-80% of every recording was never decoded for person
detection**, and video 01 was only looked at between 53.6s and 102.9s.

| run dir | source | duration | resolution |
|---|---|--:|---|
| `12_paper` | Seat No. 12 was seen taking a piece of paper from the desk | 88.4 s | 1280×720 @25 |
| `01_phone` | 01.Candidate was found using a mobile phone… | 131.3 s | 1280×720 @25 |

Sampled at **4 Hz across the whole duration**, ignoring motion candidates
entirely.

---

## What changed since 1446

**Continuous coverage.** `person_detection.plan_continuous()` lays a uniform
grid over the full recording. Every person in every sampled frame is detected,
tracked, posed and measured, whether or not anything interesting is happening —
which is what per-person baselines require. A metric computed only on flagged
moments has already assumed its conclusion.

**Object evidence at no extra model cost.** D-FINE returns all 80 COCO classes
from one inference; the person pass filtered to `person` and discarded the rest,
so every frame paid for a phone detection and then deleted it. An `object_sink`
now keeps them.

**A wrist-anchored crop pass**, because the full-frame pass resizes 1280×720 to
640×640 and a 30×20 px phone becomes ~15×13 px and vanishes. Measured: 1,786
crops → 1,962 objects on video 01 that the full-frame pass could not resolve.

**Per-person flags**, each carrying the figure that produced it. Thresholds are
deliberately generous: 1446's guardrails let **zero of 113 ranked events** reach
the top disposition band, and a triage tool that surfaces nothing is worse than
one that surfaces too much.

---

## Results

| | 12_paper | 01_phone |
|---|--:|--:|
| sampled frames | 354 | 526 |
| coverage of recording | **99%+** | **99%+** |
| person boxes | 6,109 | 1,301 |
| pose observations | 6,109 | 1,301 |
| tracks | 236 | 84 |
| tracked ≥90% of the video | **4** | **0** |
| with a usable yaw baseline | 43 | 15 |
| objects kept (full frame) | 20,803 | 20,390 |
| wrist crops → objects | 5,111 → 25,855 | 1,786 → 1,962 |
| objects attached to hands | 10,198 | 3,032 |
| unattached (no wrist within 2 torso widths) | 10,605 | 17,358 |

### Flags

| flag | 12_paper | 01_phone |
|---|--:|--:|
| `no_orientation_baseline` | 193 | 69 |
| `target_object_near_hands` | 25 | 13 |
| `low_coverage` | 22 | 10 |
| `persistent_object_beside_seat` | 10 | 5 |
| `sustained_attention_away` | 4 | 4 |
| `frequent_orientation_change` | 3 | 3 |
| `prolonged_downward_orientation` | 2 | 1 |

### The head-movement signal, where it can be measured

```
12_paper track 18:  129 samples, 80% away from own baseline, 44 left / 41 right
12_paper track 16:  351 samples, 99% coverage, 53% away, 36 left /  5 right
01_phone track 25:   56 samples, 73% away, 12 left / 21 right
```

Track 18 turning both directions across 129 sightings is the repeated-glance
pattern PS #1 describes.

---

## Defects found and corrected during this run

1. **Phones lost to a dictionary miss.** `classroom.detect` renames `cell phone`
   → `phone_like` and `book` → `paper_like` before the pipeline sees them, while
   `COCO_TO_TAXONOMY` keys on the raw COCO names. Every phone and paper was
   dropped by `dict.get()` returning `None`; only classes absent from that remap
   (`tv`→monitor, `backpack`→bag) survived. Measured before the fix on the phone
   video: 11,964 objects kept, 1,435 attached to hands, **zero** a target class.
   `dfine_runner` already passed `class_map={}` for exactly this reason; the two
   new call sites did not. **I first explained this as a resolution limit — that
   was wrong, and I should have tested the known-positive case before offering
   an explanation.**

2. **Object counts dominated by persistent false positives.** After the fix,
   1,032 "phone" attachments on the paper video, with 18 of 45 tracks carrying
   one in over half their sightings and rates of **1.62 and 3.00 per sighting** —
   arithmetically impossible for a single held phone. Flags jumped from 4 people
   to 32. Added a persistence rule: a class attached in >50% of a person's
   sightings is `persistent_object_beside_seat` (info, non-corroborating), and
   only a *transient* attachment is reportable. The same argument
   `flag_recurrent` makes in image space, applied inside one person's timeline.

3. **Re-running into an existing run id.** The immutability guard refused
   (`Run IDs are immutable; pick a new one`) but the driver continued past it,
   and later stages would have appended to the existing artifacts — the same
   duplication that previously doubled `ranked_events.jsonl`. Cleared and
   restarted as a fresh id.

4. **A patch that never ran.** I reported flag counts from patched code, but
   both videos' subprocesses had started before the patch landed on disk;
   `near_hand_object_rates` was absent from every profile. `tools/reprofile.py`
   now rebuilds profiles and flags from stored samples in seconds, so a rule
   change never needs a GPU re-run again.

5. **Stale overlay frames.** Overlays are drawn from the current strongest
   excursions, so files left from a previous pass pointed at events that no
   longer existed. Cleared before writing.

---

## Verified by looking at the pixels

**The object flags are firing on seat placards.** Track 29 on video 01 — the
highest phone-rate track — is flagged because D-FINE calls the numbered cards on
the wall ("21", "22", "23") cell phones. All six rendered crops, confidence
**0.30–0.49**, against **0.933** for the one phone verified by eye at 69.17 s.

Rendered evidence: `artifacts/phone_check_01.jpg`.

The persistence rule cannot catch this: a placard is not attached in *most* of a
person's sightings, only whenever they are near that wall. **A confidence floor
of ~0.6 on target classes would remove this entire class of error without
touching the one true positive**, and is the next change to make.

---

## Honest status

**Solid:** continuous coverage; AlphaPose returning a person on 100% of manifest
rows with full skeletons; per-person profiles and baselines; annotated videos
with skeleton, track id, facing state, yaw value and gaze vector on every person
in every frame.

**Measurable but partial:** head orientation resolves for 43 of 236 people on
video 12 and 15 of 84 on video 01. For those it works and the signal is real.
For the rest the head is too small or too often occluded.

**Not trustworthy:** object flags, for the reason above.

**Unaddressed:** re-identification does not exist, so 84 tracks on video 01
represent a handful of people and no one is tracked ≥90% there. Seat attribution
is blocked on 6 of 7 draft calibrations, so every row is `unattributed` and the
pipeline cannot distinguish a seated candidate from a passer-by. The audit
findings in `docs/AUDIT_1446.md` are also still open.

**No precision or recall anywhere** — there is no reference manifest, so every
count is what the system reported, not what was true.
