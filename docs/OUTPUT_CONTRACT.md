# The output contract

**Version 1.0.0.** Frozen before the dashboard is built, because the dashboard
can only render what the output already says, and a UI built over an
underspecified output ends up inventing the missing parts in CSS.

This document is the specification. The code is
`pipeline/reason_codes.py`, `pipeline/evidence_record.py`, `pipeline/fusion.py`,
and `tools/build_evidence_records.py`. The invariants are pinned by
`tests/test_output_contract.py` (37 tests).

`docs/DASHBOARD_OUTPUTS.md` is an **additive** companion covering the four
orientation labels on `PersonProfile`. It changes no state, code or flag here;
a client that ignores it sees exactly this contract.

---

## 0. The one sentence

**The system produces defensible review evidence. It does not pronounce
cheating.** Every component is a proposal source; no component is a judge; the
only states that assert a policy violation are written by a human.

---

## 1. Component roles

| Component | Role | Must not do | Enforced by |
|---|---|---|---|
| D-FINE (ONNX, stock COCO) | Person tracking, workstation context | Produce a phone or chit policy flag | `DFINE_OBJECT_CONTEXT` routes to `context_observation` only; `person_timeline.py` no longer lists `phone` in `target_names` |
| `chit-paper-new/2` (Roboflow) | Class-specific proposal generator | Flag a candidate by itself | Proposal always needs geometry + verification + association + persistence |
| SAM 3 (Roboflow-hosted) | Verify/refine the proposal against hard negatives | Erase a proposal because it is unsupported | `unsupported` → `SAM3_NOT_CONFIRMED` → `needs_better_view`, and the record survives |
| AlphaPose FastPose_DUC | Wrist/head/body features and observability | Claim eye gaze, talking, or intent | `ORIENTATION_*` guardrails; `TALKING_NOT_MEASURABLE` is emitted on every track |
| Fusion (`pipeline/fusion.py`) | Route evidence to review | Determine cheating | `assert_machine_state()` raises on the two human states |

### The COCO phone route is removed as an *escalation* — it survives as a *proposal*

`person_timeline.py` previously promoted `target_object_near_hands` to
`severity: high` on COCO's `cell phone` rate. **Across the four completed runs,
40 flags cited a COCO phone and 27 of those were high severity.** Every one of
them is now context: `DFINE_OBJECT_CONTEXT`, which may never route anyone.

What D-FINE's `cell phone` class *may* still do is raise a question.
`OBJECT_PROPOSAL_PHONE` is emitted on every such detection — measured: 332 of
the 5,186 records in run 1511/01_phone — and that proposal is then put to SAM 3
with keyboard, mouse, monitor and bottle as explicit hard negatives. Only
SAM 3 supporting it (`SAM3_PHONE_NAMED`), on a frame whose own wrist distance
clears the association gate, reaches review.

**Detectors propose; the referee verifies; neither judges.** A COCO score is a
reason to look, never a finding.

---

## 2. The evidence record

One record per proposal. Written the moment a detector fires; **never deleted**,
not by a gate, not by SAM 3, not by a reviewer. Suppression is written *into*
the record as a reason code.

```
record_id             content-addressed sha1, stable across re-runs
contract_version      consumers refuse a version they do not know
provenance            run_id, video, camera, pts_ms, stage
detector    [frozen]  name, version, class_name, confidence
geometry    [frozen]  box, person_box, size_ratio_to_person, torso_scale_px
sam3                  attempted, responded, error, prompt, model, latency,
                      target_claims[], negative_claims[]
association           track_id, seat, seat_calibrated, wrist_resolved,
                      wrist_distance_norm, nearest_wrist, competing_tracks
quality               person_box_px, head_resolvable, hands_resolvable,
                      occluded, track_coverage
reason_codes [append] the whole argument, including the parts against
review       [append] immutable human decisions
```

Three properties are load-bearing:

- **`detector` and `geometry` are frozen dataclasses.** No later stage can
  rewrite what the detector said. SAM 3 disagreeing shows up as two stored
  claims plus a reason code, never as an edited class name.
- **`sam3.responded` is separate from "found nothing".** An outage that reads
  as an all-clear is the single most dangerous failure this contract prevents.
- **No masks, no crops.** A four-hour recording at 4 Hz produces ~10⁵ records.
  The box plus `pts_ms` is enough to re-cut the crop from the source video.

A record may never carry a field named `cheating`, `guilty` or `violation`, nor
any number presented as a probability of one.

---

## 3. The reason vocabulary

36 codes, closed. `reason_codes.get()` raises on anything invented at a call
site. Each code carries a `route` (the state it argues for), a `meaning` (what
a reviewer reads) and a `guardrail` (what it must never be read as saying).

**27 are implemented. 9 are declared only** — they exist because the behaviour
is in the operating matrix and the dashboard must have somewhere to put it, and
they are marked so no panel silently reads zero for something the pipeline
cannot detect.

| Domain | Codes | Declared only |
|---|---|---|
| ingest | `VIDEO_FRAME_UNDECODABLE`, `TARGET_BELOW_RESOLUTION` | `VIDEO_FRAME_UNDECODABLE` |
| calibration / seat | `SEAT_UNCALIBRATED`, `SEAT_ASSOCIATION_AMBIGUOUS`, `ROLE_UNCERTAIN` | both of the last two |
| tracking | `TRACK_FRAGMENT`, `TRACK_IDENTITY_UNVERIFIED` | — |
| object | `OBJECT_PROPOSAL_PAPER`, `OBJECT_PROPOSAL_PHONE`, `DFINE_OBJECT_CONTEXT`, `OBJECT_AT_WRIST`, `OBJECT_AWAY_FROM_WRIST`, `OBJECT_TOO_LARGE`, `OBJECT_NO_WRIST_RESOLVED`, `OBJECT_STATIONARY_ON_DESK`, `OBJECT_HANDLING_SUSTAINED`, `OBJECT_HANDLING_RECURRENT` | `OBJECT_PROPOSAL_PHONE` |
| SAM 3 | `SAM3_SUPPORTED`, `SAM3_EQUIPMENT_CONTEXT`, `SAM3_NOT_CONFIRMED`, `SAM3_PHONE_NAMED`, `SAM3_MASK_UNSTABLE`, `SAM3_UNAVAILABLE`, `SAM3_NOT_RUN` | `SAM3_MASK_UNSTABLE` |
| orientation | `ORIENTATION_NOT_ASSESSABLE`, `ORIENTATION_AWAY_SUSTAINED`, `ORIENTATION_CHANGE_FREQUENT`, `ORIENTATION_DOWN_PROLONGED`, `ORIENTATION_TOWARD_NEIGHBOUR`, `TALKING_NOT_MEASURABLE` | `ORIENTATION_TOWARD_NEIGHBOUR` |
| hands | `HAND_OCCLUDED_LONG`, `HAND_BELOW_DESK_LINE`, `HAND_INTERACTION_RECURRENT` | `HAND_INTERACTION_RECURRENT` |
| presence | `PRESENCE_GAP`, `SEAT_VACANCY_SUSTAINED` | `SEAT_VACANCY_SUSTAINED` |
| policy | `POLICY_MATERIAL_PERMITTED` | yes (configuration exists, no exam profile loaded) |

### The three guardrails that exist because we got them wrong once

- **`SAM3_NOT_CONFIRMED`** — *not confirmed is not false.* SAM 3 missed known
  true positives on this corpus (frames 2 and 5 of the reference set). This
  code may never be rendered, counted or summarised as `clear`, `safe` or
  `no cheating`. Pinned by a test.
- **`ORIENTATION_NOT_ASSESSABLE`** — no baseline means not measured. Across the
  four runs **909 of 1,010 tracks never resolved a head baseline**. Folding
  those into "no signal" would report 909 clean candidates the system never
  looked at.
- **`OBJECT_TOO_LARGE`** — the size cap is per-camera. 0.12 is the 63rd
  percentile on 1280×720 and the 25th on 360×450. Open question Q7.

---

## 4. The state machine

```
no_action  →  context_observation  →  needs_better_view  →  review_candidate
                                                                    ↓
                                              human_confirmed / human_dismissed
```

| State | Means | Who writes it |
|---|---|---|
| `no_action` | nothing reliable was observed | machine |
| `context_observation` | observed and explained — paper on a desk, a keyboard, ordinary workstation activity | machine |
| `needs_better_view` | the system could not see well enough to say | machine |
| `review_candidate` | signals converged; a human should look | machine |
| `human_confirmed` | a reviewer watched it and confirmed a violation | **human only** |
| `human_dismissed` | a reviewer watched it and cleared it | **human only** |

`needs_better_view` deliberately outranks `context_observation`: not being able
to see is a more urgent operational fact than having seen something ordinary.

`fusion.assert_machine_state()` raises if automated code tries to write either
human state. Nothing skips from a detection to "cheating".

### What makes a `review_candidate`

Five conditions. All are evaluated, and **all are recorded whether they pass or
fail** — a card that cannot show its own arithmetic does not get shown.

| Condition | Test | Threshold provenance |
|---|---|---|
| `proposal_survives_geometry` | detections survived the wrist and size gates | measured |
| `sam3_supports_or_cannot_exclude` | ≥3 supported, **or** ≥2 sampled replies of which ≥67% supported, or a phone is SAM 3 phone-supported, or no adjudication available | count measured (133 of 136 on one man, 12_paper); the rate is an **additional** path for sampling-thinned tracks, never a replacement — see below |
| `associated_with_this_person` | paper: a wrist resolved in ≥25% of sightings; phone: the individual SAM 3 phone-supported frame is ≤0.75 person widths from that track's recorded wrist | paper threshold declared; per-frame phone association declared pending the evaluation pack |
| `lasts_or_recurs` | ≥5 s continuous, **or** ≥2 episodes totalling ≥5 s | measured (top false positive on 12_paper handled for 5.0 s) |
| `no_dominant_equipment_explanation` | <60% of adjudications named equipment | measured (04_talking 402/409 = 98%; 12_paper 48/184 = 26%) |

**The corroboration rate may only ever add passes.** `verdict.corroborated_enough()`
is the single implementation, called by both `verdict.py` and `fusion.py`.
Making the rate a replacement for the count was tried and reverted: SAM 3's
support rate on the two known 12_paper incidents is 5 of 19 (26%, track 118)
and 5 of 29 (17%, track 53), so a 67% rate test dropped both out of review,
along with 01_phone track 18. SAM 3 is a conservative referee on this corpus.
Pinned by `test_corroboration_rate_supplements_the_count_and_never_replaces_it`.

Plus two escalation shortcuts and one asymmetry:

- **A phone SAM 3 supports routes to review on its own — on the frame where it
  was supported.** The item is prohibited outright, so a second sense adds
  nothing; but the naming still has to land at *this* person's wrist in *that*
  frame. Track-level wrist coverage may not stand in for it.
- **Two independent behaviour modalities route to review.** Orientation, object
  and presence are the three; the chit detector and SAM 3 are **one** modality,
  not two, because they look at the same pixels for the same thing.
- **Object evidence routes to review alone; behaviour evidence does not.**
  Orientation alone is "turned away from their own baseline", which on this
  corpus is true of ordinary people looking at their own screens.

---

## 5. Object-specific handling

### Chit / paper

| State | Code | Route |
|---|---|---|
| paper on the desk, unmoving | `OBJECT_STATIONARY_ON_DESK` | context |
| paper near the candidate, no wrist resolved | `OBJECT_NO_WRIST_RESOLVED` | needs better view |
| paper at the wrist | `OBJECT_AT_WRIST` | context, on its own |
| handled continuously past threshold | `OBJECT_HANDLING_SUSTAINED` | argues for review |
| handled in separate episodes past threshold | `OBJECT_HANDLING_RECURRENT` | argues for review |
| exam permits this material | `POLICY_MATERIAL_PERMITTED` | demotes to context |

Persistence makes an object desk furniture; it does **not** clear a genuine
chit that happens to be lying still. `Policy.loose_paper_prohibited` is the
per-exam switch, and flipping it demotes paper handling to context — tested.

### Phone

The COCO path is gone. A phone reaches review only when SAM 3 phone-supports it
(`SAM3_PHONE_NAMED`) **and** the `associated_with_this_person` condition
passes on that same frame. Naming alone is not enough: a phone on a neighbouring
desk, or in an invigilator's hand, falls inside the crop cut for a track, and
routing on the name alone would make it evidence against whoever that crop
belonged to. When the object is named but not attributable the state is
`needs_better_view`.
Keyboard, mouse, monitor and water bottle are explicit hard negatives in the
prompt. When the object is too small or hidden the answer is
`TARGET_BELOW_RESOLUTION` / `needs_better_view`, never a negative finding.

---

## 6. Behaviour signals

| Behaviour | Code | Guardrail |
|---|---|---|
| hands hidden | `HAND_OCCLUDED_LONG` | observability warning; escalates only with an independent object signal |
| hands below desk | `HAND_BELOW_DESK_LINE` | the resting posture of most seated people |
| repeated reaching | `HAND_INTERACTION_RECURRENT` | **declared only** — needs pocket/bag zones; writing looks like this |
| head turned away | `ORIENTATION_AWAY_SUSTAINED` | orientation is not gaze; at 640×480 we may not say what was looked at |
| constant head movement | `ORIENTATION_CHANGE_FREQUENT` | fidgeting is normal; frequency alone may not escalate |
| head down | `ORIENTATION_DOWN_PROLONGED` | the posture of writing as much as of reading something hidden |
| looking at a neighbour | `ORIENTATION_TOWARD_NEIGHBOUR` | **declared only** — needs seat geometry |
| talking | `TALKING_NOT_MEASURABLE` | emitted on **every** track, so no panel can show a zero |
| absent and returned | `PRESENCE_GAP` + `TRACK_IDENTITY_UNVERIFIED` | track loss, occlusion and walking out of shot are indistinguishable here; this is not "left the seat" |
| seat empty | `SEAT_VACANCY_SUSTAINED` | **declared only** — no occupancy model exists |
| standing / walking | `ROLE_UNCERTAIN` | **declared only** — standing is not evidence of being an invigilator |

---

## 7. Measured results on the four completed runs

Produced by `tools/build_evidence_records.py`, from artifacts already on disk.
All four runs are reproducible from the committed repository: the two sample
files the tool needs (`samples_with_chits.jsonl`,
`samples_sam3_adjudicated.jsonl`) are tracked for every run. They were not, for
1503/1504/1505, until an audit of `7d2359f` found that only 12_paper could be
rebuilt from a fresh checkout.
No model was re-run; the gate was recomputed with the run's own config and the
tool **exits non-zero unless its recomputed kept-counts reproduce the run's
own** (it caught the newsclip's 0.20 size cap on the first attempt, which is
what the check is for).

| Run | Records | Tracks | `review_candidate` | `context` | `needs_better_view` |
|---|--:|--:|--:|--:|--:|
| 1501 / 12_paper | 1,817 | 236 | **2** (tracks 118, 53) | 13 | 221 |
| 1503 / newsclip_12 | 2,213 | 515 | 0 | 27 | 488 |
| 1504 / 04_talking | 41,695 | 92 | **1** (track 4) | 2 | 89 |
| 1505 / 03_mobile | 17,413 | 167 | **1** (track 1) | 0 | 166 |
| **total** | **63,138** | **1,010** | **4** | **42** | **964** |

Proposal funnel across all four:

| Stage | Count |
|---|--:|
| paper proposals | 5,717 |
| survived geometry gate | 2,619 |
| SAM 3 supported | 144 |
| SAM 3 named equipment | 455 |
| SAM 3 could not confirm | 1,967 |
| SAM 3 unavailable | 0 |
| SAM 3 named a phone | 2 |

**What lands where it should:** 12_paper's two review candidates are tracks 118
and 53 — the one man who took the paper, split across two track IDs. 03_mobile's
is track 1, carrying the two `SAM3_PHONE_NAMED` records (the red phone at
172.0 s), and it is the highest-priority entry in the whole corpus.

**What does not:** 04_talking track 4 passes all five conditions with 7 SAM 3
supports against 22 equipment calls. It is very likely a false positive and
there is currently **no way to prove that either way**. The newsclip dropped
from 4 candidates to 0 once the recurrence clause required a time floor — those
four were 1.0-second sampler bursts.

**None of the above is precision or recall.** These are counts of what the
system did. See §9.

---

## 8. Dashboard specification

Five panels, built around evidence packages rather than detector counts.

1. **Review queue** — one card per `review_candidate`: clip with boxes, SAM 3
   overlay, wrist/pose overlay, seat boundary, object timeline; the full reason
   chain; the data-quality warnings; and the four reviewer actions (confirm,
   dismiss, need better view, annotate).
2. **Seat timeline** — occupied / standing / empty / staff-like / not
   assessable. **Blocked**: no occupancy model. The panel must say so rather
   than render an empty timeline.
3. **Object evidence** — proposals, SAM 3 verdicts, equipment context, hand
   association, recurrence.
4. **Behaviour evidence** — orientation, hand visibility, posture, presence
   gaps, observability.
5. **System quality** — API health, latency, coverage, tracking quality,
   model disagreement, human outcomes.

### Four things the dashboard must never do

1. Display a raw detection total as a cheating count or a cheating rate.
2. Rank by detection count — that ranks the busiest tracker.
3. Hide abstentions. `needs_better_view` is 964 of 1,010 tracks and that number
   is the honest headline, not something to bury.
4. Show a probability of cheating. There is no such number.

---

## 9. The evaluation gate

**No accuracy figure may be published before this exists.** The current
12_paper-versus-04_talking comparison is a smoke test, not a measurement.

Required: an independently annotated evaluation pack covering 04_talking,
12_paper, true phones, permitted paper and notebooks, keyboards/mice/monitors,
bags, desks, white clothing, occluded hands, staff moving, candidates standing
and leaving, and genuine non-events. Labels for object presence, object
handling, seat association, equipment context and review-worthiness — **not**
inferred intent.

Split by **video and camera**, never by random frame. Random-frame splits leak
near-identical tracking bursts between train and test.

Then compare three configurations — Roboflow only; Roboflow + SAM 3; Roboflow +
SAM 3 + temporal/pose fusion — and only then publish precision, recall,
abstention rate, reviewer-confirmation rate and stage loss, separately for
phone and chit.

Every record needed for this is already on disk. `evidence_records.jsonl`
preserves rejections precisely so that recall against a future label set is
computable without re-running anything.

---

## 10. What is not built

Stated plainly so nothing here is mistaken for finished. Confirmed by an
independent audit of revision `7d2359f` in an isolated checkout.

| Gap | Consequence |
|---|---|
| **No fine-tuned phone detector** | `OBJECT_PROPOSAL_PHONE` *is* emitted now — D-FINE's stock COCO `cell phone` class proposes and SAM 3 verifies (332 proposals in run 1511/01_phone). What is missing is a phone detector trained on this corpus: COCO's class fires on monitors, dark bezels and desk edges, so recall rests on a class that was never tuned for a 26×21 px handset on overhead CCTV. Blocked: `handphone-dataset-2` has **0 trained versions** — it is a dataset, not a deployable model. The dismissed mouse/monitor/bezel cards from review are the hard-negative set that would fix it |
| **Sub-threshold phones are dropped** | a proposal below the D-FINE confidence floor is never adjudicated, even when it recurs in the same place at the same wrist across many frames. The floor is a call-volume cap, not a judgement, so spatially-consistent low-score recurrence should open an episode anyway. Not implemented |
| **No SAM 3 result cache** | identical crops are re-sent across reruns. Keying by crop hash + prompt + model version would cut repeat-run cost to near zero. Not implemented |
| No temporal SAM 3 window | verification is per-crop; `SAM3_MASK_UNSTABLE` is declared, never computed |
| No object identity across people | object passing / handoff is undetectable |
| **Seat calibration exists but is unapproved** | `calib/` holds 7 profiles covering 58 seats. Only `cam12` is `approved`; the other 6 are machine-proposed `draft`s from `tools/propose_calibration.py`. `seat_association.associate()` returns nothing for a draft, so `run_person_timeline.py` falls back to `seat_state="unattributed"` for every person — which is why all 1,010 tracks carry `SEAT_UNCALIBRATED`. **Zero blocking validation issues stand on any draft**; each needs a named reviewer and a human look at the polygons. No code change. |
| No occupancy model | get up, leave, return, seat replacement and seat swap are all undetectable; `SEAT_VACANCY_SUSTAINED` is declared |
| No staff/aisle model | `ROLE_UNCERTAIN` exists in the vocabulary and **is never emitted**; candidate and invigilator are indistinguishable |
| No hand occlusion or action stage | `HAND_OCCLUDED_LONG` and `HAND_INTERACTION_RECURRENT` exist and **are never emitted** |
| No neighbour geometry | `ORIENTATION_TOWARD_NEIGHBOUR`, reciprocity and mutual attention are declared only |
| No talking detection, and none planned on this footage | `TALKING_NOT_MEASURABLE` on every track is the whole of it |
| No re-identification | a person who leaves and returns is not verified to be the same person |
| No local SAM 3 | `pipeline/sam3_local.py` has never run; the working path is a network call, conflicting with PS 2 (Q17/Q23) |
| **Dashboard not rebuilt** | `artifacts/dashboard.html` still renders the older `pipeline/verdict.py` ladder. Its buttons change browser HTML only — **no reviewer decision is persisted**. The page now says so on its face |
| No ground truth | zero precision/recall figures exist for this system |

---

## 11. Rules still needed for people moving in the hall

None of these are implemented. Each is listed with the machinery it needs and
the outcome it must produce, so that no future implementation quietly picks a
more accusatory one.

| Situation | Required machinery | Safe outcome |
|---|---|---|
| Candidate raises hand, invigilator approaches | seat ownership + staff/aisle route + proximity episode | legitimate interaction context |
| Invigilator standing throughout | staff-zone and route model | must never create candidate evidence |
| Candidate stands but stays at their seat | occupancy state + body position | `standing_at_seat`, never "left" |
| Candidate leaves and returns | seat-empty interval + reappearance confidence | procedural review, never an identity claim |
| New person occupies the seat | before/after seat appearance comparison | `seat_occupant_changed`, for review |
| Two candidates swap seats | two seat transitions in one window | high-priority review candidate |
| Two people orient toward each other | calibrated neighbour graph + simultaneous direction episodes | mutual-attention review context |
| Object passed between people | object track: A wrist → gap → B wrist | review only when continuity is credible |
| Several people gather at one seat | seat map + crowd proximity | crowd/assistance review context |
| Camera covered or blind spot created | frame-observability + persistent occlusion state | `not_assessable`, logged separately |
| Medical or accessibility movement | policy/accommodation configuration | never compared against a generic "normal posture" |
| Legitimate materials | per-exam permitted-items policy | detected material stays context |
| Camera cut, freeze, time gap | ingest continuity check | no behaviour inference across the gap |

---

## 12. The acceptance test is a scenario pack

Unit tests pin invariants; they cannot show the system behaves correctly on
footage. The end-to-end gate is a scenario pack. Every scenario carries video
or frames, truth labels, expected reason codes, expected machine state, and
expected reviewer display.

**Coverage required:**

- *Objects*: real phone; mouse; keyboard; screen bezel; calculator; seat
  placard; permitted paper/notebook; prohibited chit; paper stationary on a
  desk; paper handled and moved.
- *Hands*: hand under desk; occluded hand; ordinary writing; bag/pocket
  interaction.
- *People*: candidate standing, leaving, returning; invigilator standing and
  walking; candidate/invigilator overlap.
- *Attention*: brief look away; repeated look; neighbour reciprocity; ordinary
  screen-facing.
- *Interaction*: object handoff; seat swap; crowding around one seat.
- *Degradation*: dark footage; glare; blur; dropped frames; SAM 3 outage;
  tracking split; uncertain ownership.

**Every case asserts the same chain:**

```
raw proposal retained
  → correct reason codes
    → correct machine state
      → no automatic human verdict
        → uncertainty visible in the output
          → reviewer action persisted
```

The last link cannot be asserted yet: nothing persists reviewer actions.

---

## 13. Engineering order

1. Independent Roboflow phone proposal adapter, with tests. **Blocked on
   sourcing a trained phone model** — the identified dataset has no deployable
   version.
2. Seat calibration, occupancy, and candidate/staff/aisle classification.
   Unblocks the whole of §11 and is the highest-value item that needs no GPU.
3. Object temporal association and SAM 3 temporal stability.
4. Hand occlusion/action, neighbour reciprocity, handoff, crowding,
   leave/return.
5. Replace the legacy dashboard with one that reads `EvidenceRecord` and
   `TrackOutcome`, persists append-only reviews, and never says "cleared" for
   an unassessable track.
6. Hand-label the scenario pack, then publish phone/chit precision, recall,
   abstention, review-routing recall, false-positive reasons and reviewer
   agreement — separately, per §9.

---

## 14. Related documents

- `docs/CLASSIFICATION_AND_OPEN_QUESTIONS.md` — behaviours B1–B17, open
  questions Q1–Q24, threshold provenance
- `docs/SESSION_LOG_2026-08-22.md` — how the SAM 3 referee was built and what
  it measured
- `docs/ARCHITECTURE.md` — the pipeline stages
