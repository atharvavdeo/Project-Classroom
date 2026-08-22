# From detections to a verdict: classification, thresholds, and what is still unsolved

*Team DataDynamo — Drishti AI Hackathon 2026, Problem Statement 2*
*Written 2026-08-22. No code in this document. It is a specification and a map of gaps.*

---

## 0. The problem in one paragraph

The pipeline currently produces two kinds of output: **detections** (a box, a class,
a score) and **rejections** (the same thing, discarded by a gate). Neither is a
finding. A reviewer opening `chit_evidence.json` sees `kept_detections: 393` and
has no way to know whether that means one person cheated for a minute or four
hundred people rustled paper. There is no rung on the ladder between "the
detector fired" and "this is cheating", and that missing rung is the single
biggest gap between what we have built and what an invigilator could use.

This document defines that ladder, states the thresholds, and then maps
everything we know we cannot yet do.

---

## 1. What actually happens in an exam hall

Before any threshold, the behaviours worth naming. This is the domain model
everything else is measured against. The venue in our corpus is a **computer-based
test centre** — candidates at monitors in numbered carrels, not desks in rows —
which changes what is normal and what is not.

| # | Behaviour | Is it suspicious? | What makes it so |
|--:|---|---|---|
| B1 | Reading one's own answer sheet / rough sheet | **No** — universal | Paper is expected at every seat |
| B2 | Looking at own monitor, typing, using mouse | **No** — the task itself | |
| B3 | Stretching, yawning, rubbing eyes, drinking water | **No** | Brief, no object, no direction |
| B4 | Turning head briefly (<1.5 s) in any direction | **No** | Everyone does it constantly |
| B5 | Invigilator walking the aisle, standing over a candidate | **No** — expected | Must not be flagged as a candidate at all |
| B6 | Candidate raising hand, invigilator responds | **No** | Legitimate interaction |
| B7 | Leaving seat for a washroom break and returning | **Procedural** | Legitimate but must be *logged* |
| B8 | **Sustained** head turn toward a neighbour's screen | **Yes** | Direction + duration + a target |
| B9 | **Repeated** brief glances in the same direction | **Yes** | Rhythm is the signal, not any one glance |
| B10 | Two candidates turning toward each other in the same window | **Yes, strongly** | Reciprocity is very hard to explain innocently |
| B11 | Talking — mouth movement plus mutual orientation | **Yes** | The named incident in video 04 |
| B12 | Handling a small object at the hands, below desk line | **Yes** | Concealment is the discriminator |
| B13 | Passing an object between two people | **Yes, strongly** | Object continuity across two tracks |
| B14 | A phone: any handling at all | **Yes, decisive if verified** | Prohibited outright |
| B15 | A person leaving and a **different** person occupying the seat | **Yes, decisive** | Impersonation |
| B16 | Two candidates swapping seats | **Yes, decisive** | The named incident in the exchanged-seats video |
| B17 | Crowding — several people converging on one seat | **Yes** | The named incident in video 05 |

**The three we currently measure at all: B8, B9, B12.** Everything else is either
unmeasured or measured and thrown away.

---

## 2. The classification ladder

Five rungs. A finding moves up only when the evidence for the rung below is
already satisfied. The point of the ladder is that **each rung names who is
accountable for the judgement** — the last two are human, always.

| Rung | Name | Who decides | What it means | Shown in dashboard as |
|--:|---|---|---|---|
| 0 | `observation` | Machine | A measurement was taken. Head at yaw −0.4; a box scored 0.61. | Not shown. Data layer only. |
| 1 | `signal` | Machine | An observation that departs from this person's own baseline, past a stated threshold. | Not shown alone. |
| 2 | `indicator` | Machine | A signal that survives every gate: duration, direction, corroboration, and a plausibility check against the room. | **Amber card, "worth a look"** |
| 3 | `incident_candidate` | Machine, ranked | Two or more *independent* indicators coincide in time on the same person. | **Red card, "review this"** |
| 4 | `confirmed_incident` | **Human reviewer** | A person watched the clip and agreed. | **Confirmed, with reviewer name + timestamp** |

**Nothing in this system may ever write rung 4.** That is not modesty, it is the
design: an automated system that declares a candidate a cheat creates a liability
we cannot discharge, and our own measurements show why — the strongest object
signal we have fires on seat placards and shirt fabric.

### The rule that makes rung 3 meaningful

> **Independence.** Two indicators count as independent only if they come from
> *different sensing modalities*. Two chit detections 200 ms apart are one
> indicator, not two. A sustained head turn **plus** an object at the hands is
> two. A head turn plus a second head turn is one.

Without this rule, rung 3 is just "the detector fired a lot", which is exactly
the failure mode we already measured: 1,817 chit detections on one 88-second
video, of which 972 were nowhere near a hand.

---

## 3. Thresholds, and where each number comes from

Every threshold below is either **measured** on our corpus or **declared** as a
policy decision. None is invented. Where a number is a guess, it says so.

### 3.1 Head orientation → `sustained_attention_away` (B8)

| Parameter | Value | Basis |
|---|--:|---|
| Departure from own baseline | **≥ 0.30** normalised yaw | Measured: `HeadPoseConfig.min_departure_from_baseline`, tuned when an absolute threshold flagged 45 resting postures on video 04 |
| Minimum duration | **≥ 1500 ms** | Declared. B4 says brief turns are universal |
| Merge gap | **2.5 × sample interval** | Measured: a fixed 800 ms gap under 1 Hz sampling produced 1 event from 379 observations |
| Direction consistency | same sign throughout | Declared |
| **Escalates to** | `indicator` | |

### 3.2 Repetition → `frequent_orientation_change` (B9)

| Parameter | Value | Basis |
|---|--:|---|
| Episodes in a 60 s window | **≥ 4** | Declared. Needs validation against real footage |
| Same direction | ≥ 75% of episodes | Declared |
| **Escalates to** | `indicator` | |

### 3.3 Object at the hands → `handled_object` (B12)

| Parameter | Value | Basis |
|---|--:|---|
| Detector confidence | ≥ 0.40 | Measured — **and it does almost no work.** True positives run 0.52–0.74; the worst false positive (a photo on a monitor) is 0.65 |
| Size, fraction of person box | ≤ 0.12 | Measured on 1280×720. **Does not transfer** — see Q7 |
| Distance to nearest wrist | **≤ 0.35 person-widths** | Measured. Does the real work *on 12_paper*: removed 729 of 1,817 and moved the true subject to rank 1. **Fails on 04_talking — see below** |
| Overlap with a confidently-detected tool | **not implemented** | Required, see Q20 |
| **Escalates to** | `indicator` | |

> **The wrist gate is not general, and run 1504 proved it.** On 12_paper the
> false positives were shirts, partitions and monitors — all *away from* the
> wrists — so geometry removed them. On 04_talking they are **keyboards, mice
> and desk objects, which sit exactly where a wrist is**. The gate cannot help,
> because geometrically those detections are correct. Result: 54% of detections
> survived and four people were ranked as handling paper for 64–132 s of a 143 s
> recording. In a computer-based test centre, hands rest on a keyboard for three
> hours, so **this is the common case, not the edge case.**

### 3.4 What each object class is worth

The taxonomy must stop pretending all objects are equal.

| Class | Max rung reachable alone | Why |
|---|---|---|
| `phone` (verified by a human) | **3** — decisive | Prohibited outright |
| `phone` (detector only) | **1** | Measured: fires on seat placards at 0.30–0.49 |
| `paper_like_object` at hands | **2** | Cannot be told from an answer sheet |
| `book_or_notebook_like_object` | **1** | Stationery is expected |
| `monitor`, `keyboard`, `mouse`, `bag` | **0** | Furniture. Context only |

### 3.5 The escalation table

| Combination, same person, overlapping in time | Rung | Dashboard |
|---|:--:|---|
| Object at hands only | 2 | Amber |
| Sustained head turn only | 2 | Amber |
| Sustained head turn **+** object at hands | **3** | Red |
| Sustained head turn **+** neighbour turns back (B10) | **3** | Red |
| Repeated glances **+** object at hands | **3** | Red |
| Any verified phone handling | **3** | Red |
| Identity discontinuity at a seat (B15/B16) | **3** | Red |
| Three or more independent indicators | **3, top of queue** | Red, pinned |

---

## 4. Open questions — the map

Everything we cannot currently do. **Difficulty** is engineering effort;
**blocker** is what actually stands in the way.

### 4.1 Seat occupancy and identity — the whole B7/B15/B16 family

This is the gap you named, and it is the largest one. **We currently have no
concept of a seat being occupied.**

| # | Question | Difficulty | Blocker | Approach |
|--:|---|---|---|---|
| Q1 | Is a seat occupied at time *t*? | **Low** | Nothing — just unbuilt | Seat polygon from calibration + person-box overlap, sampled continuously. Gives an occupancy timeline per seat. |
| Q2 | Did the occupant **leave**? | **Low** | Depends on Q1 | Occupancy false for > N s. Already have `absences` in `PersonProfile`, but keyed to *tracks*, not seats — which is why it reports 0 |
| Q3 | Did the **same** person return? | **High** | **Re-identification does not exist** | Appearance embedding over the torso crop. At 52–96 px person width this is marginal. Fallback: clothing colour histogram — crude but survives low resolution |
| Q4 | Did a **different** person take the seat? (B15, impersonation) | **High** | Same as Q3 | Same-seat, before/after appearance comparison. Deliberately a *lower* bar than full re-ID: we only need "same or not", at one seat, across one gap |
| Q5 | Did two candidates **swap** seats? (B16) | **High** | Q3 + Q4 | Two seats whose occupants change in the same window, and whose appearance signatures cross over |
| Q6 | Is this person a **candidate or an invigilator**? (B5) | **Medium** | Calibration | A person never inside any seat polygon, moving along aisles, is staff. Cheap and high value — stops flagging the invigilator |

**Why Q3/Q4 are the hinge:** on 12_paper one man became tracks 118 and 53. On the
news clip, ~22 people became **515 tracks**. Until identity persists, "did the
person at seat 12 change" is unanswerable, and B15/B16 — the two most decisive
behaviours in the whole domain model — stay unreachable.

### 4.2 Classification and thresholds

| # | Question | Difficulty | Blocker | Approach |
|--:|---|---|---|---|
| Q7 | Why does the size gate not transfer between resolutions? | **Low** | Design error, ours | 0.12 of the person box is p63 on 1280×720 and p25 on 360×450. Normalise against **torso width** (already computed) or against an estimated real-world size, not the box |
| Q8 | What is the right operating point — how many cards per 3-hour video? | **Low effort, needs a decision** | **Nobody has stated it** | This single number sets every threshold. 10 cards/hour is triage; 200 is noise |
| Q9 | Are our declared thresholds (1500 ms, 4 episodes) right? | **Medium** | No ground truth | Cannot be answered without Q13 |
| Q10 | How do we score *reciprocity* (B10)? | **Medium** | Nothing — unbuilt | Two tracks, overlapping windows, yaw vectors pointing at each other's boxes. Strongest available signal and completely unexploited |
| Q11 | Can we detect talking (B11)? | **High** | Resolution | Mouth keypoints are not resolvable at our scale. Proxy: sustained mutual orientation (Q10) + close proximity. Video 04 is the test case |
| Q12 | Object passed between two people (B13)? | **High** | Q3 + object tracking | Object appears at A's hands, disappears, appears at B's hands within N seconds |
| Q20 | How do we stop the chit model firing on **keyboards and mice**? | **Low** | Nothing — now unblocked | D-FINE reports `keyboard`/`mouse` with boxes since the box-carrying re-run. Suppress any chit detection that overlaps one. This is the single cheapest fix on this list |
| Q21 | How do we require the object to **move**? | **Medium** | Unbuilt | A keyboard is static relative to the desk; a chit being read moves with the hand. Needs per-detection association across frames, which we do not do for objects |
| Q22 | How do we use **absence** as evidence? | **Low** | Unbuilt | A keyboard is never absent. An object present in 100% of a person's sightings is furniture, whatever its class. `persistent_object_beside_seat` does this for D-FINE but is not applied to the chit channel |

### 4.3 Evaluation — the one that makes every other number real

| # | Question | Difficulty | Blocker | Approach |
|--:|---|---|---|---|
| Q13 | **What is actually true in our videos?** | **Low effort, high value** | Nobody has done it | Hand-label the ~6 known incidents as (time range, person, behaviour). A day's work. Gives the first precision/recall number this project has ever had |
| Q14 | How do we measure the detector separately from the rules? | **Medium** | Q13 | Two evaluations: detector mAP on labelled boxes; rule precision on labelled incidents. Conflating them hides which half is broken |
| Q15 | What is the false-positive rate on an *innocent* hall? | **Medium** | No clean footage | Every video we have is named for an incident. We have no negative control, so we cannot state a false-alarm rate |

### 4.4 Deployment reality

| # | Question | Difficulty | Blocker | Approach |
|--:|---|---|---|---|
| Q16 | Does any of this hold over 3 hours? | **Medium** | Untested | Longest run is 143 s. Baselines improve with length; tracking degrades. Videos 02, 05, 06 (2.14 h) never run |
| Q17 | Offline requirement vs hosted chit model | **Medium** | Architectural | `chit-paper-new/2` is an HTTP call. PS 2 requires offline. Must export weights and run locally before submission |
| Q18 | Throughput | **Low** | Measured | 12_paper: 88 s of video ≈ 10 min on an RTX 3050. A 3-hour video ≈ 20 h. Needs batching or a bigger card |
| Q19 | Multi-camera / same person across cameras | **High** | Q3, and unbuilt | Out of scope for the hackathon; name it so it is not assumed |

---

## 5. The dashboard

You are right that the deliverable is not a JSON file. What the reviewer needs,
in priority order.

### 5.1 The review queue — the primary screen

One card per `incident_candidate`, ranked. Never a list of detections.

| Element | Content |
|---|---|
| Thumbnail | The single frame with the strongest evidence, boxes drawn |
| Who | Seat number if calibrated, else "unattributed person #118" — **and it must say which** |
| When | `00:34:12 – 00:35:08`, click to seek |
| Why | *"Turned toward the neighbour for 4.2 s, twice, while holding a small pale object at the hands."* Plain sentences, not field names |
| Evidence strength | Which independent indicators fired — as chips, not a single fabricated confidence score |
| Actions | **Confirm** / **Dismiss** / **Need better view** — this is how rung 4 gets written |

### 5.2 The seat map

A floor plan of the hall. Each seat coloured by its highest rung. Click a seat to
get that seat's whole timeline: occupied, empty, occupant changed. **This is the
screen that makes B7/B15/B16 legible**, and it is blocked on Q1–Q6.

### 5.3 The timeline

One row per seat, time along the x-axis. Occupancy as a bar, indicators as marks.
A reviewer scanning this sees the *shape* of a session — including the empty
stretches that mean someone left.

### 5.4 What the dashboard must never do

- Show a single "cheating probability" percentage. We cannot compute one honestly.
- Rank by detection count. That ranks the busiest tracker, not the most suspicious person.
- Hide the abstentions. If a person's head was never resolvable, the card says so.
- Present `unattributed` as if it were a seat number.

---

## 6. Honest status

| | |
|---|---|
| Behaviours in the domain model | 17 |
| Behaviours we measure at all | **3** (B8, B9, B12) |
| Of those, measured *reliably* | **0** — B12 fails on keyboards (Q20), B8/B9 resolve for a minority |
| Behaviours reaching rung 2 today | 3 |
| Behaviours reaching rung 3 today | **0** — the escalation logic in §3.5 is specified here, not implemented |
| Ground-truth labels in existence | **0** |
| Precision / recall figures | **None** |
| Videos in corpus | 8 |
| Videos ever run end to end | 4 |

**The single highest-value next step is Q13** — hand-labelling the known
incidents. It is a day of work, needs no GPU, and converts every table in this
document from an assertion into a measurement.

**The second is Q20** — suppressing chit detections that overlap a keyboard or
mouse. Low effort, newly unblocked by the box-carrying re-run, and without it
every result on a computer-based test centre is dominated by people typing.

**The third is Q1/Q2/Q6** — seat occupancy. Low difficulty, already has the
calibration machinery, and unlocks the entire B7/B15/B16 family plus stops us
flagging invigilators.
