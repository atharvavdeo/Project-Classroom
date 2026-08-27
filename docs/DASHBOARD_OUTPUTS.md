# Dashboard orientation outputs

**Companion to `OUTPUT_CONTRACT.md` v1.0.0. Additive.** Nothing here changes a
machine state, a reason code, a flag or a review queue. These are presentation
classifications carried on `PersonProfile.orientation_labels` and
`PersonProfile.orientation_events`, and a client that ignores both fields sees
exactly the system that existed before them.

Produced by `pipeline/person_timeline.py`: `pitch_state()` per frame,
`_orientation_outputs()` per track.

---

## 1. The four states

| Label | Band | Classification | Dashboard action | Escalates? |
|---|---|---|---|---|
| `looking_up` | `pitch < −0.20` | `observed_only` | `display_only` | **Never**, at any duration |
| `shallow_down` | `+0.20 … +0.45` | `observed_only` | `display_only` | **Never**, at any duration |
| `deep_down_sustained` | `pitch > +0.45`, PTS-continuous **≥15 s** | `review_context` | `show_review_context` | Context only, never alone |
| `turned_and_held` | torso-relative yaw ≥0.30 from that subject's own baseline, held **≥4 s** | `context_requires_second_modality` | `show_only_with_second_modality` | Only alongside a second modality |

The asymmetry is the design. Twenty minutes of head-down-at-paper produces
`shallow_down` and nothing else.

Every label carries its own `guardrail` string. Render it with the label — a
card that cannot show why it is allowed to exist does not get shown.

---

## 2. What these actually emit on the corpus

Measured by replaying `_orientation_outputs()` over the stored joints of runs
**1511/01_phone**, **1510/12_paper** and **1509/04_talking** — 11,151 samples,
412 tracks, 250 ms sampling. Reproduce with the pooled counts below; these are
what the system reported, not what was true.

### Frame bands

| Band | Samples | Share of all |
|---|---|---|
| `not_assessable` | 6,237 | **55.9%** |
| `neutral` | 2,558 | 22.9% |
| `shallow_down` | 1,187 | 10.6% |
| `looking_up` | 639 | 5.7% |
| `deep_down` (raw frames) | 530 | 4.8% |

**The largest band is "we could not tell."** 55.9% of samples have no
resolvable pitch. Any orientation panel must show that denominator on its face;
a count of `looking_up` without it is a lie by omission.

### Emitted labels and events

| Label | 01_phone | 12_paper | 04_talking | Total |
|---|---|---|---|---|
| `looking_up` (tracks) | 8 | 18 | 7 | 33 |
| `shallow_down` (tracks) | 16 | 57 | 24 | 97 |
| `turned_and_held` (events) | 2 | 4 | 5 | **11** |
| `deep_down_sustained` (events) | 0 | 0 | 0 | **0** |

The 11 `turned_and_held` events in full:

| Video | Track | Direction | Span | Duration |
|---|---|---|---|---|
| 01_phone | 1 | right | 15.75–19.75 s | 4.0 s |
| 01_phone | 25 | right | 75.0–80.0 s | 5.0 s |
| 12_paper | 16 | left | 39.75–44.75 s | 5.0 s |
| 12_paper | 16 | right | 74.5–79.0 s | 4.5 s |
| 12_paper | 18 | left | 1.25–12.75 s | 11.5 s |
| 12_paper | 18 | right | 25.0–33.0 s | 8.0 s |
| 04_talking | 4 | left | 4.0–9.5 s | 5.5 s |
| 04_talking | 4 | right | 90.5–94.5 s | 4.0 s |
| 04_talking | 4 | left | 130.0–140.75 s | 10.8 s |
| 04_talking | 1 | right | 8.5–12.5 s | 4.0 s |
| 04_talking | 7 | left | 3.0–8.75 s | 5.8 s |

---

## 3. `deep_down_sustained` is declared, not validated

**It fires zero times on this corpus.** Of 186 PTS-continuous `deep_down`
spans, **0 reach the 15 s rule**. The longest span in each video:

| Video | Spans | Longest | Reaching 15 s |
|---|---|---|---|
| 01_phone | 37 | 5.2 s | 0 |
| 12_paper | 120 | 5.5 s | 0 |
| 04_talking | 29 | 6.0 s | 0 |

The threshold is therefore a **declared** policy value, in the sense
`OUTPUT_CONTRACT.md` §7 uses the word — chosen, not measured. It is kept
because the rule it encodes is correct (posture is not intent, so only an
implausibly long hold is worth a reviewer's time), but the dashboard must not
present it as a validated detector. Until the §9 evaluation pack exists there
is no evidence about what duration separates a real event from a slouch.

Implication for the dashboard: **build the panel to render zero of these.** An
empty state is the expected state today.

---

## 4. What `turned_and_held` is, and is not

It is built on `yaw_relative_to_torso`, which `head_pose.py` computes as
`yaw × shoulder squareness` and which stage 14 previously discarded.

Recomputed over the same three runs — 5,112 yaw-resolved samples:

| | Count | Share |
|---|---|---|
| Raw yaw pinned at the ±1.5 clamp | 1,471 | 28.8% |
| Raw yaw at the one-ear constant ±1.0 | 969 | 19.0% |
| **Raw yaw carrying no magnitude** | **2,440** | **47.7%** |
| Torso-relative resolved | 4,350 | 85.1% of yaw-resolved |
| Torso-relative at the clamp | 6 | **0.1%** |

So the scale genuinely stops pinning. But **on 43.2% of torso-resolved frames
the underlying yaw was one of those two constants**, and there the torso value
varies with shoulder geometry — whole-body rotation — not with how far the neck
turned.

That is why the label is a *departure from the subject's own baseline held over
time*, never an angle. Three consequences the dashboard must honour:

1. **Never print a degree figure** for this. There isn't one.
2. **Never call it gaze.** At a median interocular distance of 4.7–10.1 px on
   this corpus there is no iris and no gaze. `OUTPUT_CONTRACT.md` line 205
   already forbids saying what was looked at.
3. **`direction` is the subject's own left/right**, derived from the median of
   the span. It names a side, not a neighbour — seat attribution is
   uncalibrated (all tracks carry `SEAT_UNCALIBRATED`), so the system cannot
   say who was on that side, or whether the turn was toward a person at all.

An earlier version of this rule, using an absolute threshold instead of a
per-subject baseline, flagged resting posture: video 04 reported 45 `yaw_right`
of which 9 had baseline equal to peak. The baseline requirement is what took
that from 53 events to 5.

### The coverage ceiling on `turned_and_held`

A track needs `min_samples_for_baseline` (8) resolved torso-yaw samples before
this label can be evaluated at all. On run **1512/12_paper** that is **21 of
236 tracks — 8.9%**.

So `turned_and_held` speaks for under a tenth of tracks, and its silence on the
other 91% means *not measurable*, never *did not happen*. A dashboard that
shows this label must show that denominator beside it, for the same reason
§8.3 of the contract forbids hiding abstentions.

---

## 5. Payload shape

`orientation_labels` — one row per label per track:

```json
{"label": "shallow_down", "classification": "observed_only",
 "samples": 143, "dashboard_action": "display_only",
 "guardrail": "This band never escalates on duration alone."}
```

`orientation_events` — one row per span, with timings for the clip scrubber:

```json
{"label": "turned_and_held",
 "classification": "context_requires_second_modality",
 "direction": "right", "from_ms": 15750.0, "to_ms": 19750.0,
 "duration_ms": 4000.0,
 "dashboard_action": "show_only_with_second_modality",
 "guardrail": "Torso-relative turn is not gaze and never routes to review on its own."}
```

`dashboard_action` is the only field a client needs to switch on. The four
values map one-to-one onto the four states in §1.

---

## 6. Rules for whoever builds the panel

Inherited from `OUTPUT_CONTRACT.md` §8, plus three specific to this layer:

1. Do not aggregate these into a per-person score. Summing `shallow_down`
   samples produces a number that means "sat at a desk".
2. Do not sort the queue by them. `turned_and_held` is
   `context_requires_second_modality`; sorting by it makes it the primary
   signal, which is exactly what the classification forbids.
3. Show `not_assessable` (55.9%) wherever any band is shown.

And the four standing prohibitions still apply in full: no raw total as a
cheating count, no ranking by detection count, no hiding abstentions, no
probability of cheating.
