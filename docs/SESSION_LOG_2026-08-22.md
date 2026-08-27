# Session log — 2026-08-22

*Team DataDynamo — Drishti AI Hackathon 2026, Problem Statement 2*

End-to-end record of one working session: what was asked, what was built, what
was measured, what broke, and what was got wrong. Written so the reasoning
survives, not only the code.

---

## 1. Starting point

Carried in from the previous session:

- Run **1501** as the reference: `12_paper` and `01_phone`, continuous per-person
  timelines at 4 Hz, 99%+ coverage, AlphaPose skeletons, D-FINE person boxes.
- A known defect: **object attachments held bare class-name strings** with no box
  and no confidence, so the annotated videos drew no objects and no flag could be
  audited without re-running the detector. The fix was committed as `4b8a564`
  but had never actually run.
- Object flags known to be untrustworthy: D-FINE calls the numbered **seat
  placards** on the wall `cell phone` at 0.30–0.49, against 0.933 for the one
  phone verified by eye.

---

## 2. Evaluating a purpose-trained chit detector

### 2.1 The question

Four Roboflow Universe datasets were proposed for chit / book / phone detection,
with a view to training a small YOLO. My fetcher was **403'd by Roboflow on all
four URLs**, so I could not read image counts, licences, or samples.

My assessment at that point — that dataset size was not the risk, and that the
real risk was **object scale** (our verified phone is ~30×20 px; these datasets
are phone-camera close-ups) — was correct in direction but **too pessimistic in
magnitude**. I estimated mAP50 of 0.15–0.35 on our footage. That was wrong.

### 2.2 The user's own test, and what it did and did not show

Nine screenshots from our recordings were run through the model and shared with
their outputs. The Roboflow render has a **blank background**, so the boxes alone
prove nothing. I paired inputs to outputs by aspect ratio (exact to 3 decimals on
all nine) and composited the boxes back onto the real frames.

| # | Scene | Output | Verdict |
|--:|---|---|---|
| 1 | Striped shirt, crumpled white paper in hands | `chit-paper 74%` | Hit |
| 2 | Green shirt, seat 61, raising a card | `chit-paper 56%` | Hit |
| 3 | Man holding a **phone** | `chit-paper 52%` | Localised right, **class wrong** |
| 4 | **Placard wall — 19, 20, 21** | *nothing* | **Correct negative** |
| 5 | Striped shirt, hands + paper | `chit-paper 72%` | Hit |
| 6 | Seat 28, sheets clearly on the desk | *nothing* | Miss (or correct — they are legitimate) |
| 7 | Hand near monitor, dark object | `52%` | Ambiguous |
| 8 | Orange packet + paper on desk | `26/22/28%` | Weak, triplicated |
| 9 | Held white sheet, wide view | `68%` | Hit on an **answer sheet** |

**Frame 4 is the result that mattered**: the placards that poison D-FINE produce
nothing here.

**What the test could not show:** every frame was a positive, tightly framed on
someone already handling paper. It measured recall. It could not measure
precision.

### 2.3 Integration

Model resolved via the Roboflow API to `chit-paper-new/2` — **598 real images
(1,294 after 3× augmentation), single class `chit-paper`, CC BY 4.0**. I used the
model endpoint directly rather than a workflow: fewer moving parts, no workflow
ID needed. `inference-sdk` caps at Python 3.12 and this environment is 3.13, so
the REST contract is called with `requests`.

Built:

| File | Purpose |
|---|---|
| `pipeline/chit_detector.py` | HTTP client, crop geometry, upscaling, coordinate mapping back to frame space |
| `tools/attach_chit_detections.py` | Crops each person from a finished run, scores it, writes into `near_hand_objects` |

Design notes that matter:

- The class is mapped to **`paper_like_object`**, not `secondary_paper_chit`. The
  latter asserts the paper is illicit, which the model cannot establish — it
  fired on an answer sheet in frame 9.
- Crops are **upscaled** to a 320 px short side (capped 4×) because the model
  trains at 640×640 and our median person is 96 px wide.
- The API key is read from `ROBOFLOW_API_KEY`. A key in a tracked file is a key
  that has to be rotated.
- The tool **re-uses a finished run** rather than re-running detection and pose.

---

## 3. Defects found and fixed

### 3.1 The VFR frame-matching bug — the serious one

The first full pass reported success on **1,527 of 6,109 samples** and looked
complete.

Samples store the **planned** grid time (250, 500, 750 ms). Four of six corpus
files are VFR, and this one actually decodes at 280, 480, 760 ms. Exact matching
hit **88 of 353** timestamps.

This is the same class as the frame-dropping mishap already raised against this
pipeline, and it was invisible because **nothing counted**. Fixed by assigning
each planned stamp to its nearest decoded frame with a one-frame lookahead, and —
more importantly — by making the tool **hard-fail** rather than report a partial
pass as complete. Re-run: 353/353 stamps, 4,934 crops.

### 3.2 Legacy attachment format

Run 1501 predates the box-carrying change, so its D-FINE entries are bare
strings. Counting them crashed. Guarded, with the guard written to *separate*
what the chit model contributed from what was already there rather than silently
skipping.

### 3.3 Renderer

- Added `paper_like_object` to `TARGET_CLASSES`.
- Added `--samples` so an enriched copy can be drawn without overwriting the original.
- Made it **refuse to overwrite** an existing output. Run 1501 had silently
  written `annotated_video_1.mp4` beside an older file, and the stale one was the
  one people opened.

### 3.4 Roboflow rate limiting

On 04_talking, **2,937 of 3,439 calls failed (85%)** at 16 workers. The API was
healthy when tested immediately after — these were transient 429s, and the retry
policy (3 attempts, ≤1.5 s backoff) was too impatient. Re-run at 6 workers. The
`api_failures` counter existed because a pass that silently scores 85% of crops
as empty would otherwise report "few chits found" as a finding.

---

## 4. The precision collapse, and what actually fixes it

### 4.1 Raw result on 12_paper

**1,817 detections on 1,616 of 4,934 crops — 33% of every person-sighting** in a
video where one candidate takes one piece of paper.

Rendered and inspected, the detector fires on:

| What | Confidence |
|---|--:|
| A photograph displayed **on a monitor** | 0.65 |
| A light blue **shirt sleeve** | 0.42 |
| **White shirt sleeves** | 0.41, 0.38 |
| Frosted **desk partition panels** | ~0.44 |

**A confidence floor cannot separate these.** The monitor-photo false positive at
0.65 sits above four of the five verified true positives.

### 4.2 Duration gating — proposed, tested, rejected

The proposal: a chit is transient; a shirt or an answer sheet is permanent. It is
a good argument, and it is what `persistent_object_beside_seat` already does.

**On this video it removes the true positive.**

The subject — tracks **118 and 53**, one man split by the re-identification gap,
the striped-shirt candidate at the left desk beside placard 13 — carries a
detection in **95% of his sightings**, because he takes the paper and then keeps
it. Under a duty-cycle cap of 0.25 he is excluded entirely, and the five
survivors are tracks whose boxes land on background furniture (all rendered and
checked).

**For this class of event, persistence is the signal, not the artefact.**

### 4.3 What does separate them: geometry

The false positives are on torsos, panels and screens. The true ones are at the
wrists.

| Gate | Rejected on 12_paper |
|---|--:|
| Away from the hands (>0.35 person-widths from a wrist) | **729** |
| Larger than a chit (>12% of the person box) | **621** |
| No wrist resolved in that frame | 74 |
| Below 0.40 confidence | **0** |

**1,817 → 393. 236 tracks → 13 with surviving evidence.** And the ranking comes
out right:

| Rank | Track | Handling | Longest | Peak |
|--:|--:|--:|--:|--:|
| **1** | **118** (subject) | **56.5 s** | 56.5 s | 0.74 |
| 2 | 22 | 15.2 s | 5.0 s | 0.68 |
| **3** | **53** (same man) | **13.0 s** | 12.0 s | 0.77 |
| 4 | 100 | 10.5 s | 5.8 s | 0.61 |

Note the confidence gate rejected **zero**. Size and hand-proximity carry
everything.

Built: `pipeline/chit_evidence.py` and `tools/score_chit_evidence.py`. The
module docstring records that duration gating was tried and why it failed, so
nobody re-derives it.

---

## 5. Model composition — what is actually combined

The `code-review-graph` MCP server is configured in `.mcp.json` but was not
connected this session, so this was traced from source.

| Model | Role | Drawn in the video? |
|---|---|---|
| D-FINE | person detection → the box every sample is built on | **Yes** |
| D-FINE | COCO objects | **No** in run 1501 — bare strings, zero drawable |
| AlphaPose | skeleton, joints, gaze arrow, **wrists** | **Yes**, and the wrists are the chit gate |
| `chit-paper-new/2` | paper detection | **Yes** |

**The three are not independent.** The person box comes from D-FINE and the chit
gates are measured in person-box widths, so a bad person box moves both gates.

---

## 6. Runs executed

| Run | Video | Resolution | Duration | Sampled | Boxes | Tracks | Objects (boxed) |
|---|---|---|--:|--:|--:|--:|--:|
| 1501/12_paper | Seat No. 12 … paper | 1280×720 | 88.4 s | 353 (100%) | 6,109 | 236 | 0 (legacy) |
| 1503/newsclip_12 | news clip, Chhatrapati Sambhajinagar | 360×450 | 38.2 s | 153 (100%) | 3,326 | **515** | **1,887** |
| 1504/04_talking | 04.CCTV Candidate Talking | 640×480 | 143.2 s | 573 (100%) | 3,741 | 92 | **34,805 kept / 16,647 attached** |

A first attempt at 1504's chit pass **failed 2,937 of 3,439 calls (85%)** to
transient Roboflow rate limiting at 16 workers, and was discarded and re-run at
6 workers. The `api_failures` counter is why that was caught rather than
reported as "few chits found".

Run **1502** was started as a 12_paper box-carrying re-run, then killed at crop
3,000/5,111 — I had misread the instruction, which was to run the box-carrying
pipeline **on the new video**, not on 12_paper. The partial directory was removed.

### 6.1 The box-carrying re-run worked

On 1503 and 1504, **every D-FINE object now carries a box and a confidence**.
`book_or_notebook_like_object` and `phone` are spatially usable, which makes the
D-FINE-vs-chit cross-check possible for the first time. It is not yet built.

### 6.2 04_talking: the wrist gate fails when the false positive is *at the hands*

Run 1504 produced **2,689 raw chit detections -> 1,461 kept (54%)**, against 22%
on 12_paper, and ranked four people as `sustained_paper_handling` for **132 s,
108 s, 83 s and 64 s of a 143 s recording** -- everyone, nearly always.

Rendered and inspected at three timestamps, the surviving detections are:

| What fired | Confidence |
|---|--:|
| A **keyboard and mouse** under a resting hand | 0.74 |
| A hand on a **mouse** | 0.52 |
| A **water bottle** on the desk | 0.44 |

This is a different failure from 12_paper, and it matters more:

> On 12_paper the false positives were **shirts, partitions and monitors** --
> all *away from the wrists* -- so the hand-proximity gate removed 729 of them
> and moved the true subject to rank 1. On 04_talking the false positives are
> **keyboards, mice and desk objects, which sit exactly where a wrist is.**
> Geometry cannot separate them, because geometrically they are correct.

**The wrist gate is not a general solution.** It works where the confusers are
background furniture and fails where the confusers are hand-held equipment. In a
computer-based test centre the hands are on a keyboard for three hours, so this
is the common case, not the edge case.

`sustained_paper_handling: 132 s` on this video means *"a man used a computer"*.
None of run 1504's chit findings should be reported as evidence of anything.

What would actually separate them, none of it built:

- **Suppress classes D-FINE already identifies confidently at the same location.**
  D-FINE reports `keyboard` and `mouse` with boxes now; a chit detection that
  overlaps one should be dropped. This is exactly the spatial cross-check the
  box-carrying re-run unlocked.
- **Require motion.** A keyboard is static relative to the desk; a chit being
  read moves with the hand.
- **Require the object to be absent for part of the recording.** A keyboard is
  never absent.

### 6.3 A threshold that does not transfer

On 1503 the size cap rejected **237 of 326** detections. Median person width is
**52 px** there vs **96 px** on 12_paper, so detections are relatively larger:
0.12 of the person box is p63 on one video and p25 on the other. I raised it to
0.20 to be percentile-equivalent, and recorded this as a **defect** — the gate is
meant to be scale-free and is not. It should normalise against torso width.

---

## 6.4 SAM 3 as a referee — the strongest result in the project

The `general-segmentation-api-6` workflow was integrated
(`pipeline/roboflow_workflow.py`, `tools/adjudicate_with_sam3.py`). Its
definition was fetched from the Roboflow API rather than assumed: it is
**SAM 3** (`sam3/sam3_final`) with `classes` as a free-text prompt, outputs
`annotated_image` and `predictions`.

Being able to **name the confuser** is what every other detector here lacks. On
the exact 04_talking frame where the chit model returned `chit-paper 0.74` on a
keyboard, prompting `"mobile phone, paper chit, keyboard"` returns **six
keyboards at 0.52-0.89, zero phones, zero chits**.

### On the nine verification frames

| # | Truth | chit-paper-new/2 | SAM 3 |
|--:|---|---|---|
| 1 | paper in hands | chit 0.74 | paper chit 0.55 |
| 2 | card raised | chit 0.56 | **nothing — miss** |
| 3 | **a phone** | chit 0.52 (**wrong class**) | **mobile phone 0.91** |
| 4 | placard wall | silent | silent |
| 5 | paper in hands | chit 0.72 | **nothing — miss** |
| 6 | sheets on desk | silent | silent |
| 7 | dark object | 0.52 ambiguous | nothing |
| 8 | object in hand | 3 weak boxes 0.22-0.28 | mobile phone 0.64 |
| 9 | held sheet | chit 0.68, one box | **two chits 0.65 / 0.73** |

SAM 3 is the **more precise and less sensitive** instrument: it fixes the class
confusion on frame 3, finds a second chit on frame 9 that the chit model missed,
and misses two true positives the chit model found. That is why it is wired in
as a *referee over existing detections*, not as a replacement detector.

### Adjudicating 12_paper — the control

392 gated detections adjudicated. Verdicts concentrate almost perfectly:

| Track | Corroborated | Suppressed | Unsupported |
|---|--:|--:|--:|
| **118** (the subject) | **110** | 9 | 99 |
| **53** (the same man, re-ID split) | **23** | 4 | 19 |
| 114 | 3 | 0 | 4 |
| **22** (previously ranked #2) | **0** | 0 | 46 |
| every other track | **0** | — | — |

**133 of 136 corroborations land on the one man who actually took the paper.**
Track 22, which the chit pipeline ranked second with 15.2 s of handling, is
independently rejected. 48 detections were suppressed as keyboards, monitors or
a mouse.

This is the first time two independent models have agreed on a specific person
in this project, and it is exactly the "independent indicator" the
classification ladder requires -- a second chit detection 250 ms later is not
independent evidence; a different model with a different vocabulary is.

### Adjudicating 04_talking

**Zero corroborated.** Every gated detection came back `unsupported` or
`suppressed_keyboard` / `suppressed_computer mouse`. SAM 3 independently
confirms the conclusion reached by rendering: run 1504's chit findings are
keyboards and mice, and none of them is evidence of anything.

### One defect in the adjudicator, found and fixed

The first version listed `hand` as a confuser. A chit held in a hand always
overlaps a hand, so **56 of 60 adjudications came back `suppressed_hand`,
including the true positives**. `hand` is context, not a confuser: it is still
in the prompt, because naming it helps SAM 3 locate the region, but it can no
longer suppress. Confuser matching also now takes the *strongest* overlap rather
than the first in segment order.

---

## 7. Metric tables

### 7.1 Object detections (D-FINE, stock COCO)

Raw detections, **not objects**. One phone visible for 60 s at 4 Hz produces 240.

| Class | 12_paper | 01_phone | newsclip_12 |
|---|--:|--:|--:|
| monitor_or_bezel | 18,320 | 2,321 | 766 |
| keyboard | 8,362 | 200 | 31 |
| mouse | 4,171 | 943 | 10 |
| bag | 2,383 | 839 | 110 |
| **book_or_notebook_like_object** | **1,680** | **181** | **483** |
| **phone** | **1,032** | **332** | **415** |
| stationery | 105 | 178 | 72 |

### 7.2 Chit model, gated

| | 12_paper | newsclip_12 |
|---|--:|--:|
| Crops scored | 4,934 | 2,443 |
| Raw detections | 1,817 | 326 |
| Kept after gating | **393** | **51** |
| Tracks with surviving evidence | 13 of 236 | 17 of 515 |
| sustained_paper_handling | **4** | 0 |
| repeated_brief_paper_handling | 0 | 2 |
| brief_paper_handling | 9 | 12 |

### 7.3 Confirmed cheating incidents

**Zero.** Not because nothing happened — because the pipeline does not confirm
cheating and **there is no ground-truth manifest anywhere in this project**.
Every figure above is what the system reported, not what was true. No precision,
no recall, no false-alarm rate.

---

## 8. Mistakes made this session

Recorded because the pattern matters more than the individual errors.

1. **Misread the instruction** and started a 12_paper re-run when the request was
   to run the box-carrying pipeline on the new video. Cost ~8 minutes of GPU.
2. **Miscalculated elapsed time** and declared a run pathologically slow when it
   was 7 minutes in and healthy. Corrected after checking the clock.
3. **Polled a background job repeatedly**, generating a stream of task
   notifications the user had to read through.
4. **Estimated the model's accuracy before testing it** — the 0.15–0.35 mAP
   guess. The user's own test disproved it.

The recurring discipline that caught every real bug: **render it and look at the
pixels before believing a count.**

---

## 9. Where this leaves the project

Next steps, in the order I would take them, are specified in
[`CLASSIFICATION_AND_OPEN_QUESTIONS.md`](CLASSIFICATION_AND_OPEN_QUESTIONS.md).
The two highest-value:

1. **Hand-label the known incidents** (Q13). A day, no GPU, and it converts every
   table above from an assertion into a measurement.
2. **Seat occupancy** (Q1/Q2/Q6). Low difficulty, uses calibration machinery that
   already exists, and unlocks get-up/sit-down, seat swaps, impersonation, and
   not flagging invigilators.
