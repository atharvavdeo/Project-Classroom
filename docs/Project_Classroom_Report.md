# Project Classroom — Build Report and Readiness Assessment

**Problem Statement:** Drishti AI Hackathon 2026 — **PS #2: Offline Video Segmentation and ROI Detection Using Motion Estimation**
**Team:** DataDynamo
**Repository:** `Project-Classroom`
**Report date:** 18 August 2026
**Covers:** the complete build session, every defect found and fixed, the final measured state, and an honest comparison against PS #2 as written.

---

## Table of contents

1. [What PS #2 actually asks for](#1-what-ps-2-actually-asks-for)
2. [What was built, in order](#2-what-was-built-in-order)
3. [Every defect found, and its fix](#3-every-defect-found-and-its-fix)
4. [Glossary: every term and metric used](#4-glossary-every-term-and-metric-used)
5. [Final evaluation — the measured numbers](#5-final-evaluation--the-measured-numbers)
6. [Readiness against PS #2, requirement by requirement](#6-readiness-against-ps-2-requirement-by-requirement)
7. [What we are, and what we are not](#7-what-we-are-and-what-we-are-not)
8. [What it would take to close the gap](#8-what-it-would-take-to-close-the-gap)

---

## 1. What PS #2 actually asks for

The problem statement is worth reading closely, because its framing decides what "done" means. Three sentences carry most of the weight:

> "Manual review of hours of video footage is time-consuming, labor-intensive, and prone to human error, making it difficult to efficiently identify relevant events."

> "…automatically identify motion-based regions, filter out static scenes, highlight periods of unusual activity, and generate summarized outputs for investigators."

> **Mitigation Strategy:** "The system should function as an **offline investigation support tool** that automatically prioritizes and summarizes recorded footage for human reviewers… allowing investigators to quickly focus on potentially relevant portions of examination recordings while **maintaining human oversight in the final review process**."

This is not a cheating detector. It is a **triage and prioritisation** system whose success metric is *reviewer effort saved without losing real events*. Every architectural decision in this build follows from that reading.

### 1.1 The eight stated objectives

| # | Objective (verbatim) |
|---|---|
| O1 | Process recorded examination hall videos offline. |
| O2 | Detect motion regions using frame-to-frame motion estimation techniques. |
| O3 | Automatically identify Regions of Interest (ROI) containing significant activity. |
| O4 | Identify external objects like Mobile Phone, Paper/Chits or any other prohibited object **as may be feasible**. |
| O5 | Segment long video recordings into activity-based clips. |
| O6 | Generate motion heatmaps and activity timelines. |
| O7 | Reduce manual video review effort by highlighting relevant events. |
| O8 | Provide event summaries and visual analytics for investigators. |

Note the qualifier on O4 — **"as may be feasible"**. It is the only objective hedged this way in the entire document, and it turned out to be the single most important word in the problem statement for this corpus. Section 5.4 explains why.

### 1.2 The twelve stated deployment challenges

Crowded halls · subtle motion detection · camera noise · lighting variations · camera vibration · long video duration · ROI boundary accuracy · background motion · event segmentation quality · multi-camera synchronisation · scalability · false event generation.

### 1.3 The twelve stated risk factors

False motion detection · missed ROI detection · over-segmentation · under-segmentation · camera motion effects · storage constraints · processing delays · ROI localisation errors · environmental variability · data security · hardware failures · scalability.

### 1.4 Stated hardware for final deployment

| Component | Specification |
|---|---|
| GPU | RTX 4070 / 4080 / **4090** |
| RAM | 64–128 GB |
| Storage | 4–20 TB SSD/HDD |
| Monitoring workstation | 16–32 GB RAM, dual monitors |
| Network | Gigabit Ethernet |

The agreed deployment target for this build is a **32 GB RTX 4090**, the top of that range.

---

## 2. What was built, in order

### 2.1 Phase 0 — Analysis before code

The starting point was a PRD from a public repository containing **no code**. Before writing any, the entire proposed architecture was evaluated against published benchmarks and the actual source footage. Four conclusions changed the design:

**The corpus is not what the PRD claimed.** Measuring all six source files established: maximum resolution **720p, not 1080p**; compression at **~0.08 bits/pixel**; **variable frame rate on three of six files**; codec motion vectors present in **all six**, MPEG-4 included; and a phone footprint of roughly **30 × 20 pixels** at a *near* seat.

**Published accuracy on this exact problem is low.** YOLOv5/v6/v7 on classroom-cheating CCTV report **43% / 37% / 51%** accuracy. That is the realistic ceiling for appearance-first detection, and it makes "detect phones" an unsupportable product claim on 30 × 20 px objects.

**DeepStream was evaluated and rejected.** It is Linux/Jetson/WSL2-only, NVDEC does not support MPEG-4 Part 2 (two corpus files), and the whole-video motion pass was measured at **66× realtime on CPU** — 4.6 minutes for 5 hours. The dependency buys nothing this pipeline needs.

**The product was reframed** from object detection toward **behaviour ranking**: sustained motion deviation, downward head pitch, below-desk hand dwell — with object evidence as a bonus that abstains when it cannot support a claim.

### 2.2 Phases 1–8 — the pipeline

| Phase | Deliverable | Key modules |
|---|---|---|
| **1 · Ingest** | PyAV ingest, codec/PTS/VFR validation, checksums, camera registration | `ingest.py`, `db.py`, `schema.sql` |
| **2 · Calibration** | Seat polygons with three zones, overlay/reflection/environment masks | `calibration.py`, `tools/calibrate.py` |
| **3 · Motion scan** | Codec motion-vector whole-video scan, global-motion and exposure compensation, per-seat robust statistics | `motion.py` |
| **4 · Segmentation** | Per-seat hysteresis state machine, PTS-accurate boundaries | `segment.py` |
| **5 · Console** | Heatmap, timeline, event log, FastAPI, review UI — **PS2-complete with no neural network at all** | `scoring.py`, `clips.py`, `api.py`, `static/console.html` |
| **6 · Detection + pose** | D-FINE on native-resolution seat crops, top-down pose, ByteTrack with seat reconciliation | `detect.py`, `pose.py`, `track.py`, `crops.py` |
| **7 · Scoring + verifier** | Hand-set priority score, Gemma 4 E4B verifier with GBNF-constrained JSON | `verify.py`, `evidence.py` |
| **8 · Ground truth + measurement** | Annotation store and tooling, evaluation metrics, COCO export, fine-tune driver | `annotate.py`, `evaluate.py`, `tools/annotate_*.py`, `tools/export_dataset.py` |

Phases 1–5 require **no GPU**. This is deliberate: the PS2 deliverable is complete and demonstrable before any model is loaded.

### 2.3 Supporting instruments built

| Tool | Purpose |
|---|---|
| `tools/run_video.py` | One video end-to-end into a self-contained run folder with overlays |
| `tools/prd_check.py` | Reads the PRD as data and checks it against the code and the footage |
| `tools/validate_corpus.py` | Re-measures every source file against the recorded inventory |
| `tools/annotate_timeline.py` | PTS-accurate scrubber for temporal ground truth |
| `tools/annotate_boxes.py` | Box annotation with a pixel loupe |
| `tools/export_dataset.py` | COCO export, split by camera |
| `tools/finetune_dfine.py` | Preflight and config generation for the detector fine-tune |
| `tools/pose_ab.py` | Measured pose-model comparison |
| `tools/detector_sweep.py`, `render_phones.py` | Detector behaviour audit on real frames |
| `classroom/harness.py` | Long-form test recording synthesised from real frames |

### 2.4 Final repository state

```
16 commits
21 modules in classroom/
13 tools/
15 validation gates in tests/
~11,600 lines of code
PRD at v3.2, 1,424 lines
```

---

## 3. Every defect found, and its fix

This is the substance of the work. **33 defects** were found; every one was found by *running* something, not by reading it. They are grouped by how they would have failed in production.

Legend: 🔴 would have produced silently wrong output · 🟠 would have crashed or blocked · 🟡 documentation or process · ⚪ my own error in this session

### 3.1 Silent-wrong-output defects — the dangerous class

#### 🔴 D1 — Baseline learning absorbed the events it was meant to find

**Symptom:** Event recall of 1 in 3 on planted events.
**Cause:** `learn_baselines` treated any elevated-motion population as a "typing regime". Where a seat was genuinely active, the events themselves became the baseline, so the system trained itself to ignore exactly what it exists to find.
**Fix:** `min_regime_fraction = 0.25` — a regime is only established when the active windows make up at least a quarter of the learning span. Typing in a CBT hall is near-continuous; genuine incidents are not.

#### 🔴 D2 — The MAD floor was absolute where it needed to be relative

**Symptom:** Real footage produced **zero events**, peak z of 1.5 where the true value was ~47. All fixtures passed.
**Cause:** Residual magnitudes are scene-dependent. Real 720p CCTV gives a seat median residual around 0.01 with MAD near 0.0016; a high-contrast synthetic fixture gives values two orders of magnitude larger. An absolute floor of 0.05 against a true MAD of 0.0016 divides every z-score by 30 and the gate goes blind.
**Fix:** `min_mad_fraction = 0.05` — the spread floor is *relative to the seat's own median*.

#### 🔴 D3 — …and the relative floor collapsed on static seats

**Symptom:** A completely static seat produced **z = 16.6 with zero moving blocks**.
**Cause:** On a static seat both median and MAD approach zero, so a purely relative floor also collapses — a trivial fluctuation divided by a near-zero denominator reports an enormous deviation from nothing at all.
**Fix:** `min_mad_absolute = 0.001` plus a `min_moving_blocks` guard requiring that something physically moved.

#### 🔴 D4 — Luma was sampled globally, not per window

**Symptom:** A fabricated exposure step at the end of every recording.
**Cause:** The final partial window had zero luma samples, so its mean read as a step change against the previous window.
**Fix:** A per-window frame counter seeded with the window's first frame.

#### 🔴 D5 — Backdating re-absorbed max-duration splits

**Symptom:** A long noisy stretch that had been correctly split into separate events re-merged into one unusable event.
**Cause:** Event backdating walked backwards past the split point.
**Fix:** `floor_index` set at both event-close sites, bounding how far backdating may reach.

#### 🔴 D6 — Backdated events stuck in CANDIDATE and never fired

**Symptom:** One planted event in four never reached ACTIVE.
**Cause:** A backdated event was policed by the short bridge window rather than the full cooling window.
**Fix:** Promote to ACTIVE on entry when the backdated span already exceeds the minimum duration.

#### 🔴 D7 — Rolling-max smoothing destroyed impulse rejection *(a fix that was tried and rejected)*

**Symptom:** A dropped pen became a 3-second incident.
**Cause:** I first tried smoothing the deviation signal with a rolling maximum to bridge gaps in intermittent hand motion. A rolling max **manufactures duration from a single spike**, which is precisely what the minimum-duration rule exists to prevent.
**Fix:** Rejected the smoothing entirely. Replaced with `candidate_bridge_s = 1.5` — time-since-last-active bridging, which spans gaps *between* activity without inventing activity.

#### 🔴 D8 — Seat-overlap validation produced false positives

**Symptom:** Flush-adjacent seats reported as overlapping, blocking calibration.
**Cause:** OpenCV `fillPoly` includes the polygon boundary, so two seats sharing an edge intersect on that edge.
**Fix:** Erode each mask by 1 px before intersecting.

#### 🔴 D9 — `evidence.persist` erased the other model's evidence

**Symptom:** Under sequential model residency, detections vanished. The run reported success.
**Cause:** `persist` cleared **both** the `detection` and `pose_observation` tables on every call, so a pose-only second pass deleted the detections the detector pass had just written.
**Fix:** Each stream clears only itself, conditioned on whether that stream was actually gathered. Gate added covering the two-pass sequence.

#### 🔴 D10 — The annotation scrubber silently seeked to the wrong frame

**Symptom:** None. Nothing raised.
**Cause:** A backward seek to a timestamp at or before the first keyframe does not land on the first frame — the demuxer skips forward to the *next* keyframe. Measured: `seek(0)` lands at **0.483 s** on `03.CCTV Mobile Usage.mkv` and at **1.040 s** on the Seat 12 file, skipping seven and eighteen frames.
**Why it matters:** Every mark made near the head of a file would be permanently wrong *in the ground truth everything else is measured against*.
**Fix:** The unreachable head is measured once per file at open; any seek below it decodes from the start. Verified exact against true frame times on five files including all three VFR ones.

#### 🔴 D11 — Arrow keys collapsed to the same value on Windows

**Symptom:** Step-backward and step-forward were the same key in the annotator.
**Cause:** Masking the Win32 key codes to a byte: `2424832 & 0xFFFF == 2555904 & 0xFFFF == 0`.
**Fix:** `waitKeyEx` with the full code sets for every HighGUI backend, plus letter aliases.

#### 🔴 D12 — **An empty seat outscored the occupied one** *(the largest defect found)*

**Symptom:** On a full run of the longest corpus file, an **unoccupied desk row produced the twelve highest-scoring events in the entire recording**, peaking at **29.8 σ**, against 9.5 σ for the one seat that actually contained a person. The region is visually static across every sampled frame — monitors off, nobody present.

| Seat | Median residual | MAD | Mean area ratio | Max z |
|---|---:|---:|---:|---:|
| R-desk (empty) | 0.0130 | 0.0081 | **0.0011** | **29.8** |
| S-61 (occupied) | 0.1835 | 0.1616 | 0.0465 | 9.5 |

Forty times less motion, three times the score.

**My first hypothesis was wrong, and is recorded as wrong.** The seat polygon overlaps an interior serving window into an adjacent room, so environmental leakage was the obvious explanation. An environment mask was added and the run repeated: **21 events became 20.** The window is a real motion source and the mask is worth keeping — but it was not the cause.

**Actual cause:** `min_moving_blocks` is an **absolute count** where the quantity that matters is a **fraction**. A large zone accumulates stray blocks in proportion to its area, so one moving block means something entirely different in a 660-block region than in a 520-block one — and an absolute count passes both. A near-zero MAD in the denominator then turns that single block of compression noise into an enormous standardised value.

Across every window above the start threshold the two populations separate completely:

| Seat | Windows above start_z | Area ratio median | Range |
|---|---:|---:|---|
| R-desk | 142 | 0.0033 | max **0.0333** |
| S-61 | 39 | 0.2625 | min **0.1458** |

A **4.4× gap with nothing in between**.

**Fix:** `min_area_ratio = 0.05`, placed in that gap with wide margin on both sides.

| Configuration | Events | Empty row | Occupied seat |
|---|---:|---:|---:|
| As built | 21 | 13 | 8 |
| + environment mask | 20 | 12 | 8 |
| **+ area floor** | **8** | **0** | **8** |

**Review burden fell from 46.3% of the recording to 16.3%.** Pose observations rose from 5 to 31 over the same event count, because the events now land on a seat containing a person.

#### 🔴 D13 — **The verifier described the wrong person, at confidence 1.0**

**Symptom:** Gemma returned fluent, detailed, internally coherent descriptions of *"a candidate seated at a desk with a monitor, keyboard, and mouse"* — at `evidence_quality: "sufficient"`, `confidence: 1.0`. The events were on the **empty** desk row. It was describing a man seated elsewhere in the frame.

**Cause:** The contact sheet was the **whole frame, six times over**. The seat under review was identified only as the string `"seat": "R-desk"` in the metadata JSON. **A vision model cannot resolve a text label to a region of an image**, so the attribution was never conveyed at all.

**Why nothing caught it:** the output was schema-valid, the grammar was satisfied, the sanitizer found no verdict language, and the observations were *true of somebody*.

**Fix:** The seat polygon is drawn on every tile and a magnified crop of that seat is appended as a final tile — the tile the original prompt already described and the implementation had never produced. The prompt now states that the red box marks the subject, that other people are not the subject, and that an empty box means `insufficient`.

**Confirmed live after the fix:** *"The description is based on the magnified crop of seat 61."*

#### 🔴 D14 — **The long-form harness had never emitted a motion vector**

**Symptom:** Adding the area floor (D12) took harness recall from **5/5 to 0/5**.

**Investigation:** the harness was the unrealistic side, not the threshold. It synthesised events by **cross-fading** between two real frames — which changes pixel *values* without displacing anything, so the encoder codes it as residual and emits almost no motion vectors.

| | Mean area ratio during events | Residual change |
|---|---|---|
| Harness (before) | **0.0000 – 0.0008** | rose 1.8× to 4.7× |
| Real footage | **0.15 – 0.69** | — |

**Consequence:** the harness had been validating the **statistical** path (residual → robust z) and never the **kinematic** path (motion vectors → moving blocks). Half the pipeline was not under test and nothing said so. Its previously reported recall figure meant considerably less than it appeared to.

**Fix, and a failed first attempt:** the composite now translates as well as fades. A **sinusoidal** displacement was tried first and failed — a sine has zero velocity at each turning point and peaks a quarter-cycle away from the alpha peak, so the frames where the composite is most visible are the frames where nothing is moving. That produced two to four qualifying windows per event and still no sustained detection. A **triangle wave** holds |velocity| constant at ~2.9 px/frame across the whole swing.

**Re-measured:** 5/5 recall, precision 1.00, Precision@5 1.00, **zero** false events, 76 pose observations — and now exercising the whole path.

#### 🔴 D15 — Four GBNF grammar defects, all latent runtime failures

The grammar was rejected outright by llama.cpp, or truncated output mid-token:

1. **Multi-line rules do not parse.** The pretty-printed form of the same grammar — `root` wrapped across eight indented lines — is rejected with "failed to parse grammar". One rule per line, however long.
2. **A backslash inside a character class does not parse.** The conventional JSON string rule `[^"\\] | "\\" ["\\/bfnrt]` is refused.
3. **Unbounded repetition truncates.** `obslist`, `string`, and especially `ws ::= [ \t\n]*` — the last is a pure token sink, letting the model emit unlimited blank lines and exhaust its budget without producing content.
4. **The review note was cut mid-sentence at 200 characters** — worse than useless, because a truncated explanation reads as a complete one.

**Fixes:** one rule per line; character classes that exclude rather than escape; bounded repetition everywhere including `ws ::= [ \t\n]{0,4}`; a separate 400-character bound for the note.

#### 🔴 D16 — Observation strings truncated mid-word, and filler padding

**Symptom (live):** `"...Both hands are visible on the desk surface, near the keyboard. The 7"` — cut mid-word at the 200-character bound. And one response padded its observation list with four entries of the literal string `" ,"`.
**Cause:** the model wrote multi-sentence paragraphs into a field meant for one short statement; and the grammar accepted any string length including near-zero.
**Fix:** the prompt asks for one short statement per item and forbids repeats; `string` gained a **minimum** length of 12 characters so degenerate padding is unrepresentable rather than discouraged; the upper bound moved to 240.

### 3.2 Crash / blocking defects

#### 🟠 D17 — PyAV 18 API changes

`frame.pict_type.name` fails (it is an int enum in PyAV 18) → `getattr(pt, "name", None) or str(pt).split(".")[-1]`.
`codec_context.export_mvs` does not exist → `stream.codec_context.options = {"flags2": "+export_mvs"}`.

#### 🟠 D18 — `Path().glob()` rejects absolute patterns

Ingest could not accept absolute paths on the command line → wrote `_expand()` using `glob.glob`.

#### 🟠 D19 — `llama-server.exe` exists but cannot start

**Symptom:** `FileNotFoundError: [WinError 2] The system cannot find the file specified` — naming a file that demonstrably exists. The first full run died at the **last stage**, after five minutes of CV work.
**Cause:** the shipped `llama-server.exe` is a 9 KB stub that loads `llama-server-impl.dll` and the ggml backend DLLs **from its own directory**. Launched from anywhere else, `CreateProcess` cannot resolve them and reports the *executable* as missing.
**Lesson:** `Path(binary).exists()` proves nothing about whether a binary can launch.
**Fix:** run it with its own directory as the working directory, use absolute paths for the model, and **prove it launches with `--version`** before committing to the slow startup path.

#### 🟠 D20 — OpenMP double-load aborts `import torch`

**Symptom:** `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized` — process aborts before any of our code runs.
**Cause:** in an Anaconda environment two copies of `libiomp5md.dll` are reachable, one from MKL and one inside the torch wheel.
**Measured:** `import torch` alone aborts; `import numpy, torch` succeeds — loading numpy first pins the MKL runtime.
**Fix:** documented import order in the fine-tune preflight.

#### 🟠 D21 — `score_recall` raised `KeyError` on a zero-detection run

**Symptom:** a run that detected nothing crashed in the *reporting* code, so a total-recall failure surfaced as a stack trace rather than as the failing check it is.
**Fix:** `precision_at_k` is always reported.

### 3.3 Documentation and process defects

#### 🟡 D22 — The PRD's frame-rate table was wrong on two of six files

`validate_corpus.py` reported DRIFT. **The code was right and the document was wrong.** File 03 recorded 12.1 fps against a true **15.0**; Seat 12 recorded 25.0 against a true **21.9**. Both are the rates the *container declared* — and both are VFR files, precisely where declared metadata is unreliable. Rebuilt from measurement.

#### 🟡 D23 — The §5.1 table had nine headers and ten cells per row

The frame-count column was never named. Found by `prd_check.py` on its first pass.

#### 🟡 D24 — §9.2.2 sat at the wrong heading depth

`###` while its sibling §9.2.1 sat at `####`, so it read as a child of §9 rather than of §9.2.

#### 🟡 D25 — §9.3 never stated the verifier context size

4096 tokens is a governing runtime value and was absent from the document entirely.

#### 🟡 D26 — §8.2 still declared a model choice that §8.2.1 had reversed

The pose section header table listed **RTMO-l** as the decision while the subsection immediately below it presented the measurement that reversed that decision in favour of top-down. An internal contradiction that survived several revisions. Resolved throughout the document.

#### 🟡 D27 — The hard-negative list was written for the wrong room

The detector's hard-negative list inherited **erasers, pens and rulers** from a paper-examination model. The measured confusers on this CBT footage are **computer mice above all**, then keyboards and monitor bezels. A model trained against rulers would not touch the errors this footage actually produces. Rebuilt by observed confusion rate.

#### 🟡 D28 — Stock detector precision quantified at ~40%

Not a defect but a finding that reshaped the product. All 23 `cell phone` detections across 48 real frames were cropped and adjudicated by eye: **~9 genuine, 5 keyboard/monitor edges, 3 computer mice, ~3 ambiguous**. Median detection footprint was 768 px² — *larger* than a real phone's 600 px², because the false positives are bigger objects. **No genuine phone was detected on any seated-candidate camera**; all nine came from a reception view where people stand close to the lens.

### 3.4 My own errors in this session

#### ⚪ D29 — I reported the corpus as unreachable when it was not

D: went offline mid-session and I only ever checked D:. The videos were on C: the entire time. **This deferred real work for several rounds** until the user corrected me. Everything that had been blocked was then completed: full-corpus validation, frame extraction, the detector sweep and visual adjudication.

#### ⚪ D30 — Two of five initial `prd_check` failures were my own bad regexes

The abstention-floor and block-threshold claims *were* in the PRD; my patterns were too strict. Corrected the checker, not the document.

#### ⚪ D31 — An off-by-one in my own gate

The contact-sheet highlight check used exclusive slice bounds and reported 1044 of 1256 red pixels "outside" the expected region. The drawing code was correct; the assertion was wrong. Replaced a fragile pixel-count equality with a bounding-box comparison against the mapped rectangle ±2 px.

#### ⚪ D32 — I wrote a wrong hypothesis into code before testing it

The environment mask for D12 was added on the strength of a plausible visual explanation before measuring whether it was the cause. It was not. The mask is correct and retained, but it was added for the wrong reason, and the sequence is recorded in the PRD rather than tidied away.

#### ⚪ D33 — The first displacement fix for D14 was phase-wrong

I described the sinusoidal displacement as "in step with the fade" and implemented it 90° out of phase, so the most visible frames were the still ones. Corrected to a triangle wave after measuring that it produced only two to four qualifying windows per event.

---

## 4. Glossary: every term and metric used

### 4.1 Video and codec terms

| Term | Meaning | Why it matters here |
|---|---|---|
| **Codec motion vector (MV)** | A displacement vector the video encoder already computed for each 16×16 block when compressing the file. Extracted at decode time with `flags2=+export_mvs`. | **Free motion estimation.** No optical flow needed for the whole-video pass. Present in all six corpus files including MPEG-4. This is what makes 66× realtime possible on CPU. |
| **PTS (Presentation Timestamp)** | The time at which a frame is meant to be displayed, stored per frame in the container. | **The only trustworthy clock.** Three of six files are VFR with *declared frame rates that are wrong*, so `frame_index ÷ fps` drifts against reality. Every timestamp in this system comes from PTS. |
| **VFR (Variable Frame Rate)** | Frames are not evenly spaced in time. | Confirmed on three of six files. Makes frame-index arithmetic prohibited. |
| **Keyframe / I-frame** | A frame encoded without reference to any other. Seeking can only land on these. | Cause of defect D10: seeking to the head of a file skips to the *second* keyframe. |
| **bits per pixel (bpp)** | Compressed data volume per pixel. This corpus: **~0.08 bpp**. | Extremely low. Block artefacts are guaranteed and *will* register as motion — which is the whole reason for the noise-floor thresholds. |
| **Block** | A 16×16 pixel unit. A 1280×720 frame is a 45 × 80 grid of 3,600 blocks. | The native granularity of the motion layer. |

### 4.2 Scene model terms

| Term | Meaning |
|---|---|
| **Seat** | A calibrated polygon in frame coordinates representing one candidate's station, identified by the **physical numbered placard** visible in frame — not by any biometric. |
| **Zone** | A subdivision of a seat. Three exist: **`screen`** (excluded from scoring entirely — a live monitor is an aperiodic motion source unrelated to the candidate), **`desk`** (where the typing baseline lives), **`torso`** (head and shoulder evidence). |
| **Desk line** | The frame `y` coordinate of the desk edge. Below it is lap/bag territory. |
| **Mask** | A frame-global exclusion. Four kinds: `overlay` (the burned-in date/time, which changes every second), `reflection` (glass, mirrors), `environment` (fans, curtains, motion from outside the room), `staff` (invigilator paths). |
| **Seat anchoring** | Identity is the *seat*, not the person. A tracker ID that cannot be attributed to a seat is recorded as **ambiguous** rather than guessed at. This is how the privacy requirement is met structurally. |

### 4.3 Motion and statistics terms

| Term | Meaning | Value used |
|---|---|---|
| **Window** | The analysis unit: 0.5 s of video. Long enough for a robust median, short enough to localise a 2 s gesture. | `window_s = 0.5` |
| **Moving block** | A 16×16 block whose motion vector magnitude exceeds the threshold. | `block_mag_threshold = 2.0` px/frame |
| **Area ratio** | **Fraction of a zone's blocks that are moving.** | Floor: `min_area_ratio = 0.05` |
| **Residual** | Per-zone motion magnitude after global-motion compensation. | — |
| **Median** | The middle value of a distribution. Used instead of the mean because it is unmoved by a few extreme values. | — |
| **MAD (Median Absolute Deviation)** | The median of the absolute distances from the median. A robust measure of spread. | Floors: 5% of median, absolute 0.001 |
| **Robust z-score** | `(value − median) ÷ MAD`. How unusual a window is **for that seat**, in units of that seat's own normal variation. | — |
| **Regime** | The mode a seat is in — `idle` or `typing`. Baselines are learned per regime. | `min_regime_fraction = 0.25` |
| **Global motion** | Coherent movement across most of the frame, meaning the *camera* moved, not the scene. | `global_motion_ratio = 0.55` |
| **Exposure step** | A frame-wide luminance jump from auto-exposure. Suppresses events rather than escalating them. | `exposure_step_delta = 6.0` |

**Why a robust z-score, and not a fixed threshold:** every seat has different lighting, different distance from the camera, and a different amount of normal activity. A fixed pixel threshold that works for the front row is meaningless for the back. The z-score asks *"is this unusual for this seat?"* — which is the only question that generalises.

**The failure mode of a z-score, which defect D12 is:** if a seat's normal variation is near zero, then *any* excursion divided by that near-zero spread is enormous. This is why a z-score always needs an **absolute** significance floor alongside it. Ours is now the area ratio.

### 4.4 Event segmentation terms

| Term | Meaning | Value |
|---|---|---|
| **Hysteresis / Schmitt trigger** | Using a **higher threshold to start** an event than to end it. | start 4.0 σ, stop 2.0 σ |
| **State machine** | QUIET → CANDIDATE → ACTIVE → COOLING → CLOSED, one per seat, independent. | — |
| **Minimum duration** | How long a candidate must persist to become a real event. Rejects impulses. | `min_duration_s = 1.2` |
| **Candidate bridge** | How long a forming candidate may dip below the stop threshold before being discarded. Bridges *gaps between activity*; does **not** smooth the signal. | `candidate_bridge_s = 1.5` |
| **Merge gap** | Activity resuming inside this gap re-enters the same event rather than starting a new one. | `merge_gap_s = 2.5` |
| **Maximum duration** | Forces a split so a long noisy stretch cannot become one unusable event. | `max_duration_s = 90.0` |

**Why hysteresis:** without it, a single action whose signal oscillates around one threshold fragments into dozens of clips — the **over-segmentation** risk PS2 names explicitly. The gap between 4.0 and 2.0 is what prevents that.

**Why the bridge is not smoothing:** smoothing (e.g. a rolling maximum) manufactures duration from a single spike, defeating the minimum-duration rule and turning a dropped pen into an incident. This was tried, measured, and rejected — see defect D7.

### 4.5 Detection and pose terms

| Term | Meaning |
|---|---|
| **D-FINE-L** | The object detector. 54.0 AP on COCO, 57.1 with Objects365 pretraining. |
| **Seat-crop supersampling** | Cropping each seat at **native resolution** and upscaling it into the detector's 640 px input, instead of shrinking the whole frame. This is where small-object recall comes from. |
| **Abstention** | Emitting `insufficient` instead of a class label when the source pixels cannot support one. **Structural, not a tunable threshold.** |
| **Abstention floor** | The minimum source-frame footprint for any class label, `min_object_px_area = 300 px²`. Measured *after* mapping back through the crop transform, so magnifying a crop can never make an object look big enough to qualify. |
| **Temporal confirmation** | An object must persist across 3 consecutive sampled frames. A single-frame detection at this resolution is far more likely to be a compression artefact. |
| **Top-down pose** | Detect people first (YOLOX), then estimate keypoints per person (RTMPose). |
| **One-stage pose (RTMO)** | Estimate all keypoints in one pass. Cost does not scale with crowd size. |
| **COCO-17 keypoints** | The 17-point skeleton: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. |
| **Head pitch** | Nose-to-shoulder vertical offset, normalised by shoulder width. **Camera-specific** — every upright candidate on this footage scores between −0.35 and −2.3 because the camera looks down. Calibrated per camera, never a constant. |
| **Wrist masking** | When a hand drops below the desk line, the wrist keypoint is **masked and its absence recorded as evidence** rather than imputed. The absence is the interesting signal. |
| **ByteTrack** | Multi-object tracker. Its IDs are session-local and non-biometric. |
| **Seat reconciliation** | Mapping a track to a seat by majority of frames. Below confidence, the track is `ambiguous`. |

### 4.6 Verifier terms

| Term | Meaning |
|---|---|
| **VLM (Vision-Language Model)** | Here: Gemma 4 E4B instruction-tuned, quantised to q4_0, run via `llama-server`. |
| **GBNF grammar** | A formal grammar constraining llama.cpp's decoding. Every field is enumerated, so **a malformed response is not representable** rather than merely discouraged. This is what makes "100% schema-valid" a meaningful claim. |
| **Contact sheet** | A tiled image of frames from one event: before, onset, peak, during, end — **plus a magnified crop of the seat under review**, with that seat outlined in red on every tile. |
| **Sanitizer** | A post-hoc pass that strips verdict language ("cheating", "misconduct") from model output. Defence in depth behind the prompt. |
| **Out-of-process verifier** | The VLM runs as a separate HTTP server, so a crash in it cannot take the pipeline down. A dead endpoint is an error response, not a segfault. |

### 4.7 Evaluation metrics — the important section

| Metric | Definition | What it answers |
|---|---|---|
| **Reviewed span** | A stretch of footage a human watched end to end, recorded as data. | *"How much do we actually know about?"* Without it, recall has no denominator and false-events-per-hour has no clock. |
| **Event recall** | matched labelled actions ÷ total labelled actions. | *"What fraction of real events did we surface?"* The metric that must not be sacrificed. |
| **Event precision** | matched events ÷ (matched + false events). | *"How much of what we surfaced was worth surfacing?"* |
| **Temporal IoU (tIoU)** | overlap ÷ union of two time intervals. | *"Do our boundaries agree with the human's?"* Reported at **0.1, 0.3 and 0.5 together**, never at one value. |
| **False events per hour** | unmatched events ÷ reviewed hours. | *"How much noise does a reviewer wade through?"* |
| **Precision@K** | Of the top-K events by score, how many are real. | **The metric a reviewer actually experiences.** An investigator works down a ranked list and stops. |
| **Review burden** | flagged seconds ÷ reviewed seconds. | **The product claim.** A system with perfect recall that flags 90% of the recording has not helped anyone. |
| **Fragmentation** | detections per matched truth event. Ideal = 1.00. | Over-segmentation, which PS2 names as a risk. |
| **Dismissible** | Events matching benign motion or actions the footage cannot show. | Neither a hit nor an error. Reported separately. |
| **Abstention rate** | Fraction of object queries answered `insufficient`. | On this corpus a **high** rate is correct behaviour. |

**Three scoring decisions made explicitly, because each moves the headline number:**

1. **Benign motion is not a false positive.** A candidate stretching produces real motion, and a motion-first system is *right* to surface it. Counting that as an error would reward a duller gate that also misses someone reaching into a bag. This directly addresses PS2's "Differentiating Suspicious and Natural Behavior" challenge.
2. **Actions the footage cannot show are outside the recall denominator** — but an event that fires on one is still not an error.
3. **Matching is one-to-one.** One event spanning three actions found *one* of them. Three fragments over one action score one hit and two false positives.

---

## 5. Final evaluation — the measured numbers

Every figure below was produced by running the system. Nothing here is estimated.

### 5.1 Source corpus (measured, `tools/validate_corpus.py`)

| File | Codec | Resolution | FPS | Duration | VFR | MV/frame |
|---|---|---|---:|---:|---|---:|
| 01 · mobile phone | h264 | 1280×720 | 25.0 | 131.3 s | no | 3647 |
| 02 · mobile phone | h264 | 1280×720 | 25.0 | 212.2 s | no | 3622 |
| 03 · mobile usage | mpeg4 | 1280×720 | **15.0** | 281.9 s | **yes** | 3595 |
| 04 · candidate talking | h264 | 640×480 | 8.0 | 143.3 s | no | 1282 |
| 05 · reception crowd | mpeg4 | 1280×720 | 25.0 | 240.8 s | **yes** | 3581 |
| Seat 12 · paper | mpeg4 | 1280×720 | **21.9** | 88.4 s | **yes** | 3568 |

**Total: 1,097.9 s (18.3 minutes) across four camera views and four dates.**

### 5.2 Throughput

| Measurement | Result |
|---|---|
| Whole corpus, pass 1 | **1,098 s scanned in 17 s — 66× realtime** |
| Longest single file | 42.6–76× realtime |
| Extrapolated to 5 hours | **≈ 4.6 minutes** |
| Hardware used | CPU only, on a laptop with an RTX 3050 4 GB |

The deployment target is a 32 GB RTX 4090. **All timings in this report are therefore floors, not deployment figures.**

### 5.3 Full pipeline run — `03.CCTV Mobile Usage.mkv`, 281.9 s

Models loaded strictly one at a time; verifier started last and torn down in a `finally`.

| Stage | Seconds | Result |
|---|---:|---|
| ingest | 0.3 | VFR handled, 3595 mv/frame |
| calibration | 0.0 | 2 seats, 2 masks |
| motion + segmentation | 6.6 | 529 windows, **8 events**, 42.6× realtime |
| overlay render | 65.6 | 32 MB marked recording |
| evidence clips | 28.7 | 5 |
| detection | 79.5 | 61 detections |
| pose + tracking | 99.4 | **31 observations** |
| verifier | 177.1 | **5 verified, 0 failed** |

**Review burden: 16.3%** of the recording flagged — down from 46.3% before the area-floor fix. All eight events on the seat that was actually occupied.

### 5.4 Detector behaviour on real frames (48 frames, all six cameras)

| | Count |
|---|---:|
| Person detections | 339 (2.2–17.2 per frame) |
| `phone_like` | 23 — **~9 genuine**, 5 keyboard/monitor edge, 3 computer mouse, ~3 ambiguous |
| `paper_like` | 144 |
| Monitors detected as `tv` | 624 |
| Abstentions fired | 125 |

**≈ 40% precision on stock weights** — squarely inside the 43–51% published range for this problem. Median detection footprint 768 px², *larger* than a real phone's 600 px², because the false positives are bigger objects.

**No genuine phone was detected on any seated-candidate camera.** All nine came from a reception view where people stand close to the lens. This independently confirms the 30 × 20 px measurement and is the empirical basis for treating object detection as a bonus signal that abstains rather than a primary claim.

### 5.5 Pose model comparison (8 frames, all six cameras, identical inputs)

| Model | People found | With resolvable pitch | s/frame |
|---|---:|---:|---:|
| **top-down (YOLOX + RTMPose)** | **43** | **36** | 3.74 |
| RTMO-m | 23 | 15 | 1.00 |
| RTMO-l | 25 | 16 | 2.00 |

Going from the medium to the large one-stage model buys **two more people and one more resolvable pitch for double the cost**. Top-down remains **1.72× ahead on person recall and 2.25× on usable head pitch**. The deficit is architectural, not capacity.

Overlays confirm the extra detections are genuine: on the densest frame top-down finds 13 where RTMO-l finds 6, and the seven it adds are individually visible seated candidates deeper in the row.

**This reversed the original design choice**, which had selected RTMO on its CrowdPose leaderboard position. Benchmark rank did not survive contact with 720p CCTV of a CBT hall.

### 5.6 Long-form harness — end-to-end recall

150-second synthesised recording, five planted events at known timestamps, three calibrated seats:

| Metric | Result |
|---|---|
| Event recall | **5 / 5 (100%)** |
| Precision | **1.00** |
| Precision@K (K=5) | **1.00** |
| Mean temporal IoU | 0.574 |
| Fragmentation | **1.00** (one detection per real event) |
| False events | **0** |
| Recording flagged | 11.1% |
| Pose observations | 76 on real people |
| Tracks reconciled to seats | 8 |

**Caveat stated plainly:** these are *synthesised* events composited from real frames. They are the only recall figures currently possible, because the corpus is pre-cut incident clips with no negative time. They are not a substitute for continuous recorded footage with human labels.

### 5.7 Verifier behaviour, live

5 events verified, 0 failures, 177 s on CPU. All five returned `object_assessment: not_visible` — correct abstention at this object scale. Verdict language removed: 0 instances (none produced).

**Working as designed:**
- Scene priming took — the model refers to "standard test equipment"
- Metadata fencing took verbatim — *"The provided pixel statistics describe motion but do not constitute an observation."*
- Seat attribution took after the fix — *"The description is based on the magnified crop of seat 61."*

**Not working, recorded rather than tuned away:**
- Returned `evidence_quality: "sufficient"` on **all five** events at confidence 0.95–1.0. It never once said `"limited"`, on 720p CCTV of a seat roughly 300 px tall. **The confidence figure is not calibrated and must not be presented as one.**
- Its observations are close to interchangeable across the five events.

### 5.8 Validation state

| | |
|---|---|
| Offline gates | **9 suites, all pass** |
| Real-footage gates | **3 suites, all pass** |
| PRD conformance | **74 / 74 checks** |
| Total validation checks | ~250 across 15 gate scripts |

---

## 6. Readiness against PS #2, requirement by requirement

### 6.1 The eight objectives

| # | Objective | State | Evidence |
|---|---|---|---|
| **O1** | Process recorded videos offline | ✅ **Complete** | Six files ingested, checksummed, VFR-validated; 66× realtime; no real-time path exists by design |
| **O2** | Detect motion regions via frame-to-frame motion estimation | ✅ **Complete** | Codec motion vectors at 16×16 granularity across 100% of frames, all six files including MPEG-4 |
| **O3** | Automatically identify ROI containing significant activity | ✅ **Complete** | Per-seat, per-zone robust deviation; 8 events on the correct seat in the reference run |
| **O4** | Identify phones / chits *as may be feasible* | ⚠️ **Partial — and correctly so** | Detector runs, abstains structurally below 300 px². Stock precision ~40%. **Fine-tune blocked on labels.** See 6.4 |
| **O5** | Segment recordings into activity-based clips | ✅ **Complete** | Hysteresis state machine; PTS-accurate clips written per event; fragmentation measured at 1.00 |
| **O6** | Generate motion heatmaps and activity timelines | ✅ **Complete** | `motion/heatmap.jpg`, `motion/timeline.jpg`, plus a marked video and an event ribbon |
| **O7** | Reduce manual review effort | ✅ **Demonstrated, not yet validated** | **16.3% review burden measured.** Against *human-labelled* ground truth: not yet |
| **O8** | Event summaries and visual analytics | ✅ **Complete** | Per-event explanations, score factors surfaced, ranked console, evidence clips, contact sheets, VLM notes |

**Six of eight complete. One partial by the problem statement's own qualifier. One demonstrated but not yet validated against human labels.**

### 6.2 The twelve deployment challenges

| Challenge | Response | State |
|---|---|---|
| **Crowded halls** | Independent state machine per seat; simultaneous events never merge | ✅ Verified in gate |
| **Subtle motion** | 16×16 block granularity, robust per-seat z-scores, `below_desk` feature | ✅ Implemented; sensitivity limits unquantified without labels |
| **Camera noise** | Threshold measured *against* the 0.08 bpp noise floor (2.0 px/frame separates 40 moving blocks from 4) | ✅ Measured |
| **Lighting variations** | Per-window luma tracking; exposure steps suppress rather than escalate | ✅ Implemented |
| **Camera vibration** | Global-motion detection; coherent frame-wide movement suppresses events | ✅ Implemented |
| **Long duration** | 66× realtime; 5 h → 4.6 min; only compact motion metadata retained for static intervals | ✅ Measured |
| **ROI boundary accuracy** | Seat polygons with three zones and a desk line; `screen` excluded entirely | ✅ Implemented |
| **Background motion** | Four mask kinds; **two real instances found and fixed** (a serving window, a glass reflection) | ✅ Demonstrated on real footage |
| **Event segmentation quality** | Hysteresis + min duration + merge gap + max duration | ✅ Fragmentation 1.00, zero false events on harness |
| **Multi-camera sync** | Schema supports it; PTS + burned-in timestamps available | ⚠️ **Not implemented** |
| **Scalability** | SQLite checkpointing, idempotent resume, per-video jobs | ⚠️ Single-machine only; no cluster path |
| **False event generation** | The largest defect found and fixed (D12): 21 → 8 events, all correct | ✅ **Measured** |

### 6.3 The twelve risk factors

| Risk | State |
|---|---|
| False motion detection | ✅ **Directly addressed.** D12 was exactly this risk realised, quantified and fixed |
| Missed ROI detection | ⚠️ Harness recall 5/5; **unmeasured on real labelled events** |
| Over-segmentation | ✅ Fragmentation 1.00 measured |
| Under-segmentation | ✅ `max_duration_s` forces splits; per-seat machines never merge |
| Camera motion effects | ✅ Global-motion suppression implemented |
| Storage constraints | ✅ Static intervals retain only compact metadata |
| Processing delays | ✅ 66× realtime measured |
| ROI localisation errors | ✅ Seat-anchored; **D13 was this risk realised in the verifier and is fixed** |
| Environmental variability | ⚠️ Four camera views handled; per-camera calibration required; generalisation unproven |
| Data security | ⚠️ Local-only, no identity storage, audit log present; **no formal policy or access control** |
| Hardware failures | ✅ Idempotent per-stage checkpointing; resume verified |
| Scalability | ⚠️ Single machine |

### 6.4 Why O4 is partial, and why that is the right answer

PS2 says *"as may be feasible"*. Here is what the measurements say about feasibility:

- A phone at a **near** seat measures **30 × 20 px = 600 px²**. Far seats are smaller.
- Compression is **0.08 bpp** — the object is motion-blurred with no resolvable rectangle geometry and no screen glow.
- Published accuracy on this exact task tops out at **51%**.
- Measured stock-weight precision on this footage: **~40%**, with computer mice the largest confuser.
- **Zero genuine phones detected on any seated-candidate camera.**

A system that answers confidently at this scale is wrong more often than it is useful. So abstention is enforced **structurally** — below 300 px² the output is `insufficient`, and no threshold can be quietly lowered to change that. The verifier is prompted for *posture and interaction*, not object identification, for the same reason.

**This is compliance with O4 as written, not avoidance of it.** The problem statement anticipated exactly this by hedging the objective.

### 6.5 Hardware readiness

| PS2 specification | Target | Development machine used |
|---|---|---|
| GPU: RTX 4070/4080/**4090** | ✅ 32 GB RTX 4090 | ❌ RTX 3050 Laptop, 4 GB |
| RAM 64–128 GB | ✅ Planned | ❌ Consumer laptop |
| Storage 4–20 TB | ✅ Planned | ❌ Local disk |
| llama.cpp CUDA build | ✅ Planned | ❌ **CPU-only build present** |
| PyTorch CUDA | ✅ Planned | ❌ **`torch 2.10.0+cpu`** |

**Consequence:** every timing in this report is a CPU-bound floor. The fine-tune preflight correctly refuses to run in this environment. Nothing has been executed on the actual target hardware.

---

## 7. What we are, and what we are not

### 7.1 What we are

**A working offline investigation-support tool.** Point it at a recording and it returns a ranked, timestamped, seat-attributed, clip-backed, visually-overlaid event list — in minutes, for hours of footage, on a CPU.

**Phase-5-complete without any neural network.** Heatmaps, timelines, event logs, clips, ROI, review console — the entire PS2 deliverable is demonstrable before a single model loads. Every model after that is additive evidence, not a dependency.

**Honest about its own limits, structurally.** Abstention is enforced in code, not by a threshold. The evaluation report *refuses to print percentages* when there is no ground truth. The PRD marks unmeasured claims as targets. `min_area_ratio` exists because an empty seat outscored an occupied one and we went looking for why.

**Measured, not asserted.** Every threshold in `config.py` carries the measurement that produced it. Two model choices were reversed on evidence. The document is machine-checked against the code and the footage at 74/74.

**Privacy-conscious by construction.** Identity is a seat placard. No face recognition, no biometric storage, no cross-session identity. Tracker IDs are session-local and non-biometric.

**Robust to the things PS2 warns about**, with two of them found *in the wild* during this build: background motion from an adjacent room, and a person detected inside a glass reflection.

### 7.2 What we are not

**We are not validated against human labels.** This is the single most important gap. Every accuracy number in this report is either synthetic (the harness) or descriptive (event counts, review burden). **Event recall, false-events-per-hour and Precision@K against real human-annotated ground truth do not exist.** The instrument to produce them is built and tested; the labelling has not been done. Per PRD §15.4, no accuracy claim is presentation-safe until it has.

**We are not a phone detector.** ~40% precision on stock weights, zero genuine phones on seated-candidate cameras, a 51% published ceiling. The fine-tune is driven and preflighted but has never run — it needs box labels and a CUDA build of PyTorch.

**We have never run on the target hardware.** No 4090, no CUDA llama.cpp, no CUDA PyTorch. All timings are CPU floors and all VRAM planning is on paper.

**We have not measured false-events-per-hour on continuous footage.** The corpus is pre-cut incident clips: every frame is an event, so there is no negative time. The harness synthesises the missing denominator, but that is a stand-in.

**We are not multi-camera.** The schema supports it. The correlation logic does not exist.

**We are not calibrated on confidence.** The verifier says `sufficient` at 0.95–1.0 every time and never says `limited`. That number should not appear in any interface as if it means something.

**We are not automatically calibrated.** Every camera needs its seat polygons, zones, desk lines and masks authored by hand. Auto-calibration does not exist, and the two mask defects found in this build show that manual calibration is also *error-prone*.

**We do not generalise beyond what has been seen.** Four camera views, four dates, one room type (CBT hall). Behaviour on a paper-examination hall, a different mounting geometry or a different compression profile is unknown.

**We have no security or retention policy.** Local-only storage and an audit log are not a compliance posture.

### 7.3 The honest one-line summary

> **The engine is built, measured, and defensible. The measurement of whether it is *accurate* has not been done, and cannot be done without a person watching footage and writing down what happened.**

---

## 8. What it would take to close the gap

In priority order, with an honest note on what each unblocks.

| # | Work | Blocks | Effort |
|---|---|---|---|
| **1** | **Temporal annotation** of a contiguous span — review spans plus labelled actions with seat, type and visibility | Event recall, precision, tIoU, false-events/hour, Precision@K, review-burden validation — **everything in §15.2** | Human hours. Tool is built (`annotate_timeline.py`) |
| **2** | **Box annotation** on the extracted frames, negatives first | The detector fine-tune, and therefore any object claim at all | Human hours. Tool is built (`annotate_boxes.py`) |
| **3** | **Run on the 4090** with CUDA builds of PyTorch and llama.cpp | Real throughput figures; the fine-tune; VRAM validation | Environment setup |
| **4** | **Detector fine-tune** from Objects365 weights on annotated seat crops | Whether O4 can move from "abstains honestly" to "detects usefully" | GPU hours; driver is built and preflights |
| **5** | **Continuous recorded footage** — several unbroken hours, not pre-cut clips | Honest false-events-per-hour; the harness stops being a stand-in | Depends on the data provider |
| **6** | **Verifier confidence calibration** | Making the confidence number safe to display | Needs ground truth first |
| **7** | Multi-camera correlation | Cross-camera evidence fusion | Design work; schema exists |
| **8** | Auto-calibration or a calibration QA pass | Deployment at scale; the two mask defects show manual calibration is error-prone | Design work |

**Items 1 and 2 are not engineering.** They are a person watching video and writing down what they see. Everything else in this list is either blocked behind them or is environment work.

---

## Appendix A — Defect index

| ID | Class | Defect | Section |
|---|---|---|---|
| D1 | 🔴 | Baseline absorbed the events it should find | 3.1 |
| D2 | 🔴 | MAD floor absolute where it must be relative | 3.1 |
| D3 | 🔴 | Relative floor collapses on static seats | 3.1 |
| D4 | 🔴 | Global luma sampling fabricated exposure steps | 3.1 |
| D5 | 🔴 | Backdating re-absorbed max-duration splits | 3.1 |
| D6 | 🔴 | Backdated events stuck in CANDIDATE | 3.1 |
| D7 | 🔴 | Rolling-max smoothing destroyed impulse rejection (rejected fix) | 3.1 |
| D8 | 🔴 | Seat-overlap false positive from `fillPoly` boundary | 3.1 |
| D9 | 🔴 | `persist` erased the other model's evidence stream | 3.1 |
| D10 | 🔴 | Scrubber seeked past the head of a file, silently | 3.1 |
| D11 | 🔴 | Win32 arrow keys collapsed to one value | 3.1 |
| **D12** | 🔴 | **Empty seat outscored the occupied one (29.8σ vs 9.5σ)** | 3.1 |
| **D13** | 🔴 | **Verifier described the wrong person at confidence 1.0** | 3.1 |
| **D14** | 🔴 | **Harness never emitted a motion vector** | 3.1 |
| D15 | 🔴 | Four GBNF grammar defects | 3.1 |
| D16 | 🔴 | Observation truncation and filler padding | 3.1 |
| D17 | 🟠 | PyAV 18 API changes | 3.2 |
| D18 | 🟠 | `Path().glob()` rejects absolute patterns | 3.2 |
| D19 | 🟠 | `llama-server.exe` exists but cannot launch | 3.2 |
| D20 | 🟠 | OpenMP double-load aborts `import torch` | 3.2 |
| D21 | 🟠 | `score_recall` KeyError on zero detections | 3.2 |
| D22 | 🟡 | PRD frame rates wrong on two of six files | 3.3 |
| D23 | 🟡 | §5.1 table 9 headers, 10 cells | 3.3 |
| D24 | 🟡 | §9.2.2 wrong heading depth | 3.3 |
| D25 | 🟡 | §9.3 missing context size | 3.3 |
| D26 | 🟡 | §8.2 declared a decision §8.2.1 had reversed | 3.3 |
| D27 | 🟡 | Hard negatives written for the wrong room | 3.3 |
| D28 | 🟡 | Stock detector precision quantified at ~40% | 3.3 |
| D29 | ⚪ | I reported the corpus unreachable; only checked D: | 3.4 |
| D30 | ⚪ | Two `prd_check` failures were my own bad regexes | 3.4 |
| D31 | ⚪ | Off-by-one slice bounds in my own gate | 3.4 |
| D32 | ⚪ | Wrote a wrong hypothesis into code before testing it | 3.4 |
| D33 | ⚪ | First displacement fix was phase-wrong | 3.4 |

**Distribution:** 16 silent-wrong-output · 5 crash/blocking · 7 documentation · 5 my own.

**The pattern worth noting:** sixteen of thirty-three would have produced *wrong output while reporting success*. Not one of them was found by reading code. Every single one was found by running something against real data and looking at the result.

---

## Appendix B — Reproducing every number in this report

```bash
# Corpus inventory and the PRD's own claims about it
python tools/validate_corpus.py C:/DrishtiAI
python tools/prd_check.py --corpus C:/DrishtiAI      # 74/74

# Full pipeline, one video, into a self-contained folder with overlays
python tools/run_video.py "C:/DrishtiAI/03.CCTV Mobile Usage.mkv" \
    --name mobile03 --llama-binary tools/llamacpp/llama-server.exe

# Model comparisons on real frames
python tools/pose_ab.py data/frames --models topdown rtmo-m rtmo-l
python tools/detector_sweep.py data/frames

# Ground truth, once someone has labelled some
python tools/annotate_timeline.py VIDEO --annotator <name>
python tools/annotate_boxes.py data/frames --annotator <name>
python -m classroom.cli coverage
python -m classroom.cli evaluate

# Every validation gate
for t in tests/test_*.py; do python "$t"; done
```
