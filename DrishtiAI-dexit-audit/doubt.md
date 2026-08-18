# doubt.md — open questions, conflicts, and decisions taken

Everything here needs your eyes. Nothing in this file is a blocker I have worked
around silently; where I had to pick a direction to keep moving I have said so
and marked it **[assumed]**, and it can be reversed cheaply.

Raised against **Product Zero Implementation PRD** (`PRD.md`, RTX 3090 handoff)
on 18 August 2026, comparing it to the existing build at commit `481745d`.

**Legend**
✅ **Answered** — you have decided; recorded here and implemented
🔴 **Blocking** — the PRD itself says to fail rather than guess
🟠 **Conflict** — the new PRD contradicts the old one, or contradicts something we measured
🟡 **Decision taken** — I picked a direction to keep moving; reversible
🔵 **Question** — still open

---

## 0. Answers received — 18 August 2026

Recorded verbatim where you gave wording, and applied throughout the sections
below. Items you settled are marked ✅ in place; only the genuinely open ones
still carry 🔵.

### 0.1 ✅ Hardware: 3090 now, 32 GB 4090 as the architecture target

> "Since my laptop only supports 3090, we are doing this for now, but on the main
> architecture implementation, once we are ready with everything and we will
> scale it later, we would be targeting a 32 GB 4090. This would allow everything
> to run simultaneously together, which involves multiple cameras, all of the
> models running, batch processing, streaming, and everything. But that is for
> the future scope."

**Implemented as:** sequential residency is the *current* execution mode and the
one everything is built and measured against. Co-residency is not deleted — it
stays as a configuration flag that preflight will refuse on a 24 GB card and
permit on 32 GB. The resource report records which mode a run used, so a future
4090 run is directly comparable to a 3090 one rather than being a fresh baseline.

**One factual note, not an objection.** The machine these measurements were taken
on reports **RTX 3050 Laptop, 4 GB** via `nvidia-smi`, with CPU-only builds of
both PyTorch and llama.cpp. If the 3090 is a different machine, preflight will
simply report whatever it finds there. Nothing needs changing — I am recording it
so that a resource report showing 4 GB is not later read as a measurement error.

### 0.2 ✅ 30–60 people is a design target, not a claim

> "1–17 people is the validated corpus range. '30–60 people' is a
> capacity/design target, not a result we can claim from this corpus. The final
> report must state that plainly."

**Implemented as:** the dense-hall code paths are built to spec. The final report
carries a mandatory `validated_density_range` field populated from what was
actually observed, and any dense-hall path reports
`validated_on_fixture_only`. The report template refuses to emit a density claim
that exceeds the measured range.

### 0.3 ✅ Process the entire available video; claim nothing about duration

> "Process the entire available video. The clips are short and pre-cut, but they
> are still processed end-to-end; this does not justify claiming long-recording
> or false-events/hour validation."

**Implemented as:** full decodable duration is processed and accounted for, per
PRD 18. `FP/hour` and `event/hour` are still *computed* — PRD 15 requires them —
but each is emitted alongside `denominator_is_representative: false` with the
reason, so the number cannot be quoted without its caveat.

### 0.4 ✅ Missing reference manifest and taxonomy block evaluation, not Phase 0

> "Missing reference manifest and approved taxonomy block detector
> selection/evaluation, not Phase 0. Health checks, artifacts, calibration QA,
> motion, and segmentation can proceed."

**Implemented as:** the blocking scope is narrowed to exactly stages 09 (detector
selection) and 12 (evaluation). Phases 0–3 run without either. This matches PRD 3
literally — the reference-manifest row says *"Run inference but mark object
evaluation incomplete"*, not "block the run".

### 0.5 ✅ Privacy policy blocks export, not engineering

> "No privacy policy should block export/distribution, not local engineering
> work."

**Implemented as:** local artifacts are written freely. There is no export path
at all, and the run folder carries an `export_blocked_no_policy` marker so that
anyone who later builds one hits the block deliberately.

### 0.6 ✅ SAHI is a comparison, and may lose

> "SAHI is a required comparison, not a mandated final detector path. If it
> performs worse or costs too much, the measurement report should reject it."

**Implemented as:** all four object configurations are run and reported. The
selection report is permitted to recommend against SAHI, and the recommendation
template has no "winner required" constraint — PRD 15 already says "No forced
winner".

### 0.7 ✅ Old defects become permanent regression tests

> "Carry old defects forward as regression tests. The absolute moving-block
> issue, rolling-max issue, and wrong-person VLM issue are not merely PRD
> requirements—they must become permanent tests."

**Implemented as:** `tests/test_regressions.py`, holding the *measured* values
from each original failure, not a paraphrase of the rule. Each test names the
defect, carries the numbers that were observed when it fired, and fails if the
guard is removed. Three at minimum, and any future defect of this class joins
them.

### 0.8 ✅ Do not migrate wholesale; adapt, then replace behind tests

> "Do not migrate classroom/ → pipeline/ wholesale. Build Phase 0 as compatible
> adapters/run-manifest/artifact infrastructure around the working code, then
> replace modules gradually behind tests."

**Implemented as:** `pipeline/` holds the new infrastructure — run manifest,
artifact folder, config validation, preflight, gates — plus thin adapters that
call the existing, measured `classroom/` code and write its output into the PRD
13 artifact layout. No working module is rewritten until a test covers its
replacement. The port table in §2.5 below becomes a *replacement order*, not a
migration plan.

---

---

## 0.9 🔵 Measurement correction found while building Phase 1

Two defects and one correction came out of the first complete-decode pass. The
correction needs your awareness because it changes a number in the previous
report.

**Four of six corpus files are variable frame rate, not three.** The previous
implementation judged VFR from the standard deviation of PTS deltas and
classified `04.CCTV Candidate Talking.mkv` as constant rate. Measuring the
coefficient of variation of inter-frame intervals says otherwise, and the
interval histogram settles it: **144 of its frames are held for 160 ms against a
120 ms majority** — a third longer, in two exactly discrete durations. That is
variable frame rate by any definition.

The two populations separate by a factor of 150 with nothing between them, so
this is not a threshold judgement call:

| File | cv | Verdict |
|---|---:|---|
| 01 mobile phone | 0.0000 | CFR |
| 02 mobile phone | 0.0007 | CFR |
| **04 candidate talking** | **0.1061** | **VFR — previously called CFR** |
| 05 reception crowd | 0.2041 | VFR |
| Seat 12 paper | 0.3131 | VFR |
| 03 mobile usage | 0.3571 | VFR |

**Nothing downstream was wrong because of it** — the pipeline reads PTS
everywhere regardless of the flag, which is exactly why that rule exists. But
the old PRD §5.1 table and the build report both say three files, and both will
be corrected.

**Two defects of mine, both found by running the decode:**

- **The complete-duration requirement was being violated by 11 frames per
  file.** The decode loop constructed a fresh generator on every iteration, so
  the decoder's internal buffer was never flushed and the last 11 frames of
  every file were dropped. Under 1% and invisible in any summary — and a direct
  violation of PRD 18's first line. Fixed; all six files now decode to exactly
  the frame count a plain full decode produces.
- **End-of-stream was being recorded as a decode error.** PyAV surfaces EOF from
  `avcodec_send_packet` as `EOFError`, which was landing in the error list, so
  every perfectly readable file reported one failure. Fixed.

---

## 1. 🔴 Blocking inputs that do not exist

§3 lists six blocking inputs. Four of them are absent today. The PRD is explicit
that implementation "must fail clearly instead of guessing" — so the code will
refuse, and these become your critical path.

### 1.1 🔴 Reference manifest — does not exist

§3 and §14 require an independently maintained JSONL/CSV of known phone/chit
sightings with `source_video_sha256`, exact `source_pts_ms`, class, and source
bounding boxes.

**We have none.** No frame of this corpus has ever been annotated by a human.

Consequence per §3: *"Run inference but mark object evaluation incomplete; make
no quality claim."* So the pipeline will run end to end and stage 12 will report
`evaluation_incomplete`. **Every metric in §15 — precision, recall, F1, FP/hour,
stage-loss waterfall — is unobtainable until this exists.**

This is the same gap the previous build reported. It has not moved, and no
amount of engineering moves it. It needs a person watching footage.

**Question:** who is producing it, in what tool, and against which videos? I have
annotation tooling already built (`tools/annotate_timeline.py`,
`tools/annotate_boxes.py`) that can emit this schema with a small adapter — say
the word and I will write the adapter.

### 1.2 🔴 Taxonomy approval — not approved

§3: *"Product-owner-approved mapping of target and hard-negative objects… If
absent: block detector selection/evaluation."*

§10 gives a **provisional** taxonomy "requiring owner confirmation":

```
phone · secondary_paper_chit · answer_sheet · calculator · mouse · keyboard
hand · bag · monitor_or_bezel · stationery · unknown_object
insufficient_visual_evidence
```

**Two notes from measurement, not opinion:**

- The existing build measured the actual confusers on this exact footage: **computer
  mice above all, then keyboards and monitor bezels**. All three are in your list.
  Good — the list matches the failure modes.
- `answer_sheet` vs `secondary_paper_chit` is the hardest distinction in the set
  and may not be separable at this source scale. A chit is small paper; an answer
  sheet is large paper. At 720p from a ceiling mount, a folded answer sheet and a
  chit can be the same object footprint.

**Question:** please confirm or amend the list. Until you do, detector selection
is blocked by design and I will keep the harness running but refuse to name a
winner.

### 1.3 🔴 Privacy policy — does not exist

§3: *"Access, retention/export/delete rules for CCTV and crops. If absent: block
distribution/export beyond execution owner."*

**Assumed** 🟡: everything stays on the execution machine. No export path is
built. The artifact directory is local-only. Nothing is uploaded, and the run
folder is not shareable until you provide the policy.

### 1.4 🔴 RTX 3090 environment — we do not have one

§3 requires CUDA runtime, weights, license status, writable capacity,
launchable models.

**The machine this was developed on is an RTX 3050 Laptop with 4 GB VRAM**, a
**CPU-only** PyTorch build (`torch 2.10.0+cpu`) and a **CPU-only** llama.cpp
build. The PRD's entire §5 resource plan — 24 GB, sequential residency, 2 GB
safety margin — cannot be exercised here.

**What I will do** 🟡: `tools/preflight_rtx3090.py` will detect this and mark
every GPU configuration `resource_unavailable` rather than silently downscaling,
exactly as §5 requires. The pipeline stages that do not need a GPU will still
run and produce real artifacts.

**Question:** when do we get access to the 3090, and is it the same machine that
holds the source video?

---

## 2. 🟠 Conflicts between this PRD and the previous one

### 2.1 🟠 Target hardware changed: 4090 32 GB → 3090 24 GB

| | Old PRD (v3.2) | New PRD (Product Zero) |
|---|---|---|
| GPU | RTX 4090, 32 GB | **RTX 3090, 24 GB** |
| Residency | "co-resident, 32 GB has ample room" | **strictly sequential, 2 GB margin** |

This is a real change and it invalidates a design decision. The old PRD §14
explicitly removed "sequential-residency logic" as unnecessary machinery on a
32 GB card. **At 24 GB with Gemma, D-FINE, RF-DETR, three pose models and a
tracker, sequential residency is mandatory** and that machinery has to come back.

**Decision taken** 🟡: implementing sequential-only residency with explicit
release between phases and a configurable VRAM safety margin, defaulting to
2 GB per §5.

**Question:** is 24 GB now the true target, or is the 3090 a development stand-in
for a 4090 deployment? The answer changes whether co-residency stays available as
an option.

### 2.2 🟠 SAHI was rejected on measurement; it is now mandatory

The old PRD §3.4 rejected full-frame SAHI slicing with a measured argument: at
1280×720 into a 640 input, slicing gains little, and native-resolution seat crops
upscaled into the detector are where small-object recall comes from.

The new PRD §5 and §10 **require** `rf_detr_sahi` and `dfine_sahi` as mandatory
comparison configurations.

**These do not actually conflict, and I want to flag that clearly** because it
would be easy to read them as contradictory. The new PRD is careful:

> "SAHI: only on the same **candidate crop**, never unconditional 3×3/10×10
> full-video grids"

That is candidate-crop-level slicing, which is a *different thing* from the
full-frame slicing the old PRD rejected. The new framing is better: it makes
SAHI a measured comparison rather than an assumption in either direction.

**No action needed from you** — noting it so nobody later thinks we reversed a
measured decision without reason.

### 2.3 🟠 "30–60 people" does not match the available footage

§ Coverage says 30–60 people, dense hall, occlusion-heavy.

**Measured on the actual corpus:** 2.2 to 17.2 person detections per frame across
48 frames spanning all six source files. The densest single frame had 13 people
by top-down pose. Most seated-candidate frames have **one to three people**.

The dense-hall assumptions — adjacency graphs, ambiguous boundaries, crowd
occlusion, tracker identity switching — are all correct requirements for the
stated deployment, but **the footage we have cannot exercise them**. Anything I
build for density will be untested against density.

**Question:** is there denser footage available? If not, I will build to the spec
and mark the dense-hall paths as `untested_at_stated_density` in the final
report rather than implying they were validated.

### 2.4 🟠 "Complete recorded video" vs a corpus of pre-cut clips

§1: *"The pipeline must process the full decodable duration before evaluation."*

**What we have:** six pre-cut incident clips totalling 18.3 minutes. The longest
is 4 min 42 s. Every frame of them is, by construction, near an incident.

This matters more than it sounds. There is **no negative time** in this corpus,
so `FP/hour`, `event/hour` and `candidate duration / video duration` — all
required by §15 — are measured against a denominator that does not represent a
real examination recording.

**Question:** can we get one continuous multi-hour recording? Without it, several
required metrics are computable but not meaningful, and I will label them so.

### 2.5 🟠 Module layout: `classroom/` exists, PRD specifies `pipeline/`

§16 names exact paths: `pipeline/ingest.py`, `pipeline/motion.py`,
`pipeline/segmentation.py`, and so on. The current build has equivalents under
`classroom/`.

**Decision taken** 🟡: I am building `pipeline/` as specified, and **porting**
the parts of `classroom/` that already satisfy the new requirements rather than
rewriting them. Roughly:

| New PRD file | Existing source | State |
|---|---|---|
| `pipeline/ingest.py` | `classroom/ingest.py` | Ports well; add keyframe/rotation/colour metadata, repeated-PTS and gap detection |
| `pipeline/timestamps.py` | inside `classroom/ingest.py` | Needs extracting into its own module |
| `pipeline/health.py` | — | **New.** Only 2 of 10 conditions exist today |
| `pipeline/calibration.py` | `classroom/calibration.py` | Ports; needs epochs, approval workflow, adjacency graph, more mask kinds |
| `pipeline/alignment.py` | partial in `classroom/motion.py` | **Mostly new.** We have global-motion *detection*, not compensation |
| `pipeline/motion.py` | `classroom/motion.py` | Ports well — including the area-fraction rule §6.3.8 already demands |
| `pipeline/baseline.py` | inside `classroom/motion.py` | Extract |
| `pipeline/segmentation.py` | `classroom/segment.py` | Ports well; add epoch/gap bridging rules and transition-reason logging |
| `pipeline/gates.py` | — | **New.** Five-state protocol does not exist |
| `pipeline/crops.py` | `classroom/crops.py` | Ports well |
| `pipeline/pose.py` | `classroom/pose.py` | Ports; add AlphaPose |
| `pipeline/object_detection.py` | `classroom/detect.py` | Ports; add RF-DETR |
| `pipeline/sahi_adapter.py` | — | **New** |
| `pipeline/gemma_review.py` | `classroom/verify.py` | Ports; schema differs, see §4.1 below |
| `pipeline/evaluate.py` | `classroom/evaluate.py` | Substantial rework — new PRD demands stage-loss attribution |

**Question:** do you want `classroom/` kept as-is alongside `pipeline/` during the
port, or removed once each piece is migrated? I am keeping it for now so nothing
is lost, and it can be deleted in one commit when the port is complete.

---

## 3. 🔵 Three of our hardest-won findings are already rules in your PRD

Worth saying explicitly, because it is a strong signal that the two documents
agree on what matters. These are not questions — they are confirmations.

| Your PRD rule | What we independently measured |
|---|---|
| §6.3.8 *"require moving **area fraction** plus persistence. Absolute moving-block count alone is forbidden."* | An unoccupied desk row produced the **twelve highest-scoring events** in a recording at 29.8 σ against 9.5 σ for the one occupied seat, because an absolute block count passed a region carrying 0.1% area motion. Fixed with an area-fraction floor of 0.05; events fell 21 → 8, all correct. |
| §8 *"Rolling-max smoothing that invents duration from a spike is prohibited."* | We tried exactly that smoothing to bridge intermittent hand motion, measured that it turned a dropped pen into a 3-second incident, and replaced it with gap bridging. |
| §11 *"Whole-frame-only packets are prohibited"* and *"marked/labeled subject seat, magnified native subject crops"* | Gemma described a candidate in the **wrong seat** at `confidence: 1.0` because the contact sheet was six whole frames and the seat was named only in metadata text. Fixed by drawing the seat and appending a magnified crop. |

I have carried all three forward into the new implementation, with their
measurements attached.

---

## 4. 🔵 Contract changes that need confirming

### 4.1 🔵 Gemma output schema differs from what is built

| Built today | New PRD §11 |
|---|---|
| `supported_observations` | `observable_actions` |
| — | **`subject_marker_visible`** |
| `object_assessment` | `object_assessment` |
| — | **`object_certainty`** (`not_visible\|possible\|supported`) |
| `evidence_quality` | `evidence_quality` (same three values) |
| `confidence` (0–1 float) | **removed** |
| — | **`quality_limitations`** |
| — | **`needs_human_review`** |
| `review_note` | `reasoning_note` |
| `interaction_assessment` | **removed** |

**I think dropping `confidence` is correct and I want to record why.** Measured
live: the verifier returned `evidence_quality: "sufficient"` at confidence
**0.95–1.0 on every single event**, and never once said `limited`, on 720p CCTV
of a seat roughly 300 px tall. That number was not calibrated and should never
have been displayed. Replacing it with `object_certainty` as a three-value
enumeration is strictly better — an enumeration cannot fake precision.

**Decision taken** 🟡: implementing the new schema. `subject_marker_visible` is a
particularly good addition — it makes the D13 failure above *self-reporting*.

### 4.2 🔵 `alphapose_wholebody` — which checkpoint?

§5: *"only if exact checkpoint passes preflight."*

**Question:** which checkpoint, from where? AlphaPose whole-body weights are
distributed in several variants and some carry a **non-commercial research
licence**. §3 requires license status to be recorded, so I need the exact
artefact you intend to use.

### 4.3 🔵 RF-DETR — which implementation and weights?

§5 requires `rf_detr_native` and `rf_detr_sahi`. RF-DETR ships in several sizes
(base/large) from Roboflow.

**Question:** which variant, and are we permitted to download weights on the
execution machine, or do they need to be staged in advance?

### 4.4 🟡 "jammu" → Gemma

Your §3 already resolves this: *"'Evaluate flagged frames against jammu' is
interpreted as **Gemma**."* Recording that I have taken it at face value and
built against Gemma. If Jammu is a real separate tool, this is the one place it
would need to change.

---

## 5. 🔵 Things the PRD requires that have no test data

I can build all of these to spec. None of them can be *validated* against this
corpus, and I would rather say so now than produce a green test that means
nothing.

| Requirement | Why it cannot be exercised |
|---|---|
| `blackout` / `whiteout` | No corpus file contains either |
| `frozen_video` | No corpus file freezes |
| `decode_gap` | All six files decode cleanly end to end |
| `camera_repositioned` / epochs | No file contains a reposition |
| `camera_motion` compensation | All six cameras are rigidly mounted; no vibration observed |
| Tracker identity switching | Needs crowd density we do not have (§2.3) |
| Multi-hour scalability | Longest file is 4 min 42 s |

**Decision taken** 🟡: I will build each condition detector with **synthetic
fixtures** that inject the condition into real frames — the same approach that
caught a real defect in the previous build, where a synthetic harness turned out
never to exercise the motion-vector path at all. Fixtures prove the detector
fires; they do not prove the threshold is right for real footage. The final
report will distinguish `validated_on_source` from `validated_on_fixture_only`.

**Question:** if any real footage exists with these conditions — even from a
different site — it is worth more than any fixture I can write.

---

## 6. 🔵 Smaller items

### 6.1 🔵 §7 requires reviewed calibration with operator and reviewer

The current build has seat polygons **hard-coded in a Python dict** in the run
tool, traced by eye. The new PRD requires a versioned profile with operator,
reviewer, approval status, adjacency graph, ambiguous boundaries, and review
images under four named conditions.

**Question:** who is the approving reviewer? The PRD makes calibration approval a
gate on seat attribution, so this is a named human role, not a config field.

### 6.2 🔵 Run folder is immutable and reprocessing creates a new run ID

§13. This is right, and it means disk grows per run. Our current single run
produced 96 MB including a 32 MB overlay video. At the artifact density this PRD
requires — every slice, every raw and merged detection, every alignment
diagnostic — **I would estimate 1–5 GB per hour of source video.**

**Question:** what is the storage budget, and where does the artifact root live?
§3 requires writable capacity to be a preflight check, so I need a number to
check against.

### 6.3 🔵 §15 requires "rank of every reference"

This is a good metric and I want to confirm I read it right: for each known
phone/chit sighting, report **where in the ranked event list** it appeared — so
we can say "every real incident was in the top N". That is a much better
statement of product value than precision/recall alone.

**Question:** confirm, and tell me the N you care about. Top 10? Top 5% of
events?

### 6.4 🟡 Seed and determinism

§5 requires a recorded seed and §17.10 requires versioning and hashing every
threshold, input, slice, model and seed. **Decision taken** 🟡: a single run-level
seed propagated to every stochastic component, recorded in `RUN_MANIFEST.json`,
with the run refusing to start if any configured model does not accept it.

---

## 7. What I am doing next, absent answers

Working in the PRD's own mandated order — §4's "comparison-first order" exists
precisely so that a threshold change cannot later be credited to a model.

1. **Phase 0** — run manifest, artifact directory, config validation, preflight.
   This is fully buildable now and blocks everything else.
2. **Phase 1** — ingest, timestamps, health. Buildable now; health conditions
   get fixtures per §5 above.
3. **Phase 2** — calibration and alignment. Buildable, but seat attribution stays
   blocked until a reviewed profile exists (§6.1).
4. **Phase 3** — motion, ROI, baseline, segmentation, gates. Largely a port of
   working, measured code plus the new five-state gate protocol.

Phases 4–7 need the environment (§1.4), the taxonomy (§1.2) and the reference
manifest (§1.1). I will build the harnesses so they are ready to run the moment
those land, and they will refuse to report a winner until they can.
