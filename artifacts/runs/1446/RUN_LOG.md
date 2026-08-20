# Run 1446 — full log

Three videos, locked architecture (AlphaPose + D-FINE native/SAHI, no VLM),
executed on GPU. This log records what each stage did, what was marked, what
broke, and what was corrected — including the corrections made *during* the
run, because several results changed as a result.

**Hardware:** RTX 3050 Laptop, 4096 MiB. torch 2.11.0+cu128, onnxruntime-gpu
1.22.0, both on `CUDAExecutionProvider` (verified by `session.get_providers()`,
not by the available-providers list).

---

## Videos processed

| run dir | source | duration | resolution | sha256 prefix |
|---|---|--:|---|---|
| `01_phone` | 01.Candidate was found using a mobile phone… | 131.3 s | 1280×720 @25 | `b842b03e5784d867` |
| `04_talking` | 04.CCTV Candidate Talking | 143.2 s | **640×480 @8** | — |
| `12_paper` | Seat No. 12 was seen taking a piece of paper… | 88.4 s | 1280×720 @25 | — |

Not processed: 02 (52 MB), 03 (66 MB, ran earlier as `corpus_03`), 05 (317 MB),
06 (5.83 GB), exchanged-seats (2.83 GB).

---

## Stage timings (GPU)

| stage | 01_phone | 04_talking | 12_paper |
|---|--:|--:|--:|
| preflight | 54.4 s | 66.5 s | 30.4 s |
| ingest | 24.0 s | — | — |
| preprocessing A/B | 74.1 s | 33.4 s | 15.9 s |
| alignment A/B | 23.3 s | 18.6 s | 11.5 s |
| health + motion + ROI + segmentation | 27.0 s | 34.0 s | 13.6 s |
| **tracking + pose A/B @3 Hz on CUDA** | **307.2 s** | **495.5 s** | — |
| head/neck + overlays | 5.4 s | 8.0 s | — |
| **crops + D-FINE native/SAHI** | **635.5 s** | **1658.4 s** | **2531.4 s** |
| evidence + ranking | 329.9 s | 35.1 s | 35.2 s |
| evaluation | 21.5 s | 20.3 s | 27.4 s |

Preflight reports 11/12 configurations available; the missing one is
`gemma_marked_evidence` (`missing_weights` — the VLM was deliberately deleted).
It is recorded unavailable, never substituted.

---

## Results

| | 01_phone | 04_talking | 12_paper |
|---|--:|--:|--:|
| pose manifest rows (3 Hz) | 1,076 | 3,542 | 5,881 |
| AlphaPose coverage | **100.0%** | **100.0%** | — |
| RTMPose coverage | 88.8% | 81.4% | — |
| RTMO coverage | 50.6% | **12.5%** | — |
| head observations resolvable | 77.2% | 58.3% | — |
| **head excursions** | **1** | **5** | **4** |
| crops | 1,076 | 3,542 | 5,881 |
| `dfine_sahi` raw phone | 416 | 4,188 | 2,910 |
| `dfine_sahi` raw book/notebook | 111 | 257 | **3,092** |
| `secondary_paper_chit` | **0** | **0** | **0** |
| phone groups surviving all filters | **42** | **69** | **55** |
| `recurrent_scene_object` | 6,235 | 28,662 | 33,417 |
| ranked events | 46 | 38 | 29 |
| **reaching a reviewer** | **15** | **10** | **7** |

Every configuration processed byte-identical crop and manifest rows
(`crop_equivalence: verified`), so differences are attributable to the model.

---

## What was marked, and why

**`recurrent_scene_object`** — restated because the class recurs at one image
location as an outlier against the median cell for that class, over ≥5 s. The
group keeps its frames, confidence and crops; only its status changes, and
`reason` records the count and span. Largest single case on `12_paper`: a desk
edge with **recurrence 122 across 86.0 s**.

**`handheld_implausible`** — nearest wrist beyond 2.0 torso widths. 11,074
groups on `12_paper`, 6,189 on `04_talking`. `handheld_unjudged: 0` on
`12_paper`, meaning pose resolved a wrist for every group.

**`rejected_not_offered`** — zero across all three videos. No quality gate
withheld anything.

---

## Defects found and corrected during this run

1. **`classroom/` deleted as "superseded", but it is the compute layer.**
   `pipeline/motion.py`, `object_detection.py`, `pose.py`, `person_detection.py`
   and `crops.py` all delegate to it — including `detect.ONNXDetector`, which is
   how D-FINE runs. The run failed at the motion stage with
   `ModuleNotFoundError`. Restored from git. *Inbound imports were checked after
   deletion rather than before.*

2. **Head excursion merge gap shorter than the sample period.** A fixed 800 ms
   gap under 1 Hz sampling split every run into singletons, so no event could
   reach the 1.5 s minimum. Measured: **1 event across 379 observations of 29
   tracks**. Gap now derived per track from the observed sampling interval.

3. **Head summary keyed on `seat_state`**, which is `unattributed` for every row
   under a draft calibration — pooling every person in the hall into one
   baseline. Now keyed by track.

4. **Yaw threshold unreachable.** Yaw saturates at ±1.0; the distribution is
   bimodal, so 3×MAD put the threshold at **+1.093**, above saturation. Zero
   events formed across 42 subjects while the same rows classified 360 frames
   `turned_left`. Capped at the absolute turn threshold.

5. **…which then flagged resting posture.** Candidates face their monitors while
   the camera sits at an angle, so a permanently turned head is normal. Video 04
   reported **45 `yaw_right`**, 9 with baseline exactly equal to peak. An
   excursion now requires **both** a turn past the absolute threshold **and** a
   ≥0.30 departure from that subject's own baseline. Video 04: **53 → 5**.

6. **Temporal confirmation promoted furniture.** A monitor at (275,650) on video
   01 produced **23** `temporally_supported` phone groups across 38.4 s; the one
   genuine phone produced **2**. Added `flag_recurrent` and `flag_handheld`.

7. **Recurrence grid was absolute pixels.** 25 px gives 1,479 cells on 1280×720
   but only **494** on video 04's 640×480 — four times the relative area — and
   flagged **97%** of that video's groups (35,131 of 36,332). Grid is now 2% of
   frame width, and the count threshold is an outlier test against the median
   cell rather than a fixed number.

8. **Overlay frames not cleared between passes.** 8 stale images sat beside 1
   live event, and the findings report lists whatever is in the directory. Now
   cleared first; counts match exactly.

9. **`render_detections.py` drew raw detections, not survivors.** The contact
   sheets showed exactly what the filters had rejected. Added `--from-confirmed`.

10. **Re-running ranking in place duplicated events.** `append_jsonl` is
    append-only by design — a stage re-run is meant to create a *new* run.
    01_phone went to 92 rows for 46 events, 04_talking to 76 for 38.
    Deduplicated keeping the later pass.

11. **Appended tests landed after the `__main__` block** and were never
    executed. Runner moved to end of file.

---

## Verified findings

**Video 01 — the phone is real.** 69.17 s, confidence **0.933**, 39×40 px,
bright motion-blurred object in a hand. Recurrence 2, span 0.0 s, **0.244 torso
widths from a wrist**. Highest-confidence detection in the recording and the top
survivor after both filters. Frames:
`09_object/dfine_sahi/rendered_phone/`.

**Video 04 — a repeated-glance pattern.** `track_85`, baseline −0.217 (near
frontal): `yaw_right` at 90.8 s (4.12 s), `yaw_left` at 98.4 s (2.45 s),
`yaw_left` at 103.9 s (2.49 s). Three sustained departures alternating direction
inside 13 seconds, on a recording titled *Candidate Talking*. Frames:
`08_pose/head/overlays/`.

**Video 12 — paper is detected but not nameable.** 3,092
`book_or_notebook_like_object` and **zero** `secondary_paper_chit`, on a video
titled *taking a piece of paper from the desk*. COCO has no chit class, so real
paper surfaces under the book class or not at all. Rendering the survivors
showed white shirts, partitions and desk edges at confidence 0.518 down to
0.370, against 0.933 for the real phone. **Paper/chit detection does not work
on this corpus.**

---

## Not measured

- **Precision and recall** — there is no reference manifest, so every count is
  what the system reported, not what was true.
- **Eye gaze / iris direction** — impossible at 4.7–10.1 px interocular.
- **Face identity** — never computed, by design.
- **Seat attribution** — 6 of 7 calibrations are DRAFT, so all rows are
  `unattributed`; the pipeline cannot distinguish a seated candidate from a
  person walking the aisle, and seat-change detection is impossible.
