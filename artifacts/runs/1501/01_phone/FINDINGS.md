# 01_phone

`01.Candidate was found using a mobile phone in the examination hall..mkv`  
sha256 `b842b03e5784d867…` · run `C:/DrishtiAI/Project-Classroom/artifacts/runs/1501/01_phone`

Observations for a human reviewer. Nothing here asserts that misconduct occurred.

## What the pipeline did

7 stages completed: `source`, `01_ingest`, `02_health`, `04_calibration`, `05_motion_roi`, `06_segmentation`, `14_person_timeline`.

- `03_alignment` — **skipped_unavailable**: no camera_motion or camera_repositioned condition was observed, so no global transform was estimated. Identity is the named baseline, not a silent default (PRD 6.3.2)
- `09_object` — **blocked**: no approved taxonomy; PRD 3 blocks detector selection
- `12_evaluation` — **blocked**: no reference manifest; object evaluation is incomplete and no quality claim may be made

**Preprocessing** — four named transforms measured on identical frames, none applied to the evaluation path:

| transform | Δ sharpness | Δ blocking | seconds |
|---|--:|--:|--:|
| `identity_no_preprocessing` | +0.00 | +0.0000 | 0.6 |
| `clahe` | +203.18 | -0.0122 | 0.7 |
| `denoise_nlmeans` | -30.66 | -0.0091 | 6.4 |
| `unsharp_mask` | +878.47 | -0.0090 | 0.6 |

**Camera motion** — static background support 88.7%, 0 epoch break(s).

| configuration | compensated | median residual | confidence |
|---|--:|--:|--:|
| `ecc` | 0/15 | 0.1957 | 0.820 |
| `features_ransac` | 0/15 | 0.1019 | 0.872 |
| `identity` | 0/15 | 4.2228 | 0.192 |

**Health** — `compression_noise` 17.

**Segmentation** — 46 candidate events.

## What we found

## Frames

No rendered frames in this run. Head overlays are written only where a sustained excursion formed; detection frames only where that class was detected.

## Read this before quoting any number above

- `09_object` was blocked: no approved taxonomy; PRD 3 blocks detector selection
- `12_evaluation` was blocked: no reference manifest; object evaluation is incomplete and no quality claim may be made
- No reference manifest exists for this corpus, so there is **no precision or recall anywhere in this report**. Every count is a count of what the system reported, not of what was true.

## Not measured

- **Eye gaze and iris direction** — impossible at this source scale; an eye is 2–3 px and there is no iris to track.
- **Face identity** — never computed, by design.
- **Chit versus notebook** — no detector class and no labelled data.
- **Seat changes, absences, re-identification** — blocked on calibration approval.

---

143 files · 117 images · 97.7 MB