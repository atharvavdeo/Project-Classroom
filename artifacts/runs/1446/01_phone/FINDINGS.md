# 01_phone

`01.Candidate was found using a mobile phone in the examination hall..mkv`  
sha256 `b842b03e5784d867…` · run `artifacts/runs/1446/01_phone`

Observations for a human reviewer. Nothing here asserts that misconduct occurred.

## What the pipeline did

11 stages completed: `source`, `01_ingest`, `02_health`, `04_calibration`, `05_motion_roi`, `06_segmentation`, `07_tracking`, `08_pose`, `09_object`, `10_evidence`, `13_reports`.

- `03_alignment` — **skipped_unavailable**: no camera_motion or camera_repositioned condition was observed, so no global transform was estimated. Identity is the named baseline, not a silent default (PRD 6.3.2)
- `11_gemma` — **skipped_unavailable**: no VLM review was performed; every event needs human review
- `12_evaluation` — **blocked**: no reference manifest; evaluation incomplete and no quality claim may be made

**Preprocessing** — four named transforms measured on identical frames, none applied to the evaluation path:

| transform | Δ sharpness | Δ blocking | seconds |
|---|--:|--:|--:|
| `identity_no_preprocessing` | +0.00 | +0.0000 | 2.4 |
| `clahe` | +203.18 | -0.0122 | 3.2 |
| `denoise_nlmeans` | -30.66 | -0.0091 | 42.0 |
| `unsharp_mask` | +878.47 | -0.0090 | 2.8 |

**Camera motion** — static background support 88.7%, 0 epoch break(s).

| configuration | compensated | median residual | confidence |
|---|--:|--:|--:|
| `ecc` | 0/15 | 0.1957 | 0.820 |
| `features_ransac` | 0/15 | 0.1019 | 0.872 |
| `identity` | 0/15 | 4.2228 | 0.192 |

**Health** — `compression_noise` 17.

**Segmentation** — 46 candidate events.

**Pose** — 1076 identical manifest rows, equivalence `verified`, digest `c1b2f98d2bb45bb5…`.

| configuration | returned | coverage | joints visible | rows/s |
|---|--:|--:|--:|--:|
| `rtmpose_baseline` | 955 | 88.8% | 0.925 | 21.65 |
| `rtmo_baseline` | 545 | 50.6% | 0.779 | 29.90 |
| `alphapose_crowd_baseline` | 1076 | 100.0% | 0.863 | 13.01 |

## What we found

**Head and neck** — 1076 observations, 77.2% resolvable, 42 tracked subject(s). `look_down` 0, `yaw_left` 1, `yaw_right` 0.

| kind | subject | start | duration | peak | baseline | frames |
|---|---|--:|--:|--:|--:|--:|
| `yaw_left` | track_12 | 76.3s | 2.53s | +0.822 | -0.294 | 35 |

Pitch resolved on 796/1076 observations (74%); yaw needs both ears or a nose plus one ear, and the far ear is occluded exactly when the head is most turned.

**Objects** — 1076 identical crops, equivalence `verified`, digest `5e2aea52b3c598b6…`.

| configuration | merged | phone | chit | stationery | hard negatives | crops/s |
|---|--:|--:|--:|--:|--:|--:|
| `rf_detr_native` | _resource_unavailable_ | | | | | |
| `dfine_native` | 6529 | **237** | 0 | 135 | 597 | 11.33 |
| `rf_detr_sahi` | _resource_unavailable_ | | | | | |
| `dfine_sahi` | 10339 | **416** | 0 | 378 | 1236 | 2.53 |

`dfine_sahi` reports 232 mouse, 116 keyboard, 510 monitor_or_bezel against 416 phone. In a computer-based-test hall every desk carries a mouse and a dark rectangular bezel, which is the shape a phone detector confuses most. Only target classes may corroborate an event, so these cannot carry anything into the queue — but they are the false-positive source to fix.

**Review queue** — 15 of 46 events reach a reviewer (33%).

- `retained_low_priority` — 31
- `flagged_for_review` — 15

`rejected_not_offered` means a quality gate withheld the event from the queue. It does **not** mean nothing happened; the evidence keeps its score and stays searchable.

| # | event | seat | start | score | disposition |
|--:|---|---|--:|--:|---|
| 1 | `evt_01-P06_desk_work_surface_0043` | 01-P06 | 65.5s | 85.63 | `flagged_for_review` |
| 2 | `evt_01-P04_upper_body_0030` | 01-P04 | 66.6s | 74.93 | `flagged_for_review` |
| 3 | `evt_01-P06_desk_work_surface_0041` | 01-P06 | 55.6s | 67.32 | `flagged_for_review` |
| 4 | `evt_01-P04_upper_body_0032` | 01-P04 | 95.2s | 65.88 | `flagged_for_review` |
| 5 | `evt_01-P01_desk_work_surface_0003` | 01-P01 | 76.4s | 58.22 | `flagged_for_review` |
| 6 | `evt_01-P06_desk_work_surface_0042` | 01-P06 | 60.8s | 56.90 | `flagged_for_review` |
| 7 | `evt_01-P04_upper_body_0029` | 01-P04 | 60.8s | 54.81 | `flagged_for_review` |
| 8 | `evt_01-P06_upper_body_0046` | 01-P06 | 60.3s | 54.79 | `flagged_for_review` |

## Frames

**Head vector on the source frame** — 1 frame(s), drawn at the midpoint of the longest excursions so the pictures match the table above.

- `08_pose/head/overlays/0000077500_head.jpg`

**phone** — 12 source frame(s) with the detector's own box drawn, plus the native crop at true source size.

- contact sheet — `09_object/dfine_sahi/rendered_phone/contact_sheet_phone.jpg`
- `09_object/dfine_sahi/rendered_phone/0000054807_r04_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000054953_r08_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000058620_r07_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000058820_r09_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000059340_r12_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000059673_r05_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000059673_r10_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000069167_r01_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000069193_r02_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000080420_r06_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000080607_r03_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000081273_r11_frame.jpg`

**stationery** — 12 source frame(s) with the detector's own box drawn, plus the native crop at true source size.

- contact sheet — `09_object/dfine_sahi/rendered_stationery/contact_sheet_stationery.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000054807_r08_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000054953_r12_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000054993_r09_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000055140_r10_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000055287_r11_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000067167_r04_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000067193_r05_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000069487_r06_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000069500_r07_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000069820_r02_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000069833_r03_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000092433_r01_frame.jpg`

## Read this before quoting any number above

- `12_evaluation` was blocked: no reference manifest; evaluation incomplete and no quality claim may be made
- **Every row is `unattributed`.** The calibration is draft, so seat attribution is refused and the pipeline cannot tell a seated candidate from a person walking the aisle. Head events below may include invigilators or passers-by.
- No reference manifest exists for this corpus, so there is **no precision or recall anywhere in this report**. Every count is a count of what the system reported, not of what was true.

## Not measured

- **Eye gaze and iris direction** — impossible at this source scale; an eye is 2–3 px and there is no iris to track.
- **Face identity** — never computed, by design.
- **Chit versus notebook** — no detector class and no labelled data.
- **Seat changes, absences, re-identification** — blocked on calibration approval.

---

368 files · 154 images · 169.3 MB