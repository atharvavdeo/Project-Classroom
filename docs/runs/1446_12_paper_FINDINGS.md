# 12_paper

`Seat No. 12 was seen taking a piece of paper from the desk.mkv`  
sha256 `87c8c1ff8fef393e…` · run `artifacts/runs/1446/12_paper`

Observations for a human reviewer. Nothing here asserts that misconduct occurred.

## What the pipeline did

11 stages completed: `source`, `01_ingest`, `02_health`, `04_calibration`, `05_motion_roi`, `06_segmentation`, `07_tracking`, `08_pose`, `09_object`, `10_evidence`, `13_reports`.

- `03_alignment` — **skipped_unavailable**: no camera_motion or camera_repositioned condition was observed, so no global transform was estimated. Identity is the named baseline, not a silent default (PRD 6.3.2)
- `11_gemma` — **skipped_unavailable**: no VLM review was performed; every event needs human review
- `12_evaluation` — **blocked**: no reference manifest; evaluation incomplete and no quality claim may be made

**Preprocessing** — four named transforms measured on identical frames, none applied to the evaluation path:

| transform | Δ sharpness | Δ blocking | seconds |
|---|--:|--:|--:|
| `identity_no_preprocessing` | +0.00 | +0.0000 | 0.8 |
| `clahe` | +439.35 | +0.0457 | 1.0 |
| `denoise_nlmeans` | -59.15 | -0.1185 | 8.1 |
| `unsharp_mask` | +2011.59 | +0.0330 | 0.8 |

**Camera motion** — static background support 78.6%, 0 epoch break(s).

| configuration | compensated | median residual | confidence |
|---|--:|--:|--:|
| `ecc` | 0/15 | 0.8725 | 0.487 |
| `features_ransac` | 0/15 | 0.2881 | 0.731 |
| `identity` | 0/15 | 10.8766 | 0.084 |

**Health** — `compression_noise` 15, `blur` 1.

**Segmentation** — 29 candidate events.

**Pose** — 5881 identical manifest rows, equivalence `verified`, digest `3df16113796accf0…`.

| configuration | returned | coverage | joints visible | rows/s |
|---|--:|--:|--:|--:|
| `rtmpose_baseline` | 4244 | 72.2% | 0.810 | 101.42 |
| `rtmo_baseline` | 1775 | 30.2% | 0.787 | 365.49 |
| `alphapose_crowd_baseline` | 5881 | 100.0% | 0.622 | 28.68 |

## What we found

**Head and neck** — 5881 observations, 36.6% resolvable, 253 tracked subject(s). `look_down` 0, `yaw_left` 2, `yaw_right` 2.

| kind | subject | start | duration | peak | baseline | frames |
|---|---|--:|--:|--:|--:|--:|
| `yaw_right` | track_83 | 30.2s | 3.00s | -1.500 | -0.688 | 41 |
| `yaw_left` | track_14 | 6.8s | 2.52s | +1.500 | +0.000 | 24 |
| `yaw_right` | track_14 | 4.7s | 2.15s | -1.495 | +0.000 | 15 |
| `yaw_left` | track_83 | 28.3s | 1.77s | +1.500 | -0.688 | 16 |

Pitch resolved on 2123/5881 observations (36%); yaw needs both ears or a nose plus one ear, and the far ear is occluded exactly when the head is most turned.

**Objects** — 5881 identical crops, equivalence `verified`, digest `654ca2ddec3aa28a…`.

| configuration | merged | phone | chit | stationery | hard negatives | crops/s |
|---|--:|--:|--:|--:|--:|--:|
| `rf_detr_native` | _resource_unavailable_ | | | | | |
| `dfine_native` | 32203 | **1648** | 0 | 109 | 14262 | 17.24 |
| `rf_detr_sahi` | _resource_unavailable_ | | | | | |
| `dfine_sahi` | 47888 | **2910** | 0 | 350 | 15577 | 3.36 |

`dfine_sahi` reports 1925 mouse, 3438 keyboard, 9864 monitor_or_bezel against 2910 phone. In a computer-based-test hall every desk carries a mouse and a dark rectangular bezel, which is the shape a phone detector confuses most. Only target classes may corroborate an event, so these cannot carry anything into the queue — but they are the false-positive source to fix.

**Review queue** — 7 of 29 events reach a reviewer (24%).

- `retained_low_priority` — 22
- `flagged_for_review` — 7

`rejected_not_offered` means a quality gate withheld the event from the queue. It does **not** mean nothing happened; the evidence keeps its score and stays searchable.

| # | event | seat | start | score | disposition |
|--:|---|---|--:|--:|---|
| 1 | `evt_campaper-P08_desk_work_surface_0018` | campaper-P08 | 30.0s | 595.74 | `flagged_for_review` |
| 2 | `evt_campaper-P09_desk_work_surface_0021` | campaper-P09 | 6.3s | 576.47 | `flagged_for_review` |
| 3 | `evt_campaper-P09_desk_work_surface_0022` | campaper-P09 | 22.1s | 555.88 | `flagged_for_review` |
| 4 | `evt_campaper-P08_upper_body_0020` | campaper-P08 | 30.0s | 206.03 | `flagged_for_review` |
| 5 | `evt_campaper-P09_upper_body_0024` | campaper-P09 | 22.1s | 172.06 | `flagged_for_review` |
| 6 | `evt_campaper-P09_upper_body_0023` | campaper-P09 | 6.3s | 157.12 | `flagged_for_review` |
| 7 | `evt_campaper-P08_desk_work_surface_0017` | campaper-P08 | 0.5s | 123.53 | `flagged_for_review` |
| 8 | `evt_campaper-P08_upper_body_0019` | campaper-P08 | 0.5s | 75.55 | `retained_low_priority` |

## Frames

**Head vector on the source frame** — 4 frame(s), drawn at the midpoint of the longest excursions so the pictures match the table above.

- `08_pose/head/overlays/0000005667_head.jpg`
- `08_pose/head/overlays/0000008047_head.jpg`
- `08_pose/head/overlays/0000029167_head.jpg`
- `08_pose/head/overlays/0000031580_head.jpg`

**phone** — 12 source frame(s) with the detector's own box drawn, plus the native crop at true source size.

- contact sheet — `09_object/dfine_sahi/rendered_phone/contact_sheet_phone.jpg`
- `09_object/dfine_sahi/rendered_phone/0000012380_r05_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000021833_r09_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000027247_r01_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000027247_r02_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000028647_r06_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000028673_r07_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000028980_r03_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000030060_r12_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000032727_r11_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000033060_r04_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000052727_r10_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000053393_r08_frame.jpg`

**stationery** — 12 source frame(s) with the detector's own box drawn, plus the native crop at true source size.

- contact sheet — `09_object/dfine_sahi/rendered_stationery/contact_sheet_stationery.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000005333_r01_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000006713_r07_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000010233_r06_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000010380_r12_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000024353_r08_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000024833_r10_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000026340_r09_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000032167_r05_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000053780_r04_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000054060_r11_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000057980_r02_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000057980_r03_frame.jpg`

**book_or_notebook_like_object** — 12 source frame(s) with the detector's own box drawn, plus the native crop at true source size.

- contact sheet — `09_object/dfine_sahi/rendered_book_or_notebook_like_object/contact_sheet_book_or_notebook_like_object.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000008233_r05_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000008380_r02_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000008380_r03_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000008487_r04_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000008900_r01_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000022247_r09_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000034647_r11_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000034673_r12_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000035913_r10_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000051060_r06_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000054780_r07_frame.jpg`
- `09_object/dfine_sahi/rendered_book_or_notebook_like_object/0000057113_r08_frame.jpg`

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

355 files · 191 images · 281.0 MB