# 04_talking

`04.CCTV Candidate Talking.mkv`  
sha256 `3811d5707ceed209…` · run `artifacts/runs/1446/04_talking`

Observations for a human reviewer. Nothing here asserts that misconduct occurred.

## What the pipeline did

11 stages completed: `source`, `01_ingest`, `02_health`, `04_calibration`, `05_motion_roi`, `06_segmentation`, `07_tracking`, `08_pose`, `09_object`, `10_evidence`, `13_reports`.

- `03_alignment` — **skipped_unavailable**: no camera_motion or camera_repositioned condition was observed, so no global transform was estimated. Identity is the named baseline, not a silent default (PRD 6.3.2)
- `11_gemma` — **skipped_unavailable**: no VLM review was performed; every event needs human review
- `12_evaluation` — **blocked**: no reference manifest; evaluation incomplete and no quality claim may be made

**Preprocessing** — four named transforms measured on identical frames, none applied to the evaluation path:

| transform | Δ sharpness | Δ blocking | seconds |
|---|--:|--:|--:|
| `identity_no_preprocessing` | +0.00 | +0.0000 | 0.9 |
| `clahe` | +424.06 | -0.0158 | 1.4 |
| `denoise_nlmeans` | -102.95 | -0.0181 | 25.7 |
| `unsharp_mask` | +2976.18 | +0.0098 | 1.0 |

**Camera motion** — static background support 65.5%, 0 epoch break(s).

| configuration | compensated | median residual | confidence |
|---|--:|--:|--:|
| `ecc` | 0/15 | 0.4027 | 0.684 |
| `features_ransac` | 0/15 | 0.2272 | 0.765 |
| `identity` | 0/15 | 7.9846 | 0.111 |

**Health** — `compression_noise` 29.

**Segmentation** — 38 candidate events.

**Pose** — 3542 identical manifest rows, equivalence `verified`, digest `fd5679cc30d38d47…`.

| configuration | returned | coverage | joints visible | rows/s |
|---|--:|--:|--:|--:|
| `rtmpose_baseline` | 2882 | 81.4% | 0.859 | 37.24 |
| `rtmo_baseline` | 443 | 12.5% | 0.785 | 78.36 |
| `alphapose_crowd_baseline` | 3542 | 100.0% | 0.723 | 21.23 |

## What we found

**Head and neck** — 3542 observations, 58.3% resolvable, 118 tracked subject(s). `look_down` 1, `yaw_left` 3, `yaw_right` 1.

| kind | subject | start | duration | peak | baseline | frames |
|---|---|--:|--:|--:|--:|--:|
| `yaw_right` | track_85 | 90.8s | 4.12s | -0.990 | -0.217 | 17 |
| `yaw_left` | track_85 | 103.9s | 2.49s | +0.905 | -0.217 | 13 |
| `yaw_left` | track_85 | 98.4s | 2.45s | +1.012 | -0.217 | 16 |
| `yaw_left` | track_12 | 23.8s | 1.63s | -0.090 | -0.529 | 10 |
| `look_down` | track_10 | 9.4s | 1.52s | +0.555 | -0.176 | 9 |

Pitch resolved on 1885/3542 observations (53%); yaw needs both ears or a nose plus one ear, and the far ear is occluded exactly when the head is most turned.

**Objects** — 3542 identical crops, equivalence `verified`, digest `9a30aca9fbbd5aef…`.

| configuration | merged | phone | chit | stationery | hard negatives | crops/s |
|---|--:|--:|--:|--:|--:|--:|
| `rf_detr_native` | _resource_unavailable_ | | | | | |
| `dfine_native` | 24597 | **1571** | 0 | 312 | 11957 | 11.04 |
| `rf_detr_sahi` | _resource_unavailable_ | | | | | |
| `dfine_sahi` | 40157 | **4188** | 0 | 1154 | 18039 | 3.23 |

`dfine_sahi` reports 2778 mouse, 6930 keyboard, 7177 monitor_or_bezel against 4188 phone. In a computer-based-test hall every desk carries a mouse and a dark rectangular bezel, which is the shape a phone detector confuses most. Only target classes may corroborate an event, so these cannot carry anything into the queue — but they are the false-positive source to fix.

**Review queue** — 10 of 38 events reach a reviewer (26%).

- `retained_low_priority` — 28
- `flagged_for_review` — 10

`rejected_not_offered` means a quality gate withheld the event from the queue. It does **not** mean nothing happened; the evidence keeps its score and stays searchable.

| # | event | seat | start | score | disposition |
|--:|---|---|--:|--:|---|
| 1 | `evt_04-P01_upper_body_0004` | 04-P01 | 81.2s | 211.11 | `flagged_for_review` |
| 2 | `evt_04-P06_desk_work_surface_0021` | 04-P06 | 80.0s | 31.37 | `flagged_for_review` |
| 3 | `evt_04-P06_desk_work_surface_0022` | 04-P06 | 90.3s | 16.09 | `flagged_for_review` |
| 4 | `evt_04-P04_upper_body_0014` | 04-P04 | 11.4s | 15.16 | `flagged_for_review` |
| 5 | `evt_04-P07_desk_work_surface_0027` | 04-P07 | 41.2s | 14.43 | `flagged_for_review` |
| 6 | `evt_04-P06_desk_work_surface_0020` | 04-P06 | 12.0s | 13.22 | `flagged_for_review` |
| 7 | `evt_04-P06_desk_work_surface_0023` | 04-P06 | 130.9s | 12.76 | `flagged_for_review` |
| 8 | `evt_04-P05_upper_body_0018` | 04-P05 | 18.9s | 12.12 | `flagged_for_review` |

## Frames

**Head vector on the source frame** — 5 frame(s), drawn at the midpoint of the longest excursions so the pictures match the table above.

- `08_pose/head/overlays/0000010047_head.jpg`
- `08_pose/head/overlays/0000024447_head.jpg`
- `08_pose/head/overlays/0000092780_head.jpg`
- `08_pose/head/overlays/0000099447_head.jpg`
- `08_pose/head/overlays/0000105113_head.jpg`

**phone** — 12 source frame(s) with the detector's own box drawn, plus the native crop at true source size.

- contact sheet — `09_object/dfine_sahi/rendered_phone/contact_sheet_phone.jpg`
- `09_object/dfine_sahi/rendered_phone/0000002667_r09_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000026593_r06_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000026713_r07_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000032993_r12_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000035500_r04_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000040993_r08_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000043993_r03_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000044327_r02_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000044660_r01_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000050993_r10_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000107567_r05_frame.jpg`
- `09_object/dfine_sahi/rendered_phone/0000128447_r11_frame.jpg`

**stationery** — 12 source frame(s) with the detector's own box drawn, plus the native crop at true source size.

- contact sheet — `09_object/dfine_sahi/rendered_stationery/contact_sheet_stationery.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000070113_r04_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000072780_r09_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000072833_r10_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000091447_r11_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000108447_r12_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000130047_r05_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000130113_r06_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000130447_r03_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000132020_r07_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000132047_r08_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000133020_r01_frame.jpg`
- `09_object/dfine_sahi/rendered_stationery/0000133047_r02_frame.jpg`

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

340 files · 150 images · 265.8 MB