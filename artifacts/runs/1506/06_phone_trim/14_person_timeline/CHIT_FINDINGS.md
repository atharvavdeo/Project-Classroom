# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1506\06_phone_trim`.

**1481 raw detections -> 987 kept** after gating on confidence >= 0.4, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 213
- larger than a chit: 197
- no wrist resolved in that frame: 84
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 1 | sustained_paper_handling | 126.2s | 51.0s | 12 | 0.70 |
| 2 | 66 | sustained_paper_handling | 62.8s | 20.5s | 11 | 0.71 |
| 3 | 28 | repeated_brief_paper_handling | 16.8s | 4.5s | 11 | 0.77 |
| 4 | 6 | repeated_brief_paper_handling | 16.5s | 4.2s | 10 | 0.60 |
| 5 | 42 | repeated_brief_paper_handling | 8.8s | 2.5s | 11 | 0.60 |
| 6 | 124 | repeated_brief_paper_handling | 5.8s | 2.2s | 7 | 0.71 |
| 7 | 13 | repeated_brief_paper_handling | 4.8s | 1.5s | 6 | 0.65 |
| 8 | 112 | repeated_brief_paper_handling | 4.2s | 2.8s | 5 | 0.63 |
| 9 | 125 | repeated_brief_paper_handling | 4.0s | 1.8s | 6 | 0.50 |
| 10 | 3 | repeated_brief_paper_handling | 3.2s | 2.8s | 3 | 0.67 |
| 11 | 18 | repeated_brief_paper_handling | 2.8s | 1.2s | 4 | 0.75 |
| 12 | 38 | brief_paper_handling | 2.8s | 2.8s | 1 | 0.57 |
| 13 | 108 | brief_paper_handling | 2.2s | 2.2s | 1 | 0.53 |
| 14 | 16 | brief_paper_handling | 2.0s | 1.5s | 2 | 0.58 |
| 15 | 43 | brief_paper_handling | 2.0s | 1.2s | 2 | 0.50 |
| 16 | 78 | brief_paper_handling | 1.5s | 1.0s | 2 | 0.51 |
| 17 | 133 | insufficient_observation | 1.5s | 1.5s | 1 | 0.43 |
| 18 | 46 | brief_paper_handling | 1.2s | 1.2s | 1 | 0.62 |
| 19 | 49 | insufficient_observation | 1.2s | 1.2s | 1 | 0.54 |
| 20 | 24 | repeated_brief_paper_handling | 1.2s | 0.8s | 3 | 0.50 |

## Why each of the top five

**1. Track 1** (677 sightings, 677 with a wrist)
- 463 of 463 detections survived, at the hands and small enough
- handling in 68% of sightings where a wrist resolved
- longest continuous handling 51.0s across 12 episode(s)
- D-FINE also reports notebook/paper in 23% of sightings -- there is real paperwork at this seat, which may be legitimate

**2. Track 66** (472 sightings, 470 with a wrist)
- 227 of 227 detections survived, at the hands and small enough
- handling in 48% of sightings where a wrist resolved
- longest continuous handling 20.5s across 11 episode(s)

**3. Track 28** (388 sightings, 334 with a wrist)
- 62 of 198 detections survived, at the hands and small enough
- handling in 19% of sightings where a wrist resolved
- longest continuous handling 4.5s across 11 episode(s)
- D-FINE also reports notebook/paper in 86% of sightings -- there is real paperwork at this seat, which may be legitimate

**4. Track 6** (175 sightings, 161 with a wrist)
- 46 of 108 detections survived, at the hands and small enough
- handling in 29% of sightings where a wrist resolved
- longest continuous handling 4.2s across 10 episode(s)
- D-FINE also reports notebook/paper in 191% of sightings -- there is real paperwork at this seat, which may be legitimate

**5. Track 42** (244 sightings, 201 with a wrist)
- 29 of 112 detections survived, at the hands and small enough
- handling in 14% of sightings where a wrist resolved
- longest continuous handling 2.5s across 11 episode(s)
- D-FINE also reports notebook/paper in 135% of sightings -- there is real paperwork at this seat, which may be legitimate

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new