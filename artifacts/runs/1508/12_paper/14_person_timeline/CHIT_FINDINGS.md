# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1508\12_paper`.

**1816 raw detections -> 393 kept** after gating on confidence >= 0.4, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 729
- larger than a chit: 620
- no wrist resolved in that frame: 74
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 118 | sustained_paper_handling | 56.5s | 56.5s | 1 | 0.74 |
| 2 | 22 | sustained_paper_handling | 15.2s | 5.0s | 11 | 0.68 |
| 3 | 53 | sustained_paper_handling | 13.0s | 12.0s | 2 | 0.77 |
| 4 | 100 | sustained_paper_handling | 10.5s | 5.8s | 9 | 0.61 |
| 5 | 73 | brief_paper_handling | 3.0s | 3.0s | 1 | 0.56 |
| 6 | 114 | brief_paper_handling | 1.8s | 1.5s | 2 | 0.66 |
| 7 | 18 | brief_paper_handling | 1.5s | 1.2s | 2 | 0.71 |
| 8 | 16 | brief_paper_handling | 1.5s | 1.2s | 2 | 0.63 |
| 9 | 63 | brief_paper_handling | 1.2s | 1.2s | 1 | 0.65 |
| 10 | 43 | insufficient_observation | 1.2s | 1.2s | 1 | 0.46 |
| 11 | 84 | brief_paper_handling | 1.0s | 0.8s | 2 | 0.60 |
| 12 | 39 | insufficient_observation | 1.0s | 1.0s | 1 | 0.60 |
| 13 | 108 | brief_paper_handling | 0.5s | 0.5s | 1 | 0.46 |
| 14 | 45 | insufficient_observation | 0.5s | 0.5s | 1 | 0.45 |
| 15 | 11 | insufficient_observation | 0.2s | 0.2s | 1 | 0.54 |
| 16 | 28 | insufficient_observation | 0.2s | 0.2s | 1 | 0.49 |
| 17 | 112 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.45 |
| 18 | 50 | insufficient_observation | 0.2s | 0.2s | 1 | 0.44 |
| 19 | 48 | insufficient_observation | 0.2s | 0.2s | 1 | 0.42 |
| 20 | 176 | insufficient_observation | 0.2s | 0.2s | 1 | 0.42 |

## Why each of the top five

**1. Track 118** (231 sightings, 230 with a wrist)
- 218 of 303 detections survived, at the hands and small enough
- handling in 94% of sightings where a wrist resolved
- longest continuous handling 56.5s across 1 episode(s)
- D-FINE also reports notebook/paper in 296% of sightings -- there is real paperwork at this seat, which may be legitimate

**2. Track 22** (349 sightings, 345 with a wrist)
- 46 of 233 detections survived, at the hands and small enough
- handling in 13% of sightings where a wrist resolved
- longest continuous handling 5.0s across 11 episode(s)
- D-FINE also reports notebook/paper in 20% of sightings -- there is real paperwork at this seat, which may be legitimate

**3. Track 53** (90 sightings, 90 with a wrist)
- 46 of 101 detections survived, at the hands and small enough
- handling in 51% of sightings where a wrist resolved
- longest continuous handling 12.0s across 2 episode(s)
- D-FINE also reports notebook/paper in 81% of sightings -- there is real paperwork at this seat, which may be legitimate

**4. Track 100** (265 sightings, 255 with a wrist)
- 34 of 219 detections survived, at the hands and small enough
- handling in 13% of sightings where a wrist resolved
- longest continuous handling 5.8s across 9 episode(s)
- D-FINE also reports notebook/paper in 28% of sightings -- there is real paperwork at this seat, which may be legitimate

**5. Track 73** (44 sightings, 44 with a wrist)
- 9 of 30 detections survived, at the hands and small enough
- handling in 20% of sightings where a wrist resolved
- longest continuous handling 3.0s across 1 episode(s)
- D-FINE also reports notebook/paper in 157% of sightings -- there is real paperwork at this seat, which may be legitimate

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new