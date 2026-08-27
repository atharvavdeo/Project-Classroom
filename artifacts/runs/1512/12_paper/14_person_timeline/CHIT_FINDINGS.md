# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1512\12_paper`.

**1019 raw detections -> 327 kept** after gating on confidence >= 0.5, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 430
- larger than a chit: 262
- no wrist resolved in that frame: 0
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 118 | sustained_paper_handling | 56.5s | 56.5s | 1 | 0.74 |
| 2 | 53 | sustained_paper_handling | 11.8s | 5.5s | 3 | 0.77 |
| 3 | 22 | repeated_brief_paper_handling | 10.2s | 3.5s | 9 | 0.68 |
| 4 | 100 | repeated_brief_paper_handling | 6.0s | 3.0s | 5 | 0.61 |
| 5 | 114 | brief_paper_handling | 1.5s | 1.2s | 2 | 0.66 |
| 6 | 63 | brief_paper_handling | 1.2s | 1.2s | 1 | 0.65 |
| 7 | 18 | brief_paper_handling | 1.0s | 0.8s | 2 | 0.71 |
| 8 | 16 | brief_paper_handling | 1.0s | 0.8s | 2 | 0.63 |
| 9 | 39 | insufficient_observation | 1.0s | 1.0s | 1 | 0.60 |
| 10 | 73 | repeated_brief_paper_handling | 1.0s | 0.5s | 3 | 0.56 |
| 11 | 84 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.60 |
| 12 | 11 | insufficient_observation | 0.2s | 0.2s | 1 | 0.54 |

## Why each of the top five

**1. Track 118** (231 sightings, 230 with a wrist)
- 213 of 242 detections survived, at the hands and small enough
- handling in 93% of sightings where a wrist resolved
- longest continuous handling 56.5s across 1 episode(s)
- D-FINE also reports notebook/paper in 295% of sightings -- there is real paperwork at this seat, which may be legitimate

**2. Track 53** (90 sightings, 90 with a wrist)
- 41 of 77 detections survived, at the hands and small enough
- handling in 46% of sightings where a wrist resolved
- longest continuous handling 5.5s across 3 episode(s)
- D-FINE also reports notebook/paper in 80% of sightings -- there is real paperwork at this seat, which may be legitimate

**3. Track 22** (349 sightings, 345 with a wrist)
- 32 of 125 detections survived, at the hands and small enough
- handling in 9% of sightings where a wrist resolved
- longest continuous handling 3.5s across 9 episode(s)
- D-FINE also reports notebook/paper in 20% of sightings -- there is real paperwork at this seat, which may be legitimate

**4. Track 100** (265 sightings, 255 with a wrist)
- 18 of 119 detections survived, at the hands and small enough
- handling in 7% of sightings where a wrist resolved
- longest continuous handling 3.0s across 5 episode(s)
- D-FINE also reports notebook/paper in 29% of sightings -- there is real paperwork at this seat, which may be legitimate

**5. Track 114** (11 sightings, 11 with a wrist)
- 6 of 9 detections survived, at the hands and small enough
- handling in 55% of sightings where a wrist resolved
- longest continuous handling 1.2s across 2 episode(s)
- D-FINE also reports notebook/paper in 100% of sightings -- there is real paperwork at this seat, which may be legitimate

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new