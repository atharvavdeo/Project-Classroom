# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1509\01_phone`.

**373 raw detections -> 274 kept** after gating on confidence >= 0.4, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 43
- larger than a chit: 54
- no wrist resolved in that frame: 2
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 43 | sustained_paper_handling | 17.8s | 17.8s | 1 | 0.53 |
| 2 | 1 | repeated_brief_paper_handling | 10.2s | 4.0s | 6 | 0.68 |
| 3 | 51 | sustained_paper_handling | 7.5s | 7.5s | 1 | 0.68 |
| 4 | 13 | repeated_brief_paper_handling | 7.0s | 4.8s | 3 | 0.62 |
| 5 | 25 | repeated_brief_paper_handling | 6.2s | 3.0s | 5 | 0.73 |
| 6 | 2 | repeated_brief_paper_handling | 4.2s | 2.0s | 4 | 0.65 |
| 7 | 59 | brief_paper_handling | 4.0s | 3.8s | 2 | 0.64 |
| 8 | 65 | brief_paper_handling | 4.0s | 4.0s | 1 | 0.59 |
| 9 | 18 | brief_paper_handling | 3.8s | 3.8s | 1 | 0.73 |
| 10 | 60 | brief_paper_handling | 3.2s | 3.2s | 1 | 0.65 |
| 11 | 50 | brief_paper_handling | 3.0s | 2.8s | 2 | 0.56 |
| 12 | 35 | repeated_brief_paper_handling | 1.8s | 1.0s | 4 | 0.68 |
| 13 | 80 | insufficient_observation | 1.8s | 1.8s | 1 | 0.54 |
| 14 | 54 | insufficient_observation | 1.5s | 1.5s | 1 | 0.45 |
| 15 | 71 | brief_paper_handling | 1.0s | 0.8s | 2 | 0.62 |
| 16 | 68 | brief_paper_handling | 0.8s | 0.5s | 2 | 0.57 |
| 17 | 61 | insufficient_observation | 0.8s | 0.8s | 1 | 0.56 |
| 18 | 58 | brief_paper_handling | 0.5s | 0.2s | 2 | 0.42 |
| 19 | 66 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.55 |
| 20 | 41 | insufficient_observation | 0.2s | 0.2s | 1 | 0.49 |

## Why each of the top five

**1. Track 43** (138 sightings, 138 with a wrist)
- 68 of 93 detections survived, at the hands and small enough
- handling in 49% of sightings where a wrist resolved
- longest continuous handling 17.8s across 1 episode(s)

**2. Track 1** (245 sightings, 245 with a wrist)
- 41 of 47 detections survived, at the hands and small enough
- handling in 17% of sightings where a wrist resolved
- longest continuous handling 4.0s across 6 episode(s)

**3. Track 51** (28 sightings, 28 with a wrist)
- 28 of 30 detections survived, at the hands and small enough
- handling in 86% of sightings where a wrist resolved
- longest continuous handling 7.5s across 1 episode(s)

**4. Track 13** (47 sightings, 46 with a wrist)
- 24 of 26 detections survived, at the hands and small enough
- handling in 52% of sightings where a wrist resolved
- longest continuous handling 4.8s across 3 episode(s)

**5. Track 25** (56 sightings, 55 with a wrist)
- 22 of 28 detections survived, at the hands and small enough
- handling in 40% of sightings where a wrist resolved
- longest continuous handling 3.0s across 5 episode(s)
- D-FINE also reports notebook/paper in 82% of sightings -- there is real paperwork at this seat, which may be legitimate

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new