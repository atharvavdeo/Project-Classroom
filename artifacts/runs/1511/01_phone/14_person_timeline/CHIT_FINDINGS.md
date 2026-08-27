# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1511\01_phone`.

**192 raw detections -> 141 kept** after gating on confidence >= 0.5, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 26
- larger than a chit: 25
- no wrist resolved in that frame: 0
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 1 | repeated_brief_paper_handling | 9.2s | 3.2s | 5 | 0.69 |
| 2 | 13 | brief_paper_handling | 5.0s | 4.5s | 2 | 0.62 |
| 3 | 51 | repeated_brief_paper_handling | 4.2s | 2.8s | 4 | 0.68 |
| 4 | 25 | repeated_brief_paper_handling | 3.8s | 2.0s | 5 | 0.73 |
| 5 | 65 | brief_paper_handling | 3.8s | 3.8s | 1 | 0.59 |
| 6 | 2 | brief_paper_handling | 2.8s | 1.5s | 2 | 0.65 |
| 7 | 43 | repeated_brief_paper_handling | 2.2s | 1.5s | 4 | 0.53 |
| 8 | 18 | brief_paper_handling | 2.0s | 1.8s | 2 | 0.73 |
| 9 | 60 | brief_paper_handling | 2.0s | 1.8s | 2 | 0.65 |
| 10 | 59 | brief_paper_handling | 2.0s | 1.8s | 2 | 0.64 |
| 11 | 35 | repeated_brief_paper_handling | 1.5s | 1.0s | 3 | 0.68 |
| 12 | 50 | brief_paper_handling | 1.2s | 1.0s | 2 | 0.56 |
| 13 | 80 | insufficient_observation | 1.0s | 1.0s | 1 | 0.54 |
| 14 | 71 | brief_paper_handling | 0.8s | 0.8s | 1 | 0.62 |
| 15 | 61 | insufficient_observation | 0.8s | 0.8s | 1 | 0.56 |
| 16 | 68 | brief_paper_handling | 0.5s | 0.5s | 1 | 0.57 |
| 17 | 66 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.55 |

## Why each of the top five

**1. Track 1** (245 sightings, 245 with a wrist)
- 35 of 37 detections survived, at the hands and small enough
- handling in 14% of sightings where a wrist resolved
- longest continuous handling 3.2s across 5 episode(s)

**2. Track 13** (47 sightings, 46 with a wrist)
- 20 of 20 detections survived, at the hands and small enough
- handling in 43% of sightings where a wrist resolved
- longest continuous handling 4.5s across 2 episode(s)

**3. Track 51** (28 sightings, 28 with a wrist)
- 13 of 15 detections survived, at the hands and small enough
- handling in 46% of sightings where a wrist resolved
- longest continuous handling 2.8s across 4 episode(s)

**4. Track 25** (56 sightings, 55 with a wrist)
- 13 of 18 detections survived, at the hands and small enough
- handling in 24% of sightings where a wrist resolved
- longest continuous handling 2.0s across 5 episode(s)
- D-FINE also reports notebook/paper in 82% of sightings -- there is real paperwork at this seat, which may be legitimate

**5. Track 65** (17 sightings, 17 with a wrist)
- 9 of 9 detections survived, at the hands and small enough
- handling in 53% of sightings where a wrist resolved
- longest continuous handling 3.8s across 1 episode(s)
- D-FINE also reports notebook/paper in 106% of sightings -- there is real paperwork at this seat, which may be legitimate

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new