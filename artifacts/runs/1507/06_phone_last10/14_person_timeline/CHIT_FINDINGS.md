# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1507\06_phone_last10`.

**112 raw detections -> 69 kept** after gating on confidence >= 0.4, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 17
- larger than a chit: 26
- no wrist resolved in that frame: 0
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 3 | sustained_paper_handling | 10.0s | 10.0s | 1 | 0.66 |
| 2 | 5 | brief_paper_handling | 4.0s | 4.0s | 1 | 0.66 |
| 3 | 4 | brief_paper_handling | 2.2s | 1.8s | 2 | 0.43 |
| 4 | 11 | insufficient_observation | 1.0s | 1.0s | 1 | 0.59 |
| 5 | 13 | brief_paper_handling | 0.8s | 0.8s | 1 | 0.73 |
| 6 | 7 | brief_paper_handling | 0.8s | 0.5s | 2 | 0.45 |
| 7 | 10 | insufficient_observation | 0.2s | 0.2s | 1 | 0.40 |

## Why each of the top five

**1. Track 3** (40 sightings, 40 with a wrist)
- 40 of 40 detections survived, at the hands and small enough
- handling in 100% of sightings where a wrist resolved
- longest continuous handling 10.0s across 1 episode(s)
- D-FINE also reports notebook/paper in 62% of sightings -- there is real paperwork at this seat, which may be legitimate

**2. Track 5** (40 sightings, 40 with a wrist)
- 16 of 16 detections survived, at the hands and small enough
- handling in 40% of sightings where a wrist resolved
- longest continuous handling 4.0s across 1 episode(s)

**3. Track 4** (28 sightings, 28 with a wrist)
- 5 of 9 detections survived, at the hands and small enough
- handling in 18% of sightings where a wrist resolved
- longest continuous handling 1.8s across 2 episode(s)

**4. Track 11** (5 sightings, 5 with a wrist)
- only 5 sightings

**5. Track 13** (18 sightings, 18 with a wrist)
- 3 of 13 detections survived, at the hands and small enough
- handling in 17% of sightings where a wrist resolved
- longest continuous handling 0.8s across 1 episode(s)

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new