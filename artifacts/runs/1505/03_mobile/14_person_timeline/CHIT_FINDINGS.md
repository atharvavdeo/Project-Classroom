# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1505\03_mobile`.

**885 raw detections -> 714 kept** after gating on confidence >= 0.4, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 46
- larger than a chit: 124
- no wrist resolved in that frame: 1
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 1 | sustained_paper_handling | 194.5s | 40.2s | 47 | 0.73 |
| 2 | 8 | repeated_brief_paper_handling | 6.0s | 2.8s | 3 | 0.64 |
| 3 | 16 | insufficient_observation | 1.2s | 1.2s | 1 | 0.55 |
| 4 | 17 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.56 |
| 5 | 137 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.43 |
| 6 | 18 | insufficient_observation | 0.2s | 0.2s | 1 | 0.41 |
| 7 | 139 | insufficient_observation | 0.2s | 0.2s | 1 | 0.41 |

## Why each of the top five

**1. Track 1** (1127 sightings, 1127 with a wrist)
- 690 of 814 detections survived, at the hands and small enough
- handling in 61% of sightings where a wrist resolved
- longest continuous handling 40.2s across 47 episode(s)
- D-FINE also reports notebook/paper in 177% of sightings -- there is real paperwork at this seat, which may be legitimate

**2. Track 8** (61 sightings, 61 with a wrist)
- 18 of 36 detections survived, at the hands and small enough
- handling in 30% of sightings where a wrist resolved
- longest continuous handling 2.8s across 3 episode(s)

**3. Track 16** (4 sightings, 4 with a wrist)
- only 4 sightings

**4. Track 17** (67 sightings, 67 with a wrist)
- 1 of 4 detections survived, at the hands and small enough
- handling in 1% of sightings where a wrist resolved
- longest continuous handling 0.2s across 1 episode(s)

**5. Track 137** (9 sightings, 5 with a wrist)
- 1 of 1 detections survived, at the hands and small enough
- handling in 20% of sightings where a wrist resolved
- longest continuous handling 0.2s across 1 episode(s)

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new