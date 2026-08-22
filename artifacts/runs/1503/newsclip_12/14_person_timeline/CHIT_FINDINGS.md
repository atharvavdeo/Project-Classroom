# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1503\newsclip_12`.

**326 raw detections -> 51 kept** after gating on confidence >= 0.4, size <= 0.2 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 88
- larger than a chit: 110
- no wrist resolved in that frame: 77
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 31 | repeated_brief_paper_handling | 2.8s | 1.0s | 4 | 0.70 |
| 2 | 1 | repeated_brief_paper_handling | 2.0s | 1.5s | 3 | 0.58 |
| 3 | 57 | brief_paper_handling | 1.8s | 1.8s | 1 | 0.56 |
| 4 | 283 | brief_paper_handling | 1.5s | 1.0s | 2 | 0.69 |
| 5 | 298 | brief_paper_handling | 1.2s | 1.2s | 1 | 0.65 |
| 6 | 333 | brief_paper_handling | 0.8s | 0.8s | 1 | 0.62 |
| 7 | 7 | brief_paper_handling | 0.8s | 0.8s | 1 | 0.50 |
| 8 | 142 | brief_paper_handling | 0.5s | 0.5s | 1 | 0.63 |
| 9 | 371 | insufficient_observation | 0.5s | 0.5s | 1 | 0.49 |
| 10 | 248 | brief_paper_handling | 0.5s | 0.2s | 2 | 0.47 |
| 11 | 21 | insufficient_observation | 0.2s | 0.2s | 1 | 0.57 |
| 12 | 373 | insufficient_observation | 0.2s | 0.2s | 1 | 0.54 |
| 13 | 405 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.54 |
| 14 | 2 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.54 |
| 15 | 69 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.51 |
| 16 | 128 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.42 |
| 17 | 277 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.40 |

## Why each of the top five

**1. Track 31** (69 sightings, 58 with a wrist)
- 11 of 21 detections survived, at the hands and small enough
- handling in 19% of sightings where a wrist resolved
- longest continuous handling 1.0s across 4 episode(s)

**2. Track 1** (43 sightings, 41 with a wrist)
- 6 of 17 detections survived, at the hands and small enough
- handling in 15% of sightings where a wrist resolved
- longest continuous handling 1.5s across 3 episode(s)
- D-FINE also reports notebook/paper in 53% of sightings -- there is real paperwork at this seat, which may be legitimate

**3. Track 57** (12 sightings, 11 with a wrist)
- 6 of 6 detections survived, at the hands and small enough
- handling in 45% of sightings where a wrist resolved
- longest continuous handling 1.8s across 1 episode(s)
- D-FINE also reports notebook/paper in 192% of sightings -- there is real paperwork at this seat, which may be legitimate

**4. Track 283** (67 sightings, 64 with a wrist)
- 6 of 48 detections survived, at the hands and small enough
- handling in 9% of sightings where a wrist resolved
- longest continuous handling 1.0s across 2 episode(s)
- D-FINE also reports notebook/paper in 40% of sightings -- there is real paperwork at this seat, which may be legitimate

**5. Track 298** (40 sightings, 32 with a wrist)
- 6 of 13 detections survived, at the hands and small enough
- handling in 16% of sightings where a wrist resolved
- longest continuous handling 1.2s across 1 episode(s)

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new