# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1504\04_talking`.

**2689 raw detections -> 1461 kept** after gating on confidence >= 0.4, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 432
- larger than a chit: 683
- no wrist resolved in that frame: 113
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 2 | sustained_paper_handling | 132.0s | 62.0s | 10 | 0.76 |
| 2 | 5 | sustained_paper_handling | 107.8s | 27.2s | 18 | 0.73 |
| 3 | 4 | sustained_paper_handling | 83.2s | 21.8s | 32 | 0.72 |
| 4 | 3 | sustained_paper_handling | 64.5s | 14.8s | 18 | 0.77 |
| 5 | 6 | brief_paper_handling | 1.5s | 1.2s | 2 | 0.51 |
| 6 | 1 | brief_paper_handling | 1.2s | 1.0s | 2 | 0.63 |

## Why each of the top five

**1. Track 2** (573 sightings, 573 with a wrist)
- 510 of 522 detections survived, at the hands and small enough
- handling in 89% of sightings where a wrist resolved
- longest continuous handling 62.0s across 10 episode(s)

**2. Track 5** (573 sightings, 561 with a wrist)
- 396 of 521 detections survived, at the hands and small enough
- handling in 70% of sightings where a wrist resolved
- longest continuous handling 27.2s across 18 episode(s)

**3. Track 4** (573 sightings, 573 with a wrist)
- 307 of 445 detections survived, at the hands and small enough
- handling in 50% of sightings where a wrist resolved
- longest continuous handling 21.8s across 32 episode(s)
- D-FINE also reports notebook/paper in 90% of sightings -- there is real paperwork at this seat, which may be legitimate

**4. Track 3** (573 sightings, 474 with a wrist)
- 241 of 603 detections survived, at the hands and small enough
- handling in 50% of sightings where a wrist resolved
- longest continuous handling 14.8s across 18 episode(s)

**5. Track 6** (323 sightings, 23 with a wrist)
- 4 of 5 detections survived, at the hands and small enough
- handling in 17% of sightings where a wrist resolved
- longest continuous handling 1.2s across 2 episode(s)

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new