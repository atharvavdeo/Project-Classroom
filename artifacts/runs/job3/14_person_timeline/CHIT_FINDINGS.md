# Paper-handling findings

Source: `samples_with_chits.jsonl` in `C:\DrishtiAI\Project-Classroom\artifacts\runs\job3`.

**75 raw detections -> 41 kept** after gating on confidence >= 0.5, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 23
- larger than a chit: 11
- no wrist resolved in that frame: 0
- below the confidence floor: 0

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 1 | sustained_paper_handling | 8.0s | 8.0s | 1 | 0.73 |
| 2 | 4 | sustained_paper_handling | 7.0s | 7.0s | 1 | 0.74 |
| 3 | 3 | brief_paper_handling | 5.5s | 3.5s | 2 | 0.70 |
| 4 | 5 | brief_paper_handling | 2.5s | 2.5s | 1 | 0.76 |

## Why each of the top five

**1. Track 1** (16 sightings, 16 with a wrist)
- 16 of 16 detections survived, at the hands and small enough
- handling in 100% of sightings where a wrist resolved
- longest continuous handling 8.0s across 1 episode(s)

**2. Track 4** (16 sightings, 16 with a wrist)
- 13 of 26 detections survived, at the hands and small enough
- handling in 81% of sightings where a wrist resolved
- longest continuous handling 7.0s across 1 episode(s)

**3. Track 3** (16 sightings, 16 with a wrist)
- 8 of 10 detections survived, at the hands and small enough
- handling in 50% of sightings where a wrist resolved
- longest continuous handling 3.5s across 2 episode(s)
- D-FINE also reports notebook/paper in 219% of sightings -- there is real paperwork at this seat, which may be legitimate

**4. Track 5** (16 sightings, 14 with a wrist)
- 4 of 12 detections survived, at the hands and small enough
- handling in 29% of sightings where a wrist resolved
- longest continuous handling 2.5s across 1 episode(s)

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new