# Paper-handling findings

Source: `samples_with_chits.jsonl` in `artifacts\runs\1509\04_talking`.

**2689 raw detections -> 1085 kept** after gating on confidence >= 0.5, size <= 0.12 of the person box, and within 0.35 person-widths of a resolved wrist.

Rejections, across all tracks:

- away from the hands: 340
- larger than a chit: 603
- no wrist resolved in that frame: 96
- below the confidence floor: 565

## Ranked for review

| rank | track | disposition | handling | longest | episodes | peak |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 2 | sustained_paper_handling | 122.8s | 51.0s | 14 | 0.76 |
| 2 | 5 | sustained_paper_handling | 70.0s | 11.0s | 31 | 0.73 |
| 3 | 4 | sustained_paper_handling | 52.0s | 5.0s | 39 | 0.72 |
| 4 | 3 | sustained_paper_handling | 51.2s | 9.8s | 21 | 0.77 |
| 5 | 1 | brief_paper_handling | 1.0s | 1.0s | 1 | 0.63 |
| 6 | 6 | brief_paper_handling | 0.2s | 0.2s | 1 | 0.51 |

## Why each of the top five

**1. Track 2** (573 sightings, 573 with a wrist)
- 470 of 522 detections survived, at the hands and small enough
- handling in 82% of sightings where a wrist resolved
- longest continuous handling 51.0s across 14 episode(s)

**2. Track 5** (573 sightings, 561 with a wrist)
- 249 of 521 detections survived, at the hands and small enough
- handling in 44% of sightings where a wrist resolved
- longest continuous handling 11.0s across 31 episode(s)

**3. Track 4** (573 sightings, 573 with a wrist)
- 179 of 445 detections survived, at the hands and small enough
- handling in 31% of sightings where a wrist resolved
- longest continuous handling 5.0s across 39 episode(s)
- D-FINE also reports notebook/paper in 90% of sightings -- there is real paperwork at this seat, which may be legitimate

**4. Track 3** (573 sightings, 474 with a wrist)
- 184 of 603 detections survived, at the hands and small enough
- handling in 39% of sightings where a wrist resolved
- longest continuous handling 9.8s across 21 episode(s)

**5. Track 1** (573 sightings, 555 with a wrist)
- 2 of 577 detections survived, at the hands and small enough
- handling in 0% of sightings where a wrist resolved
- longest continuous handling 1.0s across 1 episode(s)

## What this does not establish

The detector cannot tell a chit from a legitimate answer sheet -- both are small, pale and held. These rankings say *who was handling a small pale object, and for how long*. Whether that object belongs on the desk is a reviewer's decision.

There is no reference manifest for this recording, so none of the above is a precision or recall figure.

Detector: `chit-paper-new/2`. chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/chit-paper-new