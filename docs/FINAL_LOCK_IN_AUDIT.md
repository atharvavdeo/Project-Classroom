# Final lock-in audit

*Audited 2026-08-22 against upstream `40deeff`, checked-in run artifacts,
source code, rendered-evidence notes and Product Zero documents. This is an
evidence audit, not an accuracy report: these runs have no human reference
manifest.*

## Release decision

**Lock the system as a human-review evidence pipeline, not as a cheating
classifier.** It may produce observations and, after the policy below is
implemented, a review queue. It must not automatically call a person a cheat,
clear a person, or use an undetected/occluded region as adverse evidence.

`ARCHITECTURE.md` is a historical 1446 document. Its claims that the COCO phone
route works and chit detection is wholly out of scope are not the current
operating truth. Current code has a hosted paper prototype, but it does not make
a chit decision. `CLASSIFICATION_AND_OPEN_QUESTIONS.md` is the target policy;
its incident ladder is not implemented.

## What is done and can remain locked

| Area | Capability | Boundary |
|---|---|---|
| Video handling | PTS-aware ingest, health checks, sealed artifacts and continuous 4 Hz sampling exist. Run 1501 corrects the earlier motion-only sample gap. | The longest relevant run is 143 seconds, not a three-hour exam. |
| Person/pose measurement | AlphaPose produces skeletons, wrists and coarse head-orientation measurements, including a not-resolvable state. | This is not eye gaze, screen reading or an inference of intent. |
| Evidence provenance | Samples carry source boxes, timestamps, joints and object geometry, so claimed evidence can be rendered and reviewed. | A box/score is an observation, not a violation. |
| Paper prototype | `chit-paper-new/2` is mapped to `paper_like_object`; retained and rejected detections are auditable. | It cannot distinguish permitted answer paper from a cheat note, and it is hosted rather than offline. |
| Human oversight | Documentation reserves final confirmation for a person. | This must be enforced in stored review state and UI, not merely prose. |

## What the run evidence says

- Run 1501 inspected the whole 12-paper and phone clips. Only 43 of 236 tracks
  in `12_paper` and 15 of 84 in `01_phone` had a usable yaw baseline; tracking
  fragmentation makes the rest unsuitable for person-level claims.
- The visually verified phone in `01_phone` scored 0.933, but stock COCO also
  labelled wall seat placards as phones at 0.30–0.49. Current phone flags are
  therefore **review cues only**.
- On `12_paper`, hand/size gates moved a subject handling a small pale object to
  the top of the paper ranking. The same model also fires on answer sheets, so
  the observation cannot establish an illicit chit.
- Run 1504 is the counterexample to a fixed wrist gate: 2,689 raw paper-model
  detections became 1,461 retained detections and ranked four people as handling
  paper for 64–132 seconds. Rendered examples were keyboard/mouse under a hand,
  a hand on a mouse and a water bottle—normal computer-test context.

These are detector reports and rendered observations, not precision, recall,
incident totals or cheating rates.

## Current implementation gaps

| Priority | Finding | Evidence | Resulting risk |
|---:|---|---|---|
| P0 | No implemented `incident_candidate` or reviewer-decision store. | `person_timeline.flag()` emits independent flags; paper scoring writes a separate ranking. | The Confirm/Dismiss/Need-better-view workflow cannot receive output yet. |
| P0 | COCO phone confidence is not gated in the timeline. | `TimelineConfig.min_object_confidence` is `0.0`; known placards score 0.30–0.49. | A low-confidence COCO phone can make a high-severity object flag. |
| P0 | No keyboard/mouse/monitor overlap veto in the paper gate. | `chit_evidence.gate_detection()` checks score, box-size and wrist distance only. | The 1504 keyboard/mouse failure survives the intended protection. |
| P0 | The 12%-of-person-box paper size rule is not camera transferable. | 1504 needed a different effective cap at a different scale; code keeps the fixed fraction. | A global cutoff rejects valid evidence in one view or retains desk context in another. |
| P0 | Track gaps are not departures and tracks are not identities. | The same person split into tracks 118/53; dense footage fragments heavily. | Return, seat-swap, passing and repetition claims are not valid today. |
| P1 | Paper and COCO channels are not joined coherently. | Paper is separately scored; timeline treats COCO book/notebook as a target but not `paper_like_object`. | They cannot corroborate safely, and permitted stationery may still flag. |
| P1 | Repeated glances are a fraction-away-from-baseline, not target-specific episodes. | The timeline uses 25%/40% fractions with no neighbour/screen geometry. | Ordinary movement can look suspicious; meaningful repeats can be missed. |
| P1 | Paper model is not offline. | `pipeline/chit_detector.py` makes HTTP requests to Roboflow. | It conflicts with the offline Product Zero boundary and has service/rate-limit risk. |
| P1 | Tests are not green in this environment. | `pytest -q tests` stops: `ModuleNotFoundError: av`. | Do not claim a release regression pass until pinned dependencies are installed in an isolated workspace. |

## Review and flagging states to implement

Do not label anything `low_cheating` or `false review`: both imply a judgment
the footage has not earned. Store observability, review priority and human
decision separately.

| State | Meaning | Automatic escalation? |
|---|---|---|
| `not_assessable` | Occluded, too small, no wrist/head baseline, quality failure or incomplete view. | No; it is a coverage fact. |
| `context_observation` | Expected/explainable context: mouse use, paper on desk, staff in aisle, persistent desk object or ordinary head movement. | No; retain/search only. |
| `signal` | A time-bounded, auditable departure or object observation that survives quality checks. | No. |
| `review_candidate` | Time-overlapping signals from independent sources, with caveats retained. | To a human queue only. |
| `reviewer_confirmed_policy_violation` | A named reviewer watched the evidence and applied exam policy. | Human only. |
| `reviewer_dismissed` / `need_better_view` | Human outcome with a reason (placard, permitted paper, occlusion, insufficient view). | Human only. |

The review card must show original clip/frame, source boxes, time window,
quality/occlusion state, seat-attribution state, and whether evidence is from
the same physical person or only the same tracker. Never show a cheating score.

## Behaviour decisions

| Behaviour | Honest current treatment | Needed before escalation |
|---|---|---|
| Person gets up / leaves / returns | Track gap or seat-occupancy observation only. Never call a missing track a departure. | Continuous seat occupancy, staff exclusion, then same-seat appearance comparison. |
| New occupant / seat swap | Not measurable. | Seat occupancy plus evaluated re-identification; still a review candidate, not identity proof. |
| Hand invisible for a long time | `not_assessable`, neither positive nor negative. | Visibility/occlusion timeline; only combine independently visible evidence. |
| Paper lies on desk across frames | `context_observation`; persistence is not a chit. | Object motion/hand attachment plus human policy review. |
| Repeated handling | Candidate generation only. | Object association through time and per-camera context. |
| Head turns / looking around | Coarse orientation only; do not infer gaze or copying. | Adequate head scale, personal baseline, stable episodes and calibrated target direction. |
| Own screen or paper | Expected context. | Independent evidence that target is another seat or prohibited material. |
| Talking / mutual looks | Mouth movement is not measurable at this scale. | Mutual orientation/proximity may queue review; never call it talking alone. |
| Passing an object | Not measurable. | Persistent identities plus object and hand-to-hand association. |
| Phone | Candidate only until validated on this exact camera and negatives. | Model evaluation plus human visual confirmation. |

## Guards: contextual evidence, not global hard rails

Keep confidence, size, wrist distance, camera quality, occlusion, persistence
and temporal continuity as separate fields. They should generate candidates or
abstentions, be calibrated by camera/seat context, and require independent
agreement before a review candidate is created.

For paper, add keyboard/mouse/monitor and desk-zone context, object motion
relative to desk, wrist visibility and explicit abstention. Do not simply raise
one confidence threshold: the known monitor-image false positive scored above
several verified paper hits. For head motion, use a person baseline only after
sufficient visible time, then record repeated direction-specific episodes
relative to calibrated seat geometry. Current face resolution cannot support
eye gaze, intent or a claim that someone read a particular screen/paper.

## Proposed Roboflow phone model

Evaluate the Vianca `handphone-dataset-2` as a **contained benchmark**, not a
direct replacement. Public Roboflow search results expose mixed exam-behaviour
labels (phone/handphone/mobile alongside posture and cheating behaviours), so
inspect its label semantics, camera distance, annotation consistency, split and
negative coverage first.

Build a frozen local pack: the verified phone; placards COCO calls phones;
mouse, keyboard, monitor bezel, hands, paper, water bottle, occlusion and
low-resolution examples. Add intended-camera labels before fine-tuning. Compare
the proposed model, current model and abstain route on identical source boxes,
then review errors by class. Promote only if the real-phone/real-negative
trade-off improves in this room; a published/API confidence does not establish
that. Confirm exact version, licence, class map and export rights before use.
[Roboflow handphone dataset](https://universe.roboflow.com/viancas-workspace/handphone-dataset)

## Lock-in sequence

1. Preserve evidence infrastructure; stop automatic language at `signal` or
   `review_candidate`.
2. Implement stored review states and the reviewer queue before more classifiers.
3. Attach quality, occlusion and negative-context outcomes to every event.
4. Build a human reference manifest and phone hard-negative pack; use it to
   evaluate the proposed model.
5. Add keyboard/mouse/monitor context and object association to the paper path;
   re-render the 1504 failures before it can create review candidates.
6. Build seat occupancy/staff exclusion, then evaluate re-identification before
   return, swap, passing or person-level repetition claims.
7. Install pinned requirements and rerun tests in an isolated root workspace.

Until steps 2–5 are complete, the correct product result is **reviewable
observations with abstentions, not automated cheating decisions**.

