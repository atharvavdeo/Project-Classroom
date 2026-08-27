"""Use SAM 3 to adjudicate the chit detector's surviving detections.

    set ROBOFLOW_API_KEY=...
    python tools/adjudicate_with_sam3.py VIDEO --run artifacts/runs/1504/04_talking

### The problem this solves

Run 1504 ranked four people as `sustained_paper_handling` for 64-132 s of a
143 s recording. Rendered, the surviving detections are **keyboards, mice and a
water bottle**. The wrist-proximity gate cannot remove them because they are
genuinely at the hands -- geometrically those detections are correct.

`chit-paper-new/2` has one class and cannot say "that is a keyboard". SAM 3 is
prompted with free text, so it can be *asked about the confuser directly*.
Measured on the frame that produced the 0.74 keyboard-as-chit:

    prompt "mobile phone, paper chit, keyboard"
    -> 6 x keyboard at 0.52-0.89, zero phones, zero chits

### What this tool does

For each sampled frame that still carries a gated chit detection, it crops the
person, asks SAM 3 about both the target classes and the confusers, and then:

* **suppresses** a chit detection that overlaps a confuser segment,
* **reclassifies** it when SAM 3 calls the same region a phone -- which is the
  more serious finding, not a lesser one,
* **corroborates** it when SAM 3 independently agrees it is paper.

Corroboration matters for the classification ladder in
`docs/CLASSIFICATION_AND_OPEN_QUESTIONS.md`: SAM 3 is a different model with a
different vocabulary, so its agreement is an *independent* indicator in the
sense that document requires, where a second chit detection 250 ms later is not.

### What it does not do

It does not re-run detection over whole frames. SAM 3 costs ~1.4-3.3 s per call
against a serverless endpoint, so adjudicating every crop of a 3-hour recording
is not viable -- and is unnecessary, because only detections that already
survived the geometric gates can change a verdict. This is a referee, not a
detector.

It is also **online**, which conflicts with PS 2's offline requirement. Usable
for development and for building the labelled set; a submission needs local
weights.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import episodes as ep  # noqa: E402
from pipeline import object_detection as od  # noqa: E402
from pipeline import (artifacts, chit_detector as cd,  # noqa: E402
                      chit_evidence as ce, roboflow_workflow as rw)

# The prompt. Naming the confusers is the entire point -- a keyboard the model
# is not asked about is a keyboard it cannot rule out.
TARGETS = ("paper chit", "mobile phone")

# The objects that get mistaken for a chit. `hand` is deliberately NOT here:
# it is asked about, because a hand mask locates the region of interest, but a
# chit held in a hand always overlaps one, so treating it as a confuser
# suppresses every true positive. Measured: with `hand` in this tuple, 56 of 60
# adjudications came back `suppressed_hand`, including the real ones.
CONFUSERS = ("keyboard", "computer mouse", "monitor", "water bottle")
CONTEXT = ("hand",)
PROMPT = ",".join(TARGETS + CONFUSERS + CONTEXT)

# Fraction of the chit box that must lie inside a SAM 3 segment for the two to
# be talking about the same thing. Deliberately modest: SAM 3's masks are tight
# and the chit boxes are loose, so requiring high overlap would let every
# keyboard through.
MIN_OVERLAP = 0.30


def overlap_fraction(box, other) -> float:
    """How strongly a proposal and a SAM 3 segment name the same place.

    The larger of the two coverage fractions, deliberately, because the two
    boxes come from processes with very different tightness and either one can
    be the bigger.

    Dividing only by the proposal's area -- which this did until run 1506 was
    inspected -- silently discarded SAM 3's correct answers. D-FINE's COCO
    boxes are frequently loose: a hand plus a stretch of desk, around 100x70
    px. A real phone segment is about 26x21. The best achievable overlap is
    then 546/7000 = 0.078, under any sane threshold, so SAM 3 finding the phone
    exactly where the proposal pointed was recorded as `unsupported`.

    Measured consequence on run 1506: 1,438 of 1,982 phone proposals came back
    unsupported, including frames showing the same phone that was confirmed at
    85.0 s. The referee was answering; the match test was deaf.

    Taking the max fixes that without loosening suppression: a small proposal
    sitting inside a large monitor segment already matched through the first
    fraction, and still does.
    """
    x1 = max(box[0], other[0])
    y1 = max(box[1], other[1])
    x2 = min(box[2], other[2])
    y2 = min(box[3], other[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    other_area = max((other[2] - other[0]) * (other[3] - other[1]), 1e-6)
    return max(intersection / box_area, intersection / other_area)


def covers(box, other) -> float:
    """Fraction of `box` that `other` actually covers. Asymmetric on purpose.

    Confirmation and suppression are different questions and must not share a
    test.

    To CONFIRM, the question is "did SAM 3 find the claimed object where the
    proposal pointed", and either box may legitimately be the tighter one --
    hence the symmetric `overlap_fraction` above.

    To SUPPRESS, the question is "is this proposal explained away by a piece of
    equipment", and that requires the equipment to actually cover it. A mouse
    segment sitting inside a large paper proposal does not make the proposal a
    mouse.

    Measured: using the symmetric test for suppression on 12_paper collapsed
    corroborations from 136 to 20, drove suppressions from 48 to 314, and threw
    the one man who really took the paper (tracks 118 and 53) out of the review
    queue entirely.
    """
    x1 = max(box[0], other[0])
    y1 = max(box[1], other[1])
    x2 = min(box[2], other[2])
    y2 = min(box[3], other[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    return intersection / max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--samples", default="samples_chit_gated.jsonl",
                    help="which samples file to adjudicate; defaults to the "
                         "gated one, because only detections that already "
                         "survived the geometric gates can change a verdict")
    ap.add_argument("--out-name", default="samples_sam3_adjudicated.jsonl")
    ap.add_argument("--workers", type=int, default=6,
                    help="kept low: a 16-worker burst lost 85%% of calls to "
                         "rate limiting on the plain detector endpoint")
    ap.add_argument("--phone-confidence", type=float, default=0.35,
                    help="floor for D-FINE COCO `phone` proposals before they "
                         "reach the referee. Not a quality claim: COCO phone "
                         "confidence is uncalibrated on this footage (median "
                         "0.31-0.42, peak 0.89 on run 1506). It caps call "
                         "volume; SAM 3 decides the rest.")
    ap.add_argument("--max-per-episode", type=int, default=3,
                    help="referee calls per object episode: the appearance, "
                         "the detector's strongest look, and the end. 0 "
                         "disables episode sampling and calls once per "
                         "proposal row, which is what exhausted a Roboflow "
                         "quota in an afternoon.")
    ap.add_argument("--max-crops", type=int, default=0,
                    help="cap the number of adjudications (a smoke test)")
    ap.add_argument("--save-annotated", type=int, default=8,
                    help="how many of SAM 3's own annotated crops to write out")
    args = ap.parse_args()

    import av
    import cv2

    client = rw.SegmentationWorkflow()
    ccfg = cd.Config()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    stage = paths.dir / "14_person_timeline"
    source = stage / args.samples
    if not source.is_file():
        print(f"{source} not found; run score_chit_evidence.py first",
              file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in
            source.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Two proposal sources reach the referee.
    #
    # The chit detector's `paper_like_object`, as before -- and D-FINE's COCO
    # `phone`, which until now was never adjudicated at all. That omission was
    # measured on run 1506: D-FINE proposed a phone 4,041 times across 59
    # tracks, 709 of them on one man at peak confidence 0.89, on a recording
    # titled "Candidate was found using a mobile phone" -- and SAM 3 was asked
    # about none of them. The output contract bars COCO `phone` from routing
    # anyone to review on its own, and correctly: it fires on mice, keyboards,
    # paper edges and a hand raised to an ear. But barring it while giving it
    # no verified path simply deleted the only phone signal the pipeline has.
    #
    # So D-FINE proposes and never judges, exactly as the contract requires,
    # and SAM 3 decides whether the proposal is a phone, a piece of equipment,
    # or unsupported.
    def phone_proposals(row):
        return [o for o in (row.get("near_hand_objects") or [])
                if isinstance(o, dict)
                and str(o.get("cls", "")) == od.PHONE
                and float(o.get("confidence") or 0.0) >= args.phone_confidence]

    def chit_proposals(row):
        return [o for o in (row.get("near_hand_objects") or [])
                if isinstance(o, dict)
                and str(o.get("source", "")).startswith(
                    ce.CHIT_SOURCE_PREFIX)]

    def chits(row):
        """Every proposal on this row that the referee should see."""
        return chit_proposals(row) + phone_proposals(row)

    targets = [r for r in rows if chits(r)]
    n_chit = sum(len(chit_proposals(r)) for r in rows)
    n_phone = sum(len(phone_proposals(r)) for r in rows)
    print(f"{len(rows)} samples, {len(targets)} carry a proposal "
          f"({n_chit} chit, {n_phone} phone at >={args.phone_confidence})")
    if not targets:
        print("nothing to adjudicate", file=sys.stderr)
        return 2

    by_pts = collections.defaultdict(list)
    for row in targets:
        by_pts[round(float(row["pts_ms"]), 1)].append(row)
    stamps = sorted(by_pts)

    # Same nearest-frame matching as `attach_chit_detections`: sample pts are
    # planned grid times and four of six corpus files are VFR.
    jobs = []
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        frame_w = stream.codec_context.width
        frame_h = stream.codec_context.height
        pending, cursor, previous, matched = list(stamps), 0, None, 0

        def build(stamp, canvas):
            nonlocal matched
            matched += 1
            for row in by_pts[stamp]:
                rect = cd.crop_geometry(row["box"], frame_w, frame_h, ccfg)
                if rect is None:
                    continue
                cx1, cy1, cx2, cy2 = rect
                crop = canvas[cy1:cy2, cx1:cx2]
                scale = cd.upscale_factor(cx2 - cx1, cy2 - cy1, ccfg)
                if scale > 1.0:
                    crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)
                ok, buf = cv2.imencode(".jpg", crop,
                                       [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    jobs.append((row, buf.tobytes(), (cx1, cy1), scale))

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts_ms = float(frame.pts * time_base) * 1000.0
            canvas = None
            while cursor < len(pending) and pending[cursor] <= pts_ms:
                stamp = pending[cursor]
                if previous is not None and \
                        abs(stamp - previous[0]) <= abs(stamp - pts_ms):
                    build(stamp, previous[1])
                else:
                    if canvas is None:
                        canvas = frame.to_ndarray(format="bgr24")
                    build(stamp, canvas)
                cursor += 1
            if cursor >= len(pending):
                break
            if canvas is None:
                canvas = frame.to_ndarray(format="bgr24")
            previous = (pts_ms, canvas)
        while cursor < len(pending) and previous is not None:
            build(pending[cursor], previous[1])
            cursor += 1

    if matched < len(stamps):
        print(f"ERROR: {len(stamps) - matched} timestamps unmatched; refusing "
              f"to report a partial pass", file=sys.stderr)
        return 2
    # --- episode sampling ----------------------------------------------------
    # One call per proposal row asked the referee the same question hundreds of
    # times about one stationary object: run 1506 spent 1,996 calls on 169
    # seconds. Group proposals into episodes and verify a few representative
    # frames of each instead. Unsampled members keep their existing state --
    # unverified proposals, never "confirmed" and never "absent".
    if args.max_per_episode > 0:
        proposals = [(row["pts_ms"], row["track_id"], item)
                     for row, _, _, _ in jobs for item in chits(row)]
        ecfg = ep.Config(max_per_episode=args.max_per_episode)
        episodes, keep, _ = ep.plan(proposals, ecfg)
        before = len(jobs)
        jobs = [j for j in jobs
                if any((j[0]["track_id"], round(float(j[0]["pts_ms"]), 1),
                        tuple(round(float(v), 1) for v in
                              (item.get("box") or [0.0] * 4))) in keep
                       for item in chits(j[0]))]
        print(f"{len(episodes)} episode(s) from {len(proposals)} proposals; "
              f"crops {before} -> {len(jobs)} "
              f"({before / max(len(jobs), 1):.1f}x fewer calls)")

    if args.max_crops:
        jobs = jobs[:args.max_crops]
    print(f"{len(jobs)} crops to adjudicate at {args.workers} workers "
          f"(~{len(jobs) * 2.5 / args.workers / 60:.0f} min)")

    out_dir = stage / "sam3_adjudication"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts = collections.Counter()
    saved = 0
    errors = 0
    started = time.time()

    def adjudicate(job):
        row, jpeg, origin, scale = job
        want = saved < args.save_annotated
        try:
            return row, client.segment(jpeg, PROMPT, keep_annotated=want), \
                origin, scale, None
        except rw.WorkflowError as error:
            return row, None, origin, scale, error

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for row, result, origin, scale, error in pool.map(adjudicate, jobs):
            done += 1
            if error is not None:
                errors += 1
                verdicts["not_adjudicated"] += 1
                continue

            if result.annotated_png is not None and saved < args.save_annotated:
                client.write_annotated(
                    result, out_dir / f"{int(row['pts_ms'])}ms_"
                                      f"trk{row['track_id']}.jpg")
                saved += 1

            # SAM 3 works in crop space; map its boxes back to the frame.
            def to_frame(box):
                ox, oy = origin
                return (ox + box[0] / scale, oy + box[1] / scale,
                        ox + box[2] / scale, oy + box[3] / scale)

            confusers = [(s.cls, to_frame(s.box), s.confidence)
                         for s in result.segments if s.cls.lower() in CONFUSERS]
            papers = [to_frame(s.box) for s in result.segments
                      if s.cls.lower() == "paper chit"]
            phones = [(s.confidence, to_frame(s.box)) for s in result.segments
                      if s.cls.lower() == "mobile phone"]

            for item in chits(row):
                box = item["box"]
                phone_hit = max((c for c, b in phones
                                 if overlap_fraction(box, b) >= MIN_OVERLAP),
                                default=0.0)
                # Strongest overlap wins, not the first in segment order, so
                # a box grazing a monitor does not outrank the keyboard it sits
                # squarely on.
                ranked = sorted(
                    ((covers(box, b), name, conf)
                     for name, b, conf in confusers), reverse=True)
                confuser_hit = (ranked[0][1]
                                if ranked and ranked[0][0] >= MIN_OVERLAP
                                else None)
                confuser_conf = (ranked[0][2]
                                 if ranked and ranked[0][0] >= MIN_OVERLAP
                                 else 0.0)
                paper_hit = any(overlap_fraction(box, b) >= MIN_OVERLAP
                                for b in papers)

                # A D-FINE phone proposal is judged as a phone claim, on its
                # own terms. Without this branch it fell through the chit
                # logic, where SAM 3 answering "paper" scored it `corroborated`
                # -- recording a phone proposal as a confirmed chit.
                #
                # `unsupported` here is the ear/hand false positive being
                # caught: SAM 3 finds no phone where COCO saw one. It still
                # means NOT CONFIRMED, never "there was no phone".
                if str(item.get("cls", "")) == od.PHONE:
                    # When SAM 3 reads the same region as both a phone and a
                    # piece of equipment, the stronger claim wins -- and on a
                    # tie the equipment does.
                    #
                    # Taking the phone unconditionally, as this did until
                    # 04_talking was inspected, reported a black computer mouse
                    # beside a keyboard as a confirmed phone at 0.86. That
                    # video is a CBT lab: 1,202 of its proposals are equipment,
                    # and a dark rectangle there is far more often a mouse than
                    # a handset. The guardrail direction is not to accuse.
                    if phone_hit and phone_hit >= confuser_conf:
                        item["sam3"] = {"verdict": "phone_supported",
                                        "confidence": round(phone_hit, 4)}
                        verdicts["phone_supported"] += 1
                    elif confuser_hit:
                        item["sam3"] = {"verdict": "suppressed",
                                        "as": confuser_hit}
                        verdicts[f"suppressed_{confuser_hit}"] += 1
                    elif paper_hit:
                        item["sam3"] = {"verdict": "phone_was_paper"}
                        verdicts["phone_was_paper"] += 1
                    else:
                        item["sam3"] = {"verdict": "unsupported"}
                        verdicts["unsupported_phone"] += 1
                elif phone_hit:
                    # An escalation, not a rejection: a phone is the more
                    # serious finding and must not be filed as a bad chit.
                    item["sam3"] = {"verdict": "reclassified_as_phone",
                                    "confidence": round(phone_hit, 4)}
                    verdicts["reclassified_as_phone"] += 1
                elif confuser_hit:
                    item["sam3"] = {"verdict": "suppressed",
                                    "as": confuser_hit}
                    verdicts[f"suppressed_{confuser_hit}"] += 1
                elif paper_hit:
                    item["sam3"] = {"verdict": "corroborated"}
                    verdicts["corroborated"] += 1
                else:
                    item["sam3"] = {"verdict": "unsupported"}
                    verdicts["unsupported"] += 1

            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}  "
                      f"{done / max(time.time() - started, 1e-6):.1f}/s  "
                      f"{dict(verdicts)}", flush=True)

    out_path = stage / args.out_name
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {"workflow": rw.ATTRIBUTION, "prompt": PROMPT,
               "min_overlap": MIN_OVERLAP, "crops_adjudicated": len(jobs),
               "api_calls": client.calls, "api_errors": errors,
               "verdicts": dict(verdicts),
               "elapsed_s": round(time.time() - started, 1)}
    (stage / "sam3_adjudication_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nadjudicated {len(jobs)} crops in {time.time() - started:.0f}s")
    for name, count in verdicts.most_common():
        print(f"  {name:28s} {count}")
    if errors:
        print(f"WARNING: {errors} calls failed; those detections are marked "
              f"`not_adjudicated` and were left unchanged, not suppressed.")
    print(f"-> {out_path}")
    print(f"-> {out_dir} ({saved} annotated crops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
