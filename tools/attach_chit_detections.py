"""Attach paper-chit detections to an existing person timeline.

    set ROBOFLOW_API_KEY=...
    python tools/attach_chit_detections.py VIDEO --run artifacts/runs/1501/12_paper

Reads `14_person_timeline/samples.jsonl` from a completed run, crops each
person box out of the source video, scores it with `chit-paper-new/2`, and
writes the detections back into that sample's `near_hand_objects` -- the same
field D-FINE's attachments use, so `reprofile.py`, `flag()` and
`render_annotated_video.py` all pick them up with no further change.

### Why it re-uses a finished run rather than re-running the pipeline

Run 1501's `12_paper` already holds 6,109 person samples at 99%+ coverage,
each with a box, a track id, wrists and a full skeleton, produced on GPU. None
of that changes when a new object detector is introduced. Re-running detection,
tracking and pose to obtain identical results would cost an hour and risk
perturbing artifacts that other documents already cite.

The original `samples.jsonl` is left untouched and the enriched copy is written
beside it, so the two can be diffed and the claim "the chit model added these
and only these" can actually be checked.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, chit_detector as cd  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=cd.Config.confidence)
    ap.add_argument("--min-box-px", type=int, default=cd.Config.min_box_px)
    ap.add_argument("--workers", type=int, default=cd.Config.workers)
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="score only the first N sampled frames (a smoke test)")
    ap.add_argument("--out-name", default="samples_with_chits.jsonl")
    args = ap.parse_args()

    import av
    import cv2

    cfg = cd.Config(confidence=args.confidence, min_box_px=args.min_box_px,
                    workers=args.workers)
    client = cd.ChitClient(cfg)

    paths = artifacts.open_run(args.run.parent, args.run.name)
    stage = paths.dir / "14_person_timeline"
    samples = read_jsonl(stage / "samples.jsonl")
    if not samples:
        print("no samples.jsonl in that run", file=sys.stderr)
        return 2

    by_pts: dict[float, list[dict]] = defaultdict(list)
    for sample in samples:
        by_pts[round(float(sample["pts_ms"]), 1)].append(sample)
    stamps = sorted(by_pts)
    if args.limit_frames:
        stamps = stamps[:args.limit_frames]
    print(f"{len(samples)} samples over {len(by_pts)} frames; "
          f"scoring {len(stamps)} frames")

    # One job per person box: the encoded crop plus what is needed to map the
    # detections back into frame coordinates.
    jobs: list[tuple] = []
    skipped_small = 0

    def build(stamp: float, canvas) -> None:
        """Turn every person at `stamp` into a crop job against `canvas`."""
        nonlocal skipped_small
        for sample in by_pts[stamp]:
            rect = cd.crop_geometry(sample["box"], frame_w, frame_h, cfg)
            if rect is None:
                skipped_small += 1
                continue
            cx1, cy1, cx2, cy2 = rect
            crop = canvas[cy1:cy2, cx1:cx2]
            scale = cd.upscale_factor(cx2 - cx1, cy2 - cy1, cfg)
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            ok, buf = cv2.imencode(".jpg", crop,
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                continue
            jobs.append((sample, buf.tobytes(), (cx1, cy1), scale))

    # A sample's `pts_ms` is the *planned* grid time, not a decoded frame's
    # timestamp. `plan_continuous` lays a uniform grid (250, 500, 750 ms) and
    # the pose stage scored whichever frame was nearest. Four of six files in
    # this corpus are VFR, and on this one the real timestamps are 280, 480,
    # 760 ms -- so exact matching hits 88 of 353 stamps and would have scored
    # 1,527 of 6,109 samples while reporting the pass as complete.
    #
    # Each planned stamp is therefore assigned to its nearest decoded frame,
    # using a one-frame lookahead. `matched` below is the guardrail: silently
    # measuring a quarter of a recording is the exact defect this pipeline was
    # already burned by once, and it was invisible because nothing counted.
    pending = sorted(stamps)
    cursor = 0
    previous: tuple | None = None
    matched: list[float] = []

    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        frame_w = stream.codec_context.width
        frame_h = stream.codec_context.height
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts_ms = float(frame.pts * time_base) * 1000.0
            canvas = None
            # Claim every pending stamp that this frame is now the nearest
            # decoded frame for. A stamp is settled once a frame at or past it
            # arrives: the answer is whichever of the two neighbours is closer.
            while cursor < len(pending) and pending[cursor] <= pts_ms:
                stamp = pending[cursor]
                use_previous = (previous is not None and
                                abs(stamp - previous[0]) <= abs(stamp - pts_ms))
                if use_previous:
                    build(stamp, previous[1])
                else:
                    if canvas is None:
                        canvas = frame.to_ndarray(format="bgr24")
                    build(stamp, canvas)
                matched.append(stamp)
                cursor += 1
            if cursor >= len(pending):
                break
            if canvas is None:
                canvas = frame.to_ndarray(format="bgr24")
            previous = (pts_ms, canvas)

    # Any stamp past the last decoded frame falls to that frame.
    while cursor < len(pending) and previous is not None:
        build(pending[cursor], previous[1])
        matched.append(pending[cursor])
        cursor += 1

    covered = sum(len(by_pts[t]) for t in matched)
    print(f"{len(matched)}/{len(stamps)} stamps matched to a decoded frame, "
          f"covering {covered}/{sum(len(by_pts[t]) for t in stamps)} samples")
    if len(matched) < len(stamps):
        print(f"ERROR: {len(stamps) - len(matched)} sampled timestamps could "
              f"not be matched to any frame; refusing to report a partial "
              f"pass as complete", file=sys.stderr)
        return 2

    print(f"{len(jobs)} crops to score "
          f"({skipped_small} boxes under {cfg.min_box_px}px skipped)")
    if not jobs:
        print("nothing to score", file=sys.stderr)
        return 2

    started = time.time()
    done = 0
    attached = 0

    def score(job):
        sample, jpeg, origin, scale = job
        return sample, client.infer(jpeg), origin, scale

    with ThreadPoolExecutor(cfg.workers) as pool:
        for sample, predictions, origin, scale in pool.map(score, jobs):
            done += 1
            for prediction in predictions:
                sample.setdefault("near_hand_objects", []).append({
                    "cls": cd.TAXONOMY,
                    "confidence": round(float(prediction["confidence"]), 4),
                    "box": cd.to_frame_box(prediction, origin, scale),
                    "source": "roboflow_chit_paper_new_v2"})
                attached += 1
            if done % 500 == 0:
                rate = done / max(time.time() - started, 1e-6)
                print(f"  {done}/{len(jobs)} crops  {attached} detections  "
                      f"{rate:.1f} req/s", flush=True)

    elapsed = time.time() - started

    def from_chit_model(sample) -> bool:
        """Did this sample gain a detection from *this* model?

        Run 1501 predates the change that gave attachments a box and a score,
        so its existing D-FINE entries are bare class-name strings. Counting
        without this guard crashes on them -- and, worse, a version that merely
        skipped them silently would still be wrong, because the point of these
        counts is to separate what the chit model contributed from what was
        already there.
        """
        for item in (sample.get("near_hand_objects") or []):
            if isinstance(item, dict) and                     str(item.get("source", "")).startswith("roboflow"):
                return True
        return False

    out_path = stage / args.out_name
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")

    hits = [s for s in samples if from_chit_model(s)]
    people = len({s["track_id"] for s in hits})
    frames_hit = len({s["pts_ms"] for s in hits})

    summary = {
        "model": cfg.model_id,
        "attribution": cd.ATTRIBUTION,
        "confidence_floor": cfg.confidence,
        "min_box_px": cfg.min_box_px,
        "crops_scored": len(jobs),
        "boxes_skipped_too_small": skipped_small,
        "api_calls": client.calls,
        "api_failures": client.failures,
        "detections": attached,
        "samples_with_a_detection": len(hits),
        "distinct_tracks_with_a_detection": people,
        "distinct_frames_with_a_detection": frames_hit,
        "elapsed_s": round(elapsed, 1),
    }
    (stage / "chit_detection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{attached} detections on {summary['samples_with_a_detection']} "
          f"of {len(jobs)} crops, {people} distinct tracks, "
          f"{frames_hit} distinct frames, in {elapsed:.0f}s")
    if client.failures:
        print(f"WARNING: {client.failures} calls failed after retries. Those "
              f"crops were scored as empty, so the counts above are a floor.")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
