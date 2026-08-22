"""Gate, rank and report the chit detections attached to a run.

    python tools/score_chit_evidence.py --run artifacts/runs/1501/12_paper

Reads `samples_with_chits.jsonl`, applies `chit_evidence`'s per-detection
gates, aggregates per track, and writes:

    14_person_timeline/chit_evidence.json      every track, gated counts
    14_person_timeline/CHIT_FINDINGS.md        ranked, with the numbers shown

Also writes `samples_chit_gated.jsonl`, in which every rejected detection is
removed, so the annotated video can be re-rendered showing only what survived
review rather than everything the detector said.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, chit_evidence as ce  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--samples", default="samples_with_chits.jsonl")
    ap.add_argument("--min-confidence", type=float,
                    default=ce.EvidenceConfig.min_confidence)
    ap.add_argument("--max-size-ratio", type=float,
                    default=ce.EvidenceConfig.max_size_ratio)
    ap.add_argument("--max-wrist-distance", type=float,
                    default=ce.EvidenceConfig.max_wrist_distance)
    args = ap.parse_args()

    cfg = ce.EvidenceConfig(min_confidence=args.min_confidence,
                            max_size_ratio=args.max_size_ratio,
                            max_wrist_distance=args.max_wrist_distance)

    paths = artifacts.open_run(args.run.parent, args.run.name)
    stage = paths.dir / "14_person_timeline"
    source = stage / args.samples
    if not source.is_file():
        print(f"{source} not found; run attach_chit_detections.py first",
              file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in
            source.read_text(encoding="utf-8").splitlines() if line.strip()]

    by_track = collections.defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row)

    evidence = []
    for track_rows in by_track.values():
        item = ce.evaluate(track_rows, cfg)
        ce.classify(item, cfg)
        evidence.append(item)

    ranked = ce.rank(evidence)

    # A copy of the samples carrying only the detections that survived, for
    # rendering. Rejections are dropped rather than flagged so the video shows
    # the reviewed evidence, not the detector's raw opinion.
    kept_total = 0
    for row in rows:
        keep = []
        for objects in (row.get("near_hand_objects") or []):
            if isinstance(objects, str) or not str(
                    objects.get("source", "")).startswith(
                        ce.CHIT_SOURCE_PREFIX):
                keep.append(objects)
                continue
            ok, _reason, _size, _dist = ce.gate_detection(row, objects, cfg)
            if ok:
                keep.append(objects)
                kept_total += 1
        row["near_hand_objects"] = keep
    gated_path = stage / "samples_chit_gated.jsonl"
    with gated_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    (stage / "chit_evidence.json").write_text(
        json.dumps([asdict(e) for e in evidence], indent=2), encoding="utf-8")

    raw = sum(e.raw_detections for e in evidence)
    lines = [
        "# Paper-handling findings",
        "",
        f"Source: `{args.samples}` in `{args.run}`.",
        "",
        f"**{raw} raw detections -> {kept_total} kept** after gating on "
        f"confidence >= {cfg.min_confidence}, size <= "
        f"{cfg.max_size_ratio} of the person box, and within "
        f"{cfg.max_wrist_distance} person-widths of a resolved wrist.",
        "",
        "Rejections, across all tracks:",
        "",
        f"- away from the hands: {sum(e.rejected_far_from_hands for e in evidence)}",
        f"- larger than a chit: {sum(e.rejected_too_large for e in evidence)}",
        f"- no wrist resolved in that frame: "
        f"{sum(e.rejected_no_wrist for e in evidence)}",
        f"- below the confidence floor: "
        f"{sum(e.rejected_low_confidence for e in evidence)}",
        "",
        "## Ranked for review",
        "",
        "| rank | track | disposition | handling | longest | episodes | peak |",
        "|--:|--:|---|--:|--:|--:|--:|",
    ]
    for index, item in enumerate(ranked[:20], 1):
        lines.append(
            f"| {index} | {item.track_id} | {item.disposition} | "
            f"{item.total_handling_ms / 1000:.1f}s | "
            f"{item.longest_episode_ms / 1000:.1f}s | "
            f"{len(item.episodes)} | {item.peak_confidence:.2f} |")
    lines += ["", "## Why each of the top five", ""]
    for index, item in enumerate(ranked[:5], 1):
        lines.append(f"**{index}. Track {item.track_id}** "
                     f"({item.sightings} sightings, "
                     f"{item.wrist_resolved_sightings} with a wrist)")
        lines += [f"- {reason}" for reason in item.reasons]
        lines.append("")
    lines += [
        "## What this does not establish",
        "",
        "The detector cannot tell a chit from a legitimate answer sheet -- "
        "both are small, pale and held. These rankings say *who was handling "
        "a small pale object, and for how long*. Whether that object belongs "
        "on the desk is a reviewer's decision.",
        "",
        "There is no reference manifest for this recording, so none of the "
        "above is a precision or recall figure.",
        "",
        f"Detector: `chit-paper-new/2`. {__import__('pipeline.chit_detector', fromlist=['ATTRIBUTION']).ATTRIBUTION}",
    ]
    (stage / "CHIT_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"{raw} raw -> {kept_total} kept detections")
    print(f"{len(ranked)} tracks with surviving evidence, of {len(evidence)}")
    print(f"\n{'rank':>4} {'track':>6} {'handling':>9} {'longest':>8} "
          f"{'eps':>4} {'peak':>5}  disposition")
    for index, item in enumerate(ranked[:12], 1):
        print(f"{index:>4} {item.track_id:>6} "
              f"{item.total_handling_ms / 1000:>8.1f}s "
              f"{item.longest_episode_ms / 1000:>7.1f}s "
              f"{len(item.episodes):>4} {item.peak_confidence:>5.2f}  "
              f"{item.disposition}")
    print(f"\n-> {stage / 'CHIT_FINDINGS.md'}")
    print(f"-> {gated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
