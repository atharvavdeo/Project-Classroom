"""Project a run's samples into per-frame overlay geometry for the console.

The annotated MP4 the pipeline renders is a *baked* image: you cannot turn one
layer off, and you cannot ask it which track a box belonged to. Drawing the
same geometry on a canvas over the raw video gives both, at a fraction of the
bytes -- and it is what makes the wipe comparison honest, because the left and
right halves are then the same decoded frame with and without our marks, not
two different encodes.

Size is the whole design constraint. `samples.jsonl` is 12 MB because it
carries every joint with its confidence. Here coordinates are rounded to whole
pixels (sub-pixel precision is invisible at 1280x720 and costs 4 bytes a
number), confidences are dropped from joints, and keys are single letters.
Measured on 1512/12_paper: 12 MB -> ~1.2 MB.

Frames are emitted sorted by pts so the client can binary-search the nearest
sample for the current playback time.

Usage:
    python tools/build_overlay_bundle.py --run artifacts/runs/1512/12_paper \\
        --out console/public/data/1512_12_paper.overlay.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Joints the console draws as a skeleton. Ordered so the client can zip them
# into bone pairs without a lookup table.
SKELETON_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
)

# Bones, as index pairs into SKELETON_JOINTS.
BONES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (0, 1), (0, 2), (1, 3), (2, 4),
)


def _round_box(box) -> list | None:
    if not box or len(box) < 4:
        return None
    return [round(float(v)) for v in box[:4]]


def build(run: Path, samples_name: str) -> dict:
    stage = run / "14_person_timeline"
    path = stage / samples_name
    if not path.exists():
        raise SystemExit(f"ERROR: {path} does not exist")

    by_pts: dict[float, list] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        pts = round(float(row.get("pts_ms") or 0.0), 1)

        joints = row.get("joints") or {}
        # Emit a flat [x, y, x, y, ...] with -1 for a joint that was not
        # visible. A sentinel keeps the array a fixed length, which is both
        # smaller than a dict and simpler for the client to draw.
        flat: list[int] = []
        for name in SKELETON_JOINTS:
            j = joints.get(name)
            if isinstance(j, (list, tuple)) and len(j) >= 2:
                flat.extend([round(float(j[0])), round(float(j[1]))])
            else:
                flat.extend([-1, -1])

        objects = []
        for obj in row.get("near_hand_objects") or []:
            box = _round_box(obj.get("box"))
            if box is None:
                continue
            objects.append({
                "c": str(obj.get("cls") or ""),
                "b": box,
                # Confidence and wrist distance are what a reviewer hovers to
                # see, so they survive the trim.
                "p": round(float(obj.get("confidence") or 0.0), 3),
                "d": (round(float(obj["wrist_distance_norm"]), 3)
                      if obj.get("wrist_distance_norm") is not None else None),
                "w": str(obj.get("nearest_wrist") or ""),
                "s": str(obj.get("source") or ""),
            })

        person = {
            "i": int(row.get("track_id", -1)),
            "b": _round_box(row.get("box")) or [0, 0, 0, 0],
            "j": flat,
            "f": str(row.get("facing") or ""),
            "st": str(row.get("seat_state") or ""),
        }
        if objects:
            person["o"] = objects
        by_pts[pts].append(person)

    frames = [{"t": pts, "p": people} for pts, people in sorted(by_pts.items())]

    return {
        "schema": 1,
        "joints": list(SKELETON_JOINTS),
        "bones": [list(b) for b in BONES],
        "frames": frames,
    }


def frame_size(run: Path) -> tuple[int, int]:
    """The source frame size, read from the run's own ingest inventory.

    Every coordinate in this bundle is in source pixels, so the consumer needs
    the frame it was measured against. The console used to hard-code 1280x720
    beside the media, which is right for most of this corpus and wrong for
    `04.CCTV Candidate Talking.mkv` -- that one is 640x480, and a canvas told
    it was 1280x720 draws every box at half scale in the top-left quadrant.
    Carrying the number the run recorded removes the guess.
    """
    seen: list[Path] = []
    here: Path | None = run
    # A resumed run keeps only the stages it re-ran; its ingest lives in the
    # parent it branched from. Walk the chain rather than treating a missing
    # inventory as "unknown" -- 1512 has no source/ directory at all.
    while here is not None and here not in seen:
        seen.append(here)
        inventory = here / "source" / "video_inventory.jsonl"
        if inventory.exists():
            for line in inventory.open(encoding="utf-8"):
                row = json.loads(line)
                w, h = row.get("width"), row.get("height")
                if w and h:
                    return int(w), int(h)
        manifest = here / "RUN_MANIFEST.json"
        parent = None
        if manifest.exists():
            parent = json.loads(manifest.read_text(encoding="utf-8")).get(
                "parent_run_id"
            )
        here = here.parent.parent / str(parent) / run.name if parent else None

    raise SystemExit(
        f"ERROR: no video_inventory.jsonl with width/height for {run} or any "
        f"parent run ({', '.join(p.name for p in seen)}). Refusing to guess a "
        f"frame size -- the overlay would be silently misaligned."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", default="samples_sam3_adjudicated.jsonl",
                    help="which samples file to project; the adjudicated one "
                         "carries the SAM 3 verdicts, so it is the default")
    ap.add_argument("--frame-size", metavar="WxH",
                    help="source frame size, for a resumed run whose ingest "
                         "inventory is not on disk anywhere in its parent "
                         "chain. Verify it against the media before passing "
                         "it -- this is the one number nothing else can check")
    args = ap.parse_args()

    stage = args.run / "14_person_timeline"
    name = args.samples
    if not (stage / name).exists():
        # Fall back down the chain rather than failing: a run that stopped
        # before adjudication still has drawable geometry.
        for candidate in ("samples_chit_gated.jsonl", "samples_with_chits.jsonl",
                          "samples.jsonl"):
            if (stage / candidate).exists():
                name = candidate
                break

    doc = build(args.run, name)
    if args.frame_size:
        w, h = args.frame_size.lower().split("x")
        doc["width"], doc["height"] = int(w), int(h)
    else:
        doc["width"], doc["height"] = frame_size(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")

    people = sum(len(f["p"]) for f in doc["frames"])
    objects = sum(len(p.get("o") or []) for f in doc["frames"] for p in f["p"])
    size_kb = args.out.stat().st_size / 1024
    print(f"{args.run.parent.name}/{args.run.name} [{name}]: "
          f"{len(doc['frames'])} frames, {people} person-samples, "
          f"{objects} objects -> {args.out} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
