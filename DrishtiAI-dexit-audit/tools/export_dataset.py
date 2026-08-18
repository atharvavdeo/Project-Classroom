"""Export box annotations as a COCO detection set for fine-tuning.

    python tools/export_dataset.py --db data/classroom.db --out data/dataset \
        --val-camera cam12

The split is by camera, never by frame. Frames extracted from one recording
share a background, a lens, a compression history and often the same people, so
a random frame split lets the model memorise the room and report a validation
score that collapses the moment it sees a new one. Holding out an entire camera
measures the only thing worth knowing: whether it transfers.

The exporter refuses rather than produces a technically valid but useless set.
Two refusals, both deliberate:

  * fewer than `--min-boxes` labels in total -- a fine-tune on a handful of
    boxes does not improve a detector, it overfits it, and the resulting
    checkpoint would be worse than the stock weights it replaced while looking
    like progress;

  * a validation camera with no labels at all -- the run would print a
    validation AP computed over nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import annotate, db  # noqa: E402
from classroom.annotate import BOX_CLASSES, NEGATIVE_CLASSES  # noqa: E402


def camera_of(frame_path: str) -> str:
    """Derive the camera tag from an extracted frame's filename.

    tools/extract_frames.py writes `<camera>_t<seconds>.jpg`, so the tag is
    everything before the last underscore.
    """
    stem = Path(frame_path).stem
    return stem.rsplit("_t", 1)[0] if "_t" in stem else stem


def image_sizes(paths: list[str]) -> dict[str, tuple[int, int]]:
    """Read each frame's real dimensions.

    Measured rather than assumed: the corpus mixes 1280x720 with 640x480 (PRD
    5.1), and a COCO file whose declared sizes are wrong trains a model on
    boxes that do not sit where it thinks they do.
    """
    sizes: dict[str, tuple[int, int]] = {}
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(
                f"{path} is annotated but cannot be read; re-extract it or "
                f"remove its labels before exporting"
            )
        sizes[path] = (image.shape[1], image.shape[0])
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/classroom.db"))
    ap.add_argument("--out", type=Path, default=Path("data/dataset"))
    ap.add_argument("--val-camera", default=None,
                    help="camera tag held out for validation; "
                         "omit to list what is available and stop")
    ap.add_argument("--min-boxes", type=int, default=200,
                    help="refuse to export below this many labels")
    args = ap.parse_args()

    conn = db.connect(args.db)
    boxes = annotate.load_boxes(conn)
    reviewed = annotate.reviewed_frames(conn)
    conn.close()

    all_frames = sorted(set(reviewed) | {b.frame_path for b in boxes})
    if not all_frames:
        print("No frames have been reviewed. Label some with "
              "tools/annotate_boxes.py first.", file=sys.stderr)
        return 2

    print(annotate.dataset_summary(boxes, reviewed))
    print()

    by_camera: dict[str, list[str]] = {}
    for path in all_frames:
        by_camera.setdefault(camera_of(path), []).append(path)
    box_counts: dict[str, int] = {}
    for box in boxes:
        camera = camera_of(box.frame_path)
        box_counts[camera] = box_counts.get(camera, 0) + 1

    print(f"{'camera':<12} {'frames':>7} {'boxes':>7}")
    for camera in sorted(by_camera):
        print(f"{camera:<12} {len(by_camera[camera]):>7} "
              f"{box_counts.get(camera, 0):>7}")
    print()

    if args.val_camera is None:
        print("Pick one of the cameras above with --val-camera. It is held out "
              "entirely, so choose the one whose room you most want the model "
              "to generalise to.")
        return 1

    if args.val_camera not in by_camera:
        print(f"unknown camera {args.val_camera!r}", file=sys.stderr)
        return 2
    if box_counts.get(args.val_camera, 0) == 0:
        print(f"{args.val_camera} has no labelled boxes, so validation AP would "
              f"be computed over nothing. Annotate it, or hold out a different "
              f"camera.", file=sys.stderr)
        return 2
    if len(boxes) < args.min_boxes:
        print(f"only {len(boxes)} boxes labelled; below {args.min_boxes} a "
              f"fine-tune overfits rather than improves. Keep annotating, or "
              f"lower --min-boxes deliberately and record that you did.",
              file=sys.stderr)
        return 2

    negatives = sum(1 for b in boxes if b.cls in NEGATIVE_CLASSES)
    if negatives == 0:
        print("No negative boxes are labelled. The measured stock-weight "
              "failure on this corpus is false positives on computer mice, "
              "keyboards and monitor bezels (PRD 8.1.2); training only on "
              "positives will not touch it.", file=sys.stderr)
        return 2

    val_frames = set(by_camera[args.val_camera])
    train_frames = [p for p in all_frames if p not in val_frames]
    sizes = image_sizes(all_frames)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, frames in (("train", train_frames), ("val", sorted(val_frames))):
        subset = [b for b in boxes if b.frame_path in set(frames)]
        coco = annotate.to_coco(subset, frames, sizes, BOX_CLASSES)
        path = annotate.write_coco(args.out / f"{name}.json", coco)
        with_boxes = len({b.frame_path for b in subset})
        print(f"{name:<6} {len(frames):>4} images "
              f"({with_boxes} with boxes, {len(frames) - with_boxes} background), "
              f"{len(subset):>4} boxes  ->  {path}")

    print(f"\nheld out: {args.val_camera}")
    print(f"negatives in the set: {negatives} "
          f"({negatives / len(boxes) * 100:.0f}% of all boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
