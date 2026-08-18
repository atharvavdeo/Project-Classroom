"""Measured comparison of the pose models on real frames.

    python tools/pose_ab.py data/frames --models topdown rtmo-m rtmo-l \
        --sheet docs/pose_ab.jpg

The earlier comparison (PRD 8.2.1) put top-down 2.3x ahead of one-stage RTMO on
person recall, but it used RTMO-m. This runs the large model as well, so the
choice rests on the strongest one-stage option rather than the convenient one.

What is counted, and why these and not AP:

  * **people** -- how many candidates the model finds at all. There is no
    keypoint ground truth for this corpus, so AP is not computable here. Person
    recall is, and it is the quantity that governs the pipeline: a candidate the
    pose model never sees contributes no head-pitch and no below-desk evidence,
    whatever the keypoint accuracy would have been.

  * **resolvable pitch** -- people whose nose and both shoulders are confident
    enough to compute head pitch. A skeleton without those is a detection the
    scorer cannot use, so counting raw detections alone would flatter a model
    that finds many people badly.

  * **seconds per frame** -- measured after a warm-up pass, since the first call
    pays for lazy weight loading.

A contact sheet is written so the counts can be checked by eye. That check
matters: the first version of this comparison could not distinguish extra
genuine people from duplicate skeletons stacked on one person, and only the
overlay settled it.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import pose  # noqa: E402

# COCO-17 skeleton, drawn only to make the overlay readable.
LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

KEYPOINT_CONF = 0.3


@dataclass
class ModelResult:
    name: str
    people: int = 0
    pitch_resolved: int = 0
    seconds: float = 0.0
    frames: int = 0
    per_frame: list[tuple[str, int, int]] = field(default_factory=list)
    error: str | None = None

    @property
    def s_per_frame(self) -> float | None:
        return self.seconds / self.frames if self.frames else None


def draw_skeleton(image: np.ndarray, people: list[np.ndarray]) -> np.ndarray:
    out = image.copy()
    for person in people:
        for a, b in LIMBS:
            if person[a, 2] >= KEYPOINT_CONF and person[b, 2] >= KEYPOINT_CONF:
                cv2.line(out, (int(person[a, 0]), int(person[a, 1])),
                         (int(person[b, 0]), int(person[b, 1])),
                         (80, 220, 120), 1, cv2.LINE_AA)
        for x, y, conf in person:
            if conf >= KEYPOINT_CONF:
                cv2.circle(out, (int(x), int(y)), 2, (60, 60, 255), -1)
    return out


def run_model(name: str, frames: list[Path], device: str
              ) -> tuple[ModelResult, dict[str, list[np.ndarray]]]:
    result = ModelResult(name=name)
    detections: dict[str, list[np.ndarray]] = {}
    estimator = pose.PoseEstimator(model=name, device=device)

    try:
        warmup = cv2.imread(str(frames[0]))
        estimator(warmup)
    except Exception as exc:  # noqa: BLE001 - one model failing must not stop the rest
        result.error = f"{type(exc).__name__}: {exc}"
        return result, detections

    for path in frames:
        image = cv2.imread(str(path))
        if image is None:
            continue
        start = time.perf_counter()
        people = estimator(image)
        result.seconds += time.perf_counter() - start
        result.frames += 1

        resolved = sum(
            1 for person in people if pose.head_pitch(person) is not None
        )
        result.people += len(people)
        result.pitch_resolved += resolved
        result.per_frame.append((path.name, len(people), resolved))
        detections[path.name] = people

    return result, detections


def contact_sheet(
    frames: list[Path], per_model: dict[str, dict[str, list[np.ndarray]]],
    out_path: Path, cell_width: int = 480,
) -> Path:
    """One row per frame, one column per model."""
    names = list(per_model)
    rows = []
    for path in frames:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = cell_width / image.shape[1]
        cells = []
        for name in names:
            people = per_model[name].get(path.name, [])
            drawn = draw_skeleton(image, people)
            cell = cv2.resize(drawn, (cell_width, int(image.shape[0] * scale)))
            cv2.rectangle(cell, (0, 0), (cell_width - 1, 20), (0, 0, 0), -1)
            cv2.putText(cell, f"{name}  {len(people)} people", (6, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cells.append(cell)
        height = min(c.shape[0] for c in cells)
        rows.append(np.hstack([c[:height] for c in cells]))
    if not rows:
        raise SystemExit("no frames could be read")
    width = min(r.shape[1] for r in rows)
    sheet = np.vstack([r[:, :width] for r in rows])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, help="directory of extracted stills")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--limit", type=int, default=12,
                    help="frames to run; each model sees exactly the same ones")
    ap.add_argument("--models", nargs="+", default=list(pose.MODELS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sheet", type=Path, default=Path("docs/pose_ab.jpg"))
    args = ap.parse_args()

    frames = sorted(args.frames.glob(args.glob))
    if not frames:
        print(f"no frames matching {args.glob} under {args.frames}", file=sys.stderr)
        return 2
    # Spread the sample across the directory rather than taking the first N,
    # which on this naming scheme would be one camera repeated.
    if len(frames) > args.limit:
        step = len(frames) / args.limit
        frames = [frames[int(i * step)] for i in range(args.limit)]

    print(f"{len(frames)} frames, {args.device}, models: {', '.join(args.models)}\n")

    results: list[ModelResult] = []
    per_model: dict[str, dict[str, list[np.ndarray]]] = {}
    for name in args.models:
        print(f"running {name} ...", flush=True)
        result, detections = run_model(name, frames, args.device)
        results.append(result)
        if result.error:
            print(f"  FAILED  {result.error}\n")
            continue
        per_model[name] = detections
        print(f"  {result.people} people, {result.pitch_resolved} with "
              f"resolvable pitch, {result.s_per_frame:.2f} s/frame\n")

    usable = [r for r in results if not r.error]
    if not usable:
        print("every model failed; nothing to compare", file=sys.stderr)
        return 1

    print(f"{'model':<10} {'people':>7} {'pitch':>7} {'s/frame':>8} "
          f"{'people/frame':>13}")
    for r in usable:
        print(f"{r.name:<10} {r.people:>7} {r.pitch_resolved:>7} "
              f"{r.s_per_frame:>8.2f} {r.people / r.frames:>13.1f}")

    best = max(usable, key=lambda r: r.people)
    print(f"\nmost people found: {best.name}")
    for r in usable:
        if r is best or r.people == 0:
            continue
        print(f"  {best.people / r.people:.2f}x {r.name}"
              f"   (pitch {best.pitch_resolved / max(r.pitch_resolved, 1):.2f}x,"
              f" cost {best.s_per_frame / r.s_per_frame:.2f}x)")

    print("\nper frame")
    header = f"{'frame':<22}" + "".join(f"{r.name:>10}" for r in usable)
    print(header)
    for i, path in enumerate(frames):
        row = f"{path.name[:22]:<22}"
        for r in usable:
            row += f"{r.per_frame[i][1]:>10}" if i < len(r.per_frame) else f"{'-':>10}"
        print(row)

    if per_model:
        sheet = contact_sheet(frames[:6], per_model, args.sheet)
        print(f"\ncontact sheet: {sheet}")
        print("Check it. Counts alone cannot tell extra genuine people from "
              "duplicate skeletons stacked on one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
