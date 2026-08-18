"""Run D-FINE over every extracted frame and report domain measurements."""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from classroom import crops, detect
from classroom.config import DetectConfig

MODEL = Path("models/dfine/onnx/model.onnx")
FRAMES = Path(sys.argv[1] if len(sys.argv) > 1 else "data/frames")
CONF = 0.30


def main() -> int:
    if not MODEL.is_file():
        print(f"detector weights absent: {MODEL}")
        return 1
    frames = sorted(FRAMES.glob("*.jpg"))
    if not frames:
        print(f"no frames under {FRAMES}")
        return 1

    cfg = DetectConfig()
    det = detect.ONNXDetector(MODEL, input_size=cfg.input_size)
    det.load()

    per_cam = defaultdict(lambda: {"frames": 0, "person": 0, "phone": [], "paper": []})
    all_classes = Counter()

    for path in frames:
        cam = path.stem.split("_")[0]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        whole, tf = crops.extract(img, (0, 0, w, h), cfg.input_size)
        found = detect.apply_abstention(det(whole, tf, CONF), cfg.min_object_px_area)

        per_cam[cam]["frames"] += 1
        for d in found:
            all_classes[d.cls] += 1
            if d.cls == "person":
                per_cam[cam]["person"] += 1
            elif d.cls == "phone_like":
                per_cam[cam]["phone"].append(d.px_area)
            elif d.cls == "paper_like":
                per_cam[cam]["paper"].append(d.px_area)

    print(f"{'camera':8s} {'frames':>6s} {'persons':>8s} {'per frame':>10s} "
          f"{'phone_like':>11s} {'paper_like':>11s}")
    print("-" * 60)
    for cam in sorted(per_cam):
        s = per_cam[cam]
        print(f"{cam:8s} {s['frames']:>6d} {s['person']:>8d} "
              f"{s['person'] / s['frames']:>10.1f} "
              f"{len(s['phone']):>11d} {len(s['paper']):>11d}")

    print("\nall detected classes:")
    for cls, n in all_classes.most_common():
        print(f"  {cls:14s} {n}")

    phone_areas = [a for s in per_cam.values() for a in s["phone"]]
    paper_areas = [a for s in per_cam.values() for a in s["paper"]]
    print("\nsource-frame footprints (px^2), full-frame inference:")
    for label, areas in (("phone_like", phone_areas), ("paper_like", paper_areas)):
        if areas:
            a = np.array(areas)
            below = int((a < cfg.min_object_px_area).sum())
            print(f"  {label:11s} n={len(a):3d}  min={a.min():7.0f}  "
                  f"median={np.median(a):7.0f}  max={a.max():8.0f}  "
                  f"below {cfg.min_object_px_area:.0f} px^2 floor: {below}")
        else:
            print(f"  {label:11s} none detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
