"""Crop every phone_like detection so they can be adjudicated by eye."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from classroom import crops, detect
from classroom.config import DetectConfig

MODEL = Path("models/dfine/onnx/model.onnx")
FRAMES = Path("data/frames")
OUT = Path("docs/phone_candidates.jpg")
cfg = DetectConfig()
det = detect.ONNXDetector(MODEL, input_size=cfg.input_size); det.load()

tiles = []
for path in sorted(FRAMES.glob("*.jpg")):
    img = cv2.imread(str(path)); h, w = img.shape[:2]
    whole, tf = crops.extract(img, (0, 0, w, h), cfg.input_size)
    for d in detect.apply_abstention(det(whole, tf, 0.30), cfg.min_object_px_area):
        if d.cls != "phone_like":
            continue
        x1, y1, x2, y2 = (int(v) for v in d.box)
        pad = 26
        x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
        x2, y2 = min(x2 + pad, w), min(y2 + pad, h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        tile = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
        cv2.putText(tile, f"{d.px_area:.0f}px2", (4, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"{path.stem.split('_')[0]} {d.confidence:.2f}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 230, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

cols = 8
rows = (len(tiles) + cols - 1) // cols
sheet = np.zeros((rows * 160, cols * 160, 3), np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    sheet[r*160:(r+1)*160, c*160:(c+1)*160] = t
OUT.parent.mkdir(exist_ok=True)
cv2.imwrite(str(OUT), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
print(f"{len(tiles)} phone_like candidates -> {OUT}")
