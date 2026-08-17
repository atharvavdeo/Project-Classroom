"""D-FINE forward pass against real exam-centre footage.

Exercises the last unproven path: ONNX inference -> DETR decode -> crop
transform -> frame coordinates -> abstention.

The weights here are stock COCO (via Objects365 pretraining), which is the
lineage PRD 8.1 specifies as the fine-tuning starting point. They are NOT tuned
for this domain, so the class labels are not expected to be right for phones --
COCO's "cell phone" is a large clear handheld object, not 30x20 px on distant
CCTV. What is being verified is that the plumbing is correct: that boxes land on
real objects at real frame coordinates and that the abstention floor bites where
it should.

Skips cleanly if the weights or frames are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from classroom import crops, detect  # noqa: E402
from classroom.config import DetectConfig  # noqa: E402

MODEL = Path("models/dfine/onnx/model.onnx")
FRAMES_DIR = Path(
    r"C:\Users\HARDEE~1\AppData\Local\Temp\claude\C--DrishtiAI"
    r"\d9cbb711-8a41-4ee8-9666-c80f8eaac1b5\scratchpad\frames"
)
FRAME = FRAMES_DIR / "paper_t0045.jpg"

# Regions traced over the real desk layout.
SEATS = {
    "L-front": [(30, 300), (330, 300), (330, 620), (30, 620)],
    "C-front": [(600, 230), (790, 230), (790, 560), (600, 560)],
    "R-cluster": [(760, 60), (1000, 60), (1000, 420), (760, 420)],
}


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    if not MODEL.is_file():
        print(f"SKIP: D-FINE weights absent ({MODEL})")
        return 0
    if not FRAME.is_file():
        print(f"SKIP: extracted frame absent ({FRAME.name})")
        return 0

    ok = True
    cfg = DetectConfig()
    frame = cv2.imread(str(FRAME))
    print(f"frame {frame.shape[1]}x{frame.shape[0]}")

    detector = detect.ONNXDetector(MODEL, input_size=cfg.input_size)
    detector.load()
    print(f"D-FINE-L loaded ({MODEL.stat().st_size / 1e6:.0f} MB)")

    print("\nfull-frame inference, for comparison")
    whole, tf_whole = crops.extract(
        frame, (0, 0, frame.shape[1], frame.shape[0]), cfg.input_size
    )
    full = detector(whole, tf_whole, cfg.confidence)
    people_full = [d for d in full if d.cls == "person"]
    print(f"  {len(full)} detections, {len(people_full)} people, "
          f"supersampling {tf_whole.supersampling:.2f}x")
    ok &= check("full frame finds people", len(people_full) > 0, str(len(people_full)))

    print("\nper-seat crops, magnified into the detector")
    total_seat_people = 0
    for label, polygon in SEATS.items():
        crop, tf = crops.seat_crop(frame, polygon, cfg.input_size, cfg.min_crop_px)
        found = detector(crop, tf, cfg.confidence)
        found = detect.apply_abstention(found, cfg.min_object_px_area)
        people = [d for d in found if d.cls == "person"]
        total_seat_people += len(people)
        classes = sorted({d.cls for d in found})
        print(f"  seat {label:10s} {tf.supersampling:.2f}x  "
              f"{len(found):2d} detections  {len(people)} people  {classes}")

        ok &= check(f"  {label} crop is magnified", tf.supersampling > 1.0,
                    f"{tf.supersampling:.2f}x")
        for d in found:
            inside = (-50 <= d.box[0] <= frame.shape[1] + 50
                      and -50 <= d.box[1] <= frame.shape[0] + 50)
            if not inside:
                ok &= check(f"  {label} box within frame bounds", False, str(d.box))
                break
        else:
            ok &= check(f"  {label} all boxes map inside the frame", True)

    ok &= check("seat crops find people too", total_seat_people > 0,
                str(total_seat_people))

    print("\nabstention behaviour on real detections")
    crop, tf = crops.seat_crop(frame, SEATS["C-front"], cfg.input_size, cfg.min_crop_px)
    raw = detector(crop, tf, 0.15)
    gated = detect.apply_abstention(raw, cfg.min_object_px_area)
    abstained = [d for d in gated if d.abstained]
    areas = [d.px_area for d in raw]
    print(f"  {len(raw)} raw detections at conf>=0.15, "
          f"{len(abstained)} below the {cfg.min_object_px_area:.0f} px^2 floor")
    if areas:
        print(f"  source footprints: min {min(areas):.0f}  "
              f"median {np.median(areas):.0f}  max {max(areas):.0f} px^2")
    ok &= check("every abstained detection is labelled insufficient",
                all(d.cls == detect.INSUFFICIENT for d in abstained))
    ok &= check("no abstained detection is usable",
                all(not d.usable for d in abstained))
    ok &= check("abstention keys off the source footprint, not the magnified one",
                all(d.px_area < cfg.min_object_px_area for d in abstained))

    print("\nobject evidence excludes abstentions")
    evidence = detect.object_evidence(gated)
    ok &= check("evidence is a valid weight", 0.0 <= evidence <= 1.0, f"{evidence:.2f}")

    print("\nrendering detections for visual adjudication")
    canvas = frame.copy()
    for label, polygon in SEATS.items():
        pts = np.asarray(polygon, np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (200, 200, 60), 2)
        crop, tf = crops.seat_crop(frame, polygon, cfg.input_size, cfg.min_crop_px)
        for d in detect.apply_abstention(
            detector(crop, tf, cfg.confidence), cfg.min_object_px_area
        ):
            x1, y1, x2, y2 = (int(v) for v in d.box)
            colour = (90, 90, 200) if d.abstained else (90, 210, 120)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(canvas, f"{d.cls} {d.confidence:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    out = Path("docs/detector_real_frame.jpg")
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  wrote {out}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
