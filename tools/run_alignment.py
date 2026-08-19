"""PRD 6.3.1-6.3.5: global motion estimation, quality-gated, with artifacts.

    python tools/run_alignment.py --run artifacts/runs/<run_id> \
        --calibration calib/cam12_e1_v1.json VIDEO

Estimates the global transform between a reference frame and frames sampled
across the recording, using three named configurations on identical inputs:

    identity        no compensation -- the baseline PRD 6.3.2 requires be named
    features_ransac ORB features on static background, RANSAC-fitted
    ecc             OpenCV ECC, a measured challenger with a convergence score

PRD 6.3.1 says the fit must use *static background support*: pixels excluding
seats, people, monitors, aisles and reflections. Fitting on pixels containing
people measures the people, not the camera. The support mask comes from the
calibration profile, which is why this stage needs one.

PRD 6.3.3 requires raw, support mask, features, transform, confidence,
compensated frame, difference image and failure reason to be saved for every
estimate. PRD 6.3.4 requires compensation to be applied only when inlier
coverage, residual and confidence pass immutable thresholds -- otherwise the
span is labelled `camera_motion_uncertain` and the raw evidence is retained.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pipeline import alignment as al, artifacts  # noqa: E402
from pipeline.calibration import CalibrationProfile, background_support  # noqa: E402
from pipeline.run_manifest import COMPLETE, RunManifest, RunStatus  # noqa: E402


def sample_frames(video: Path, count: int):
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration_s = float(container.duration / 1e6) if container.duration else 0.0
        time_base = float(stream.time_base)
        wanted = [duration_s * (i + 0.5) / count for i in range(count)] \
            if duration_s else []
        out, index = [], 0
        for frame in container.decode(stream):
            if frame.pts is None or index >= len(wanted):
                continue
            pts_s = float(frame.pts * time_base)
            if pts_s >= wanted[index]:
                out.append((pts_s * 1000.0, frame.to_ndarray(format="bgr24")))
                index += 1
        return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--save-images", type=int, default=4)
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    if paths.sealed:
        print(f"{args.run} is sealed.", file=sys.stderr)
        return 2
    status = RunStatus(paths, RunManifest.read(paths).run_id)
    profile = CalibrationProfile.read(args.calibration)

    frames = sample_frames(args.video, args.frames)
    if len(frames) < 2:
        print("need at least two frames", file=sys.stderr)
        return 2

    support = background_support(profile)
    coverage = float(support.mean())
    print(f"{len(frames)} frames sampled; static background support covers "
          f"{coverage * 100:.1f}% of the frame")
    if coverage < 0.02:
        print("  WARNING: almost no static support. A transform fitted here "
              "would be measuring people, not the camera (PRD 6.3.1).")

    reference_pts, reference = frames[0]
    cfg = al.AlignmentConfig()
    try:
        import cv2
    except Exception:                                    # noqa: BLE001
        cv2 = None

    out_dir = paths.stage("03_alignment")
    all_results, per_config = [], {}
    saved = 0

    for pts_ms, frame in frames[1:]:
        results = al.compare(reference, frame, support, cfg, pts_ms=pts_ms)
        chosen = al.select(results, cfg)
        all_results.extend(results)

        for result in results:
            per_config.setdefault(result.method, []).append(result)
            paths.append_jsonl("03_alignment/estimates.jsonl",
                               {"pts_ms": round(pts_ms, 3), **result.to_row()})

        # PRD 6.3.3 artifacts, for the first few timestamps.
        if cv2 is not None and saved < args.save_images:
            prefix = artifacts.pts_prefix(pts_ms)
            stamp_dir = out_dir / prefix
            stamp_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(stamp_dir / "raw.png"), frame)
            cv2.imwrite(str(stamp_dir / "reference.png"), reference)
            cv2.imwrite(str(stamp_dir / "static_mask.png"), support * 255)
            compensated = al.apply(frame, chosen) if chosen else frame
            cv2.imwrite(str(stamp_dir / "compensated.png"), compensated)
            cv2.imwrite(str(stamp_dir / "difference.png"),
                        np.abs(compensated.astype(np.int16)
                               - reference.astype(np.int16)).astype(np.uint8))
            paths.write_json(f"03_alignment/{prefix}/metrics.json", {
                "pts_ms": round(pts_ms, 3),
                "reference_pts_ms": round(reference_pts, 3),
                "support_coverage": round(coverage, 5),
                "estimates": [r.to_row() for r in results],
                "selected": chosen.to_row() if chosen else None,
                "selection_reason":
                    chosen.reason if chosen else
                    "no configuration passed the quality gate; the span is "
                    "camera_motion_uncertain and raw evidence is retained "
                    "(PRD 6.3.4)",
            })
            saved += 1

    print("\nper configuration, over all sampled frames:")
    summary = {}
    for method, results in sorted(per_config.items()):
        residuals = [r.residual_px for r in results]
        passed = sum(1 for r in results if r.compensate)
        summary[method] = {
            "estimates": len(results),
            "passed_gate_and_compensated": passed,
            "median_residual": round(float(np.median(residuals)), 4),
            "median_confidence": round(
                float(np.median([r.confidence for r in results])), 4),
            "median_translation_px": round(
                float(np.median([r.translation_px for r in results])), 3),
        }
        print(f"  {method:16} {passed:3}/{len(results):3} compensated  "
              f"median residual {summary[method]['median_residual']:8.4f}  "
              f"confidence {summary[method]['median_confidence']:.3f}  "
              f"translation {summary[method]['median_translation_px']:6.2f} px")

    breaks = al.detect_epoch_breaks(all_results, cfg)
    report = {
        "reference_pts_ms": round(reference_pts, 3),
        "frames_compared": len(frames) - 1,
        "static_support_coverage": round(coverage, 5),
        "configurations": summary,
        "epoch_breaks": [{"pts_ms": b.pts_ms, "translation_px": b.translation_px,
                          "sustained_ms": b.sustained_ms, "reason": b.reason}
                         for b in breaks],
        "note": "identity is a named baseline, not the absence of a "
                "configuration (PRD 6.3.2). Compensation is applied only where "
                "the gate passes; elsewhere the span is camera_motion_uncertain "
                "and raw evidence is retained (PRD 6.3.4).",
    }
    paths.write_json("03_alignment/comparison.json", report)

    if breaks:
        print(f"\n{len(breaks)} camera epoch break(s) detected; seat "
              f"attribution is blocked past them until calibration is "
              f"re-approved (PRD 6.3.5)")
    else:
        print("\nno sustained geometry change; the recording is one camera epoch")

    status.finish("03_alignment", COMPLETE,
                  reason=f"{len(frames) - 1} frames compared under "
                         f"{len(summary)} named configurations",
                  epoch_breaks=len(breaks))
    status.write()
    print(f"written {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
