"""The 1446 run: three videos, locked architecture, full artifact set.

    python tools/run_1446.py

Runs the reverted-to architecture -- AlphaPose for pose, D-FINE native and
D-FINE+SAHI for objects, no VLM anywhere -- over three recordings, into
`artifacts/runs/1446/<video>/`.

Two settings differ from the corpus driver and both are deliberate:

**Sampling at 3 Hz, not 1 Hz.** Head and neck behaviour is the point of this
run, and 1 Hz cannot support it. Measured on the previous 1 Hz pass of video
01: 379 observations fragmented across 29 tracks, most holding one or two
observations, and zero sustained head events could form because a 1.5 s event
needs consecutive samples closer together than the 1000 ms sample period.
Denser sampling also gives the IoU tracker smaller inter-frame motion to match
against, so tracks survive longer.

**RF-DETR is skipped.** It is recorded unavailable rather than omitted, per
PRD 5. The locked detector is D-FINE, native and SAHI.

Per video the run folder holds ingest and preprocessing measurements, camera
alignment, motion/ROI/segmentation, tracking, the pose A/B on frozen identical
rows, the head and neck analysis with overlays, the detector A/B on frozen
identical crops, rendered detections, and evidence packets.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = Path("C:/DrishtiAI")

# (label, filename, calibration). Each video gets its own run directory inside
# the 1446 folder so one recording's artifacts can never overwrite another's.
VIDEOS = (
    ("01_phone", "01.Candidate was found using a mobile phone in the "
                 "examination hall..mkv", "calib/cam01_e1_v1.json"),
    ("04_talking", "04.CCTV Candidate Talking.mkv", "calib/cam04_e1_v1.json"),
    ("12_paper", "Seat No. 12 was seen taking a piece of paper from the "
                 "desk.mkv", "calib/campaper_e1_v1.json"),
)


def run(label: str, argv: list[str], log: Path, env: dict) -> tuple[int, float]:
    started = time.perf_counter()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{'=' * 72}\n{label}\n{'=' * 72}\n")
        handle.flush()
        done = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env,
                              stdout=handle, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - started
    status = "ok " if done.returncode == 0 else f"EXIT {done.returncode}"
    print(f"    {label:46} {status:8} {seconds:7.1f}s", flush=True)
    return done.returncode, seconds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--tag", default="1446")
    ap.add_argument("--sample-hz", type=float, default=3.0)
    ap.add_argument("--max-events", type=int, default=60)
    ap.add_argument("--only", nargs="*", default=None)
    # AlphaPose is torch; D-FINE is onnxruntime and picks CUDAExecutionProvider
    # by itself when the provider loads, so only pose needs telling.
    ap.add_argument("--pose-device", default="cuda",
                    help="cuda or cpu. Falls back to cpu inside the pose stage "
                         "if torch reports no CUDA device.")
    ap.add_argument("--dfine", type=Path,
                    default=ROOT / "models/dfine/onnx/model.onnx")
    args = ap.parse_args()

    import os
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONUNBUFFERED"] = "1"

    base = ROOT / "artifacts/runs" / args.tag
    base.mkdir(parents=True, exist_ok=True)
    summary = []

    for label, filename, calibration in VIDEOS:
        if args.only and label not in args.only:
            continue
        video = args.corpus / filename
        if not video.is_file():
            print(f"[{label}] MISSING {video}", flush=True)
            summary.append({"video": label, "state": "missing_source"})
            continue

        run_dir = base / label
        calib = ROOT / calibration
        log = base / f"{label}.log"
        log.write_text("", encoding="utf-8")
        print(f"\n[{label}] {filename[:56]}  "
              f"({video.stat().st_size / 1048576:.0f} MB)", flush=True)
        if not calib.is_file():
            print(f"    NO CALIBRATION at {calib}; skipping", flush=True)
            summary.append({"video": label, "state": "no_calibration"})
            continue

        timings = {}
        # Run ids carry the folder segment, so preflight builds the skeleton
        # directly at artifacts/runs/<tag>/<label> and every later stage opens
        # the same path.
        run("Phase 0  preflight + run manifest",
            ["tools/preflight_rtx3090.py", "--experiment",
             "configs/experiments/product_zero.yaml", "--sources", str(video),
             "--run-id", f"{args.tag}/{label}"], log, env)

        _c, timings["ingest"] = run(
            "Phase 1  ingest inventory + contact sheets",
            ["tools/validate_video.py", "--run", str(run_dir), str(video)],
            log, env)
        _c, timings["preprocessing"] = run(
            "Phase 1  preprocessing A/B (4 named transforms)",
            ["tools/run_preprocessing_ab.py", "--run", str(run_dir), str(video)],
            log, env)
        _c, timings["alignment"] = run(
            "Phase 2  alignment A/B (identity/features/ECC)",
            ["tools/run_alignment.py", "--run", str(run_dir),
             "--calibration", str(calib), str(video)], log, env)
        code, timings["motion"] = run(
            "Phases 2-3  health, motion, ROI, segmentation",
            ["tools/run_motion_scan.py", "--run", str(run_dir),
             "--calibration", str(calib), str(video)], log, env)
        if code != 0:
            summary.append({"video": label, "state": "motion_failed"})
            continue

        _c, timings["pose"] = run(
            f"Phase 4  tracking + pose A/B at {args.sample_hz} Hz "
            f"on {args.pose_device}",
            ["tools/run_pose_ab.py", "--run", str(run_dir),
             "--calibration", str(calib), "--person-model", str(args.dfine),
             "--pose", "--pose-device", args.pose_device,
             "--sample-hz", str(args.sample_hz),
             "--max-events", str(args.max_events), str(video)], log, env)
        run("Phase 4  pose comparison report",
            ["tools/render_pose_comparison.py", "--run", str(run_dir)],
            log, env)

        # The new stage: head and neck orientation from AlphaPose keypoints.
        _c, timings["head"] = run(
            "Phase 4b head/neck orientation + excursions + overlays",
            ["tools/run_head_analysis.py", str(video), "--run", str(run_dir),
             "--calibration", str(calib), "--overlays", "24"], log, env)

        _c, timings["detector"] = run(
            "Phase 5  crops + D-FINE native/SAHI",
            ["tools/run_detector_ab.py", "--run", str(run_dir),
             "--calibration", str(calib), "--dfine", str(args.dfine),
             "--no-rfdetr", str(video)], log, env)
        run("Phase 5  detector comparison report",
            ["tools/render_detector_comparison.py", "--run", str(run_dir)],
            log, env)
        for cls in ("phone", "secondary_paper_chit", "stationery"):
            run(f"Phase 5  render {cls} detections",
                ["tools/render_detections.py", str(video), "--run", str(run_dir),
                 "--config", "dfine_sahi", "--class", cls, "--top", "12"],
                log, env)

        _c, timings["evidence"] = run(
            "Phase 6  evidence packets + ranking + dispositions",
            ["tools/render_evidence_packet.py", str(video), "--run",
             str(run_dir), "--calibration", str(calib)], log, env)
        run("Phase 7  evaluation + configuration comparison",
            ["tools/evaluate_run.py", "--run", str(run_dir)], log, env)
        run("Final report", ["tools/generate_final_report.py", "--run",
                             str(run_dir)], log, env)

        summary.append({"video": label, "source": filename,
                        "run_dir": str(run_dir.relative_to(ROOT)),
                        "calibration": calibration,
                        "sample_hz": args.sample_hz,
                        "state": "complete",
                        "seconds": {k: round(v, 1) for k, v in timings.items()}})

    out = base / "RUN_1446_SUMMARY.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary written {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
