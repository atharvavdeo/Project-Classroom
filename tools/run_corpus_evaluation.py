"""Evaluate the whole corpus, one video and one model at a time (PRD 5).

    python tools/run_corpus_evaluation.py --corpus C:/DrishtiAI

PRD 5's resource rule is the reason this is a driver rather than a loop over
threads:

> Heavy models must run sequentially, never all resident together.

That rule was written for a 24 GB RTX 3090. This machine has a 4 GB RTX 3050
laptop GPU and a CPU-only torch build, so sequential execution is not a
preference here, it is the only thing that fits. Each video gets its own run
folder; within a run, each stage tool is a separate process, so a model's memory
is released by process exit before the next one starts.

Each video is paired with its own camera calibration. Video 03 has a reviewed
profile; the rest carry DRAFT profiles proposed by `tools/propose_calibration.py`
from real person detections, which means their seat attribution is blocked
(PRD 3) while motion, pose and object evidence still run.
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

# Video prefix -> calibration profile. 03 is the reviewed one.
CALIBRATION = {
    "01": "calib/cam01_e1_v1.json",
    "02": "calib/cam02_e1_v1.json",
    "03": "calib/cam12_e1_v1.json",
    "04": "calib/cam04_e1_v1.json",
    "05": "calib/cam05_e1_v1.json",
    "06": "calib/cam06_e1_v1.json",
}


def run(label: str, argv: list[str], log: Path, env: dict) -> tuple[int, float]:
    started = time.perf_counter()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{'=' * 70}\n{label}\n{'=' * 70}\n")
        handle.flush()
        completed = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - started
    print(f"    {label:44} exit {completed.returncode:2}  {seconds:7.1f}s")
    return completed.returncode, seconds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("C:/DrishtiAI"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="video prefixes to run, e.g. --only 01 04")
    ap.add_argument("--dfine", type=Path,
                    default=ROOT / "models/dfine/onnx/model.onnx")
    ap.add_argument("--limit-events", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--sample-hz", type=float, default=1.0)
    ap.add_argument("--gemma-url", default=None)
    ap.add_argument("--tag", default="corpus")
    args = ap.parse_args()

    import os
    env = dict(os.environ)
    # Measured: numpy and torch each ship libiomp5md.dll and the second load
    # aborts the process. Preflight already orders its imports to avoid it; the
    # AlphaPose stack imports torch first, so the override is needed here.
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONWARNINGS"] = "ignore"

    videos = sorted(p for p in args.corpus.glob("0*.m*")
                    if p.suffix.lower() in (".mkv", ".mp4"))
    if args.only:
        videos = [v for v in videos if v.name[:2] in args.only]
    if not videos:
        print(f"no videos found under {args.corpus}", file=sys.stderr)
        return 2

    print(f"corpus: {len(videos)} video(s)\n")
    summary = []

    for video in videos:
        prefix = video.name[:2]
        calibration = ROOT / CALIBRATION.get(prefix, "")
        run_id = f"{args.tag}_{prefix}"
        run_dir = ROOT / "artifacts/runs" / run_id
        log = ROOT / "artifacts/logs" / f"{run_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("", encoding="utf-8")

        size_mb = video.stat().st_size / 1048576
        print(f"[{prefix}] {video.name[:58]}  ({size_mb:.0f} MB)")
        if not calibration.is_file():
            print(f"    NO CALIBRATION for camera {prefix}; skipping. PRD 3 "
                  f"blocks seat-attributed evidence without one.")
            summary.append({"video": prefix, "state": "no_calibration"})
            continue

        timings = {}
        # Phase 0 -- preflight. Non-zero simply means some configuration is
        # unavailable, which is recorded, not fatal.
        code, timings["preflight"] = run(
            "Phase 0  preflight + run manifest",
            ["tools/preflight_rtx3090.py", "--experiment",
             "configs/experiments/product_zero.yaml", "--sources", str(video),
             "--run-id", run_id], log, env)

        # Phases 1-3 -- whole video, no neural model.
        code, timings["motion"] = run(
            "Phases 1-3  health, motion, ROI, segmentation",
            ["tools/run_motion_scan.py", "--run", str(run_dir),
             "--calibration", str(calibration), str(video)], log, env)
        if code != 0:
            summary.append({"video": prefix, "state": "motion_failed"})
            continue

        # Phase 4 -- person detection, three trackers, then RTMPose, RTMO and
        # AlphaPose one after another on identical manifest rows.
        argv = ["tools/run_pose_ab.py", "--run", str(run_dir),
                "--calibration", str(calibration), "--person-model",
                str(args.dfine), "--pose", "--sample-hz", str(args.sample_hz),
                str(video)]
        if args.limit_events:
            argv += ["--limit-events", str(args.limit_events)]
        if args.max_events:
            argv += ["--max-events", str(args.max_events)]
        code, timings["pose"] = run("Phase 4  tracking + pose A/B (3 models)",
                                    argv, log, env)
        run("Phase 4  pose comparison report",
            ["tools/render_pose_comparison.py", "--run", str(run_dir)], log, env)

        # Phase 5 -- crops, then each detector configuration in turn.
        code, timings["detector"] = run(
            "Phase 5  crops + detector/SAHI A/B",
            ["tools/run_detector_ab.py", "--run", str(run_dir),
             "--calibration", str(calibration), "--dfine", str(args.dfine),
             str(video)], log, env)
        run("Phase 5  detector comparison report",
            ["tools/render_detector_comparison.py", "--run", str(run_dir)],
            log, env)

        # Phase 6 -- evidence packets, optional Gemma, ranking.
        argv = ["tools/render_evidence_packet.py", "--run", str(run_dir),
                "--calibration", str(calibration), str(video)]
        if args.gemma_url:
            argv += ["--gemma-url", args.gemma_url]
        code, timings["evidence"] = run("Phase 6  evidence packets + ranking",
                                        argv, log, env)

        # Phase 7 -- isolation check and evaluation. No reference manifest
        # exists for this corpus, so this records the run as incomplete rather
        # than producing an accuracy number.
        code, timings["evaluate"] = run(
            "Phase 7  isolation check + evaluation",
            ["tools/evaluate_run.py", "--run", str(run_dir)], log, env)
        run("Phase 7  configuration comparison",
            ["tools/compare_configurations.py", "--run", str(run_dir)], log, env)
        run("Final report",
            ["tools/generate_final_report.py", "--run", str(run_dir)], log, env)

        summary.append({"video": prefix, "name": video.name,
                        "run_id": run_id, "state": "complete",
                        "calibration": CALIBRATION.get(prefix),
                        "seconds": {k: round(v, 1) for k, v in timings.items()}})
        print()

    out = ROOT / "artifacts" / f"{args.tag}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
