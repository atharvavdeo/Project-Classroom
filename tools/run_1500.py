"""Run 1500: continuous per-person pipeline with object evidence and flags.

    python tools/run_1500.py

Two recordings into `artifacts/runs/1500/<video>/`, everything on GPU:

  1. Seat No. 12 was seen taking a piece of paper from the desk.mkv
  2. 01.Candidate was found using a mobile phone in the examination hall..mkv

What differs from run 1446, and why:

**Every frame is considered.** 1446 sampled from the motion candidate manifest
and therefore looked at 20-39% of each recording; 60-80% was never decoded for
person detection at all. This run lays a uniform grid over the whole duration,
so a candidate who sits still and raises no motion event is still observed.

**Object evidence costs nothing extra.** D-FINE returns all 80 COCO classes
from one inference. The person pass used to filter to `person` and throw the
rest away, so every frame paid for a phone detection and deleted it. Those
detections are now kept and attached to whichever person's wrist they are
nearest, measured in torso widths.

**Flags are per person, not per event.** Each person gets a record whether or
not anything notable happened to them -- without the ordinary samples there is
no baseline to be unusual against. The thresholds are deliberately generous:
1446's guardrails were strict enough that **zero** of 113 ranked events reached
the top disposition band, and a triage tool that surfaces nothing is worse than
one that surfaces too much.

Per video the run folder holds the standard ingest, preprocessing, alignment
and motion stages, then `14_person_timeline/` with samples, profiles, flags,
annotated frames and a rendered annotated video.
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

VIDEOS = (
    ("12_paper", "Seat No. 12 was seen taking a piece of paper from the "
                 "desk.mkv", "calib/campaper_e1_v1.json"),
    ("01_phone", "01.Candidate was found using a mobile phone in the "
                 "examination hall..mkv", "calib/cam01_e1_v1.json"),
)


def run(label: str, argv: list[str], log: Path, env: dict) -> tuple[int, float]:
    started = time.perf_counter()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{'=' * 72}\n{label}\n{'=' * 72}\n")
        handle.flush()
        done = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env,
                              stdout=handle, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - started
    state = "ok " if done.returncode == 0 else f"EXIT {done.returncode}"
    print(f"    {label:52} {state:8} {seconds:7.1f}s", flush=True)
    return done.returncode, seconds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--tag", default="1500")
    ap.add_argument("--sample-hz", type=float, default=4.0)
    ap.add_argument("--pose-device", default="cuda")
    ap.add_argument("--annotate", type=int, default=60)
    ap.add_argument("--only", nargs="*", default=None)
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
        calib = ROOT / calibration
        run_dir = base / label
        log = base / f"{label}.log"
        log.write_text("", encoding="utf-8")

        if not video.is_file():
            print(f"[{label}] MISSING {video}", flush=True)
            summary.append({"video": label, "state": "missing_source"})
            continue
        print(f"\n[{label}] {filename[:58]}  "
              f"({video.stat().st_size / 1048576:.0f} MB)", flush=True)

        timings: dict = {}
        run("Phase 0  preflight + run manifest",
            ["tools/preflight_rtx3090.py", "--experiment",
             "configs/experiments/product_zero.yaml", "--sources", str(video),
             "--run-id", f"{args.tag}/{label}"], log, env)
        _c, timings["ingest"] = run(
            "Phase 1  ingest inventory",
            ["tools/validate_video.py", "--run", str(run_dir), str(video)],
            log, env)
        _c, timings["preprocessing"] = run(
            "Phase 1  preprocessing A/B",
            ["tools/run_preprocessing_ab.py", "--run", str(run_dir), str(video)],
            log, env)
        _c, timings["alignment"] = run(
            "Phase 2  alignment A/B",
            ["tools/run_alignment.py", "--run", str(run_dir),
             "--calibration", str(calib), str(video)], log, env)
        _c, timings["motion"] = run(
            "Phases 2-3  health, motion, ROI, segmentation",
            ["tools/run_motion_scan.py", "--run", str(run_dir),
             "--calibration", str(calib), str(video)], log, env)

        # The stage this run exists for.
        code, timings["timeline"] = run(
            f"Phase 14  continuous per-person timeline @{args.sample_hz} Hz "
            f"on {args.pose_device} (+objects +flags)",
            ["tools/run_person_timeline.py", str(video), "--run", str(run_dir),
             "--calibration", str(calib), "--person-model", str(args.dfine),
             "--sample-hz", str(args.sample_hz),
             "--pose-device", args.pose_device,
             "--annotate", str(args.annotate)], log, env)
        if code != 0:
            summary.append({"video": label, "state": "timeline_failed"})
            continue

        _c, timings["video"] = run(
            "Phase 14  render annotated video",
            ["tools/render_annotated_video.py", str(video), "--run",
             str(run_dir)], log, env)
        run("Findings report",
            ["tools/write_findings.py", "--run", str(run_dir)], log, env)

        entry = {"video": label, "source": filename,
                 "run_dir": str(run_dir.relative_to(ROOT)),
                 "sample_hz": args.sample_hz, "state": "complete",
                 "seconds": {k: round(v, 1) for k, v in timings.items()}}
        report = run_dir / "14_person_timeline/summary.json"
        if report.is_file():
            doc = json.loads(report.read_text(encoding="utf-8"))
            entry["people"] = doc.get("people")
            entry["flags"] = doc.get("flags")
        summary.append(entry)

    out = base / "RUN_1500_SUMMARY.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary written {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
