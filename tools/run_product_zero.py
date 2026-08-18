"""The one documented command (PRD 20).

    python tools/run_product_zero.py \
        --calibration calib/cam12_e1_v1.json \
        --experiment configs/experiments/product_zero.yaml \
        --dfine models/dfine/onnx/model.onnx \
        --pose \
        "C:/DrishtiAI/03.CCTV Mobile Usage.mkv"

PRD 20's definition of done:

> The work is complete only when the RTX 3090 owner can run one documented
> command against a full source video and obtain the immutable run folder,
> complete-duration accounting, artifacts for every preprocessing/model stage,
> A/B comparison reports, gate audit, reference-frame evaluation, and an honest
> recommendation.

This is that command. It runs the seven stage tools in the order PRD 4 mandates
and does not reorder or skip them, because the order is what makes the
comparisons attributable.

**It does not stop on an unavailable model.** A configuration that cannot run is
recorded as unavailable and the run continues, so a machine with no GPU, no
RF-DETR and no llama-server still produces a complete run folder that says
exactly which configurations were absent and why. It stops only on a *blocking*
failure — an unreadable video, a calibration with blocking issues, a reference
leak — because those invalidate what follows rather than narrowing it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Exit codes that mean "this stage found a real problem", as opposed to
# "something was unavailable". Preflight returns non-zero when a configuration
# is unavailable, which is information, not failure.
BLOCKING = 2


def run(label: str, argv: list[str], blocking_codes=(BLOCKING,)) -> int:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, *argv], cwd=ROOT)
    seconds = time.perf_counter() - started
    print(f"[{label}: exit {completed.returncode} in {seconds:.1f}s]")
    if completed.returncode in blocking_codes:
        print(f"\n{label} reported a blocking condition. Stopping: everything "
              f"after this would be measuring an input that is not valid.",
              file=sys.stderr)
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--experiment", type=Path,
                    default=ROOT / "configs/experiments/product_zero.yaml")
    ap.add_argument("--run-id", default=None,
                    help="defaults to a timestamped id under artifacts/runs/")
    ap.add_argument("--person-model", type=Path, default=None,
                    help="ONNX detector used as the person source")
    ap.add_argument("--dfine", type=Path, default=None,
                    help="D-FINE ONNX checkpoint for the object A/B")
    ap.add_argument("--rfdetr", type=Path, default=None)
    ap.add_argument("--pose", action="store_true",
                    help="run the pose configurations (rtmlib fetches weights "
                         "on first use)")
    ap.add_argument("--pose-device", default="cpu")
    ap.add_argument("--gemma-url", default=None,
                    help="a running llama-server serving Gemma")
    ap.add_argument("--reference", type=Path, default=None,
                    help="reference manifest, read only at stage 12")
    ap.add_argument("--taxonomy-approved", action="store_true")
    ap.add_argument("--limit-events", type=int, default=0)
    ap.add_argument("--seal", action="store_true",
                    help="seal the run folder when finished, making it immutable")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"no such video: {args.video}", file=sys.stderr)
        return BLOCKING

    run_id = args.run_id or f"pz_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir = ROOT / "artifacts" / "runs" / run_id
    video = str(args.video)
    person_model = args.person_model or args.dfine

    print(f"Product Zero — run {run_id}")
    print(f"source       {video}")
    print(f"calibration  {args.calibration}")

    # 0. preflight and the immutable run manifest. A non-zero exit here means
    #    configurations are unavailable, which is recorded, not fatal.
    run("Phase 0 — preflight and run manifest",
        ["tools/preflight_rtx3090.py", "--experiment", str(args.experiment),
         "--sources", video, "--run-id", run_id],
        blocking_codes=())

    # 1-3. ingest, health, calibration, motion, ROI, segmentation. No model.
    run("Phases 1-3 — health, motion, ROI, segmentation, gates",
        ["tools/run_motion_scan.py", "--run", str(run_dir),
         "--calibration", str(args.calibration), video])

    # 4. tracking and the pose A/B on identical manifest rows.
    pose_argv = ["tools/run_pose_ab.py", "--run", str(run_dir),
                 "--calibration", str(args.calibration), video]
    if person_model:
        pose_argv += ["--person-model", str(person_model)]
    if args.pose:
        pose_argv += ["--pose", "--pose-device", args.pose_device]
    if args.limit_events:
        pose_argv += ["--limit-events", str(args.limit_events)]
    run("Phase 4 — tracking and pose A/B", pose_argv)
    run("Phase 4 — pose comparison report",
        ["tools/render_pose_comparison.py", "--run", str(run_dir)],
        blocking_codes=())

    # 5. crops, detector and SAHI A/B, temporal confirmation.
    detector_argv = ["tools/run_detector_ab.py", "--run", str(run_dir),
                     "--calibration", str(args.calibration), video]
    if args.dfine:
        detector_argv += ["--dfine", str(args.dfine)]
    if args.rfdetr:
        detector_argv += ["--rfdetr", str(args.rfdetr)]
    run("Phase 5 — crops, detector and SAHI A/B", detector_argv)
    run("Phase 5 — detector comparison report",
        ["tools/render_detector_comparison.py", "--run", str(run_dir)],
        blocking_codes=())

    # 6. evidence packets, bounded Gemma review, ranking.
    evidence_argv = ["tools/render_evidence_packet.py", "--run", str(run_dir),
                     "--calibration", str(args.calibration), video]
    if args.gemma_url:
        evidence_argv += ["--gemma-url", args.gemma_url]
    run("Phase 6 — evidence packets, Gemma review, ranking", evidence_argv)

    # 7. the first and only stage permitted to read references.
    evaluate_argv = ["tools/evaluate_run.py", "--run", str(run_dir)]
    if args.reference:
        evaluate_argv += ["--reference", str(args.reference)]
    if args.taxonomy_approved:
        evaluate_argv += ["--taxonomy-approved"]
    run("Phase 7 — reference evaluation and gate audit", evaluate_argv)
    run("Phase 7 — configuration comparison",
        ["tools/compare_configurations.py", "--run", str(run_dir)],
        blocking_codes=())

    run("Final report",
        ["tools/generate_final_report.py", "--run", str(run_dir)],
        blocking_codes=())

    if args.seal:
        from pipeline import artifacts
        artifacts.open_run(run_dir.parent, run_dir.name).seal()
        print(f"\nrun sealed: {run_dir} is now immutable. Reprocessing creates "
              f"a new run id linked to this one as parent (PRD 13).")

    print(f"\n{'=' * 72}")
    print(f"run folder   {run_dir}")
    print(f"reports      {run_dir / '13_reports'}")
    for name in ("final_report.md", "model_selection_report.md",
                 "configuration_comparison.md", "quality_limitations.md",
                 "pose_comparison.md", "detector_comparison.md"):
        path = run_dir / "13_reports" / name
        print(f"  {'OK ' if path.is_file() else '-- '} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
