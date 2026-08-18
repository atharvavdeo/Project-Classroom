"""Preflight for one run (PRD 3, 5, 16 Phase 0).

    python tools/preflight_rtx3090.py --experiment configs/experiments/product_zero.yaml
    python tools/preflight_rtx3090.py --experiment ... --sources "C:/DrishtiAI/*.mkv"

Writes `RUN_MANIFEST.json` and an initial `RUN_STATUS.json` into a fresh run
folder, then reports what can and cannot run. It does not start the pipeline.

**It checks launchability, not weight existence.** That distinction is the whole
point of this file, and it is not theoretical. The shipped `llama-server.exe` is
a 9 KB stub that loads its DLLs from its own directory; it passed an existence
check and then failed `CreateProcess` with "the system cannot find the file
specified", naming a file that demonstrably existed. That killed a previous run
at its last stage after five minutes of CV work. So every configuration declares
*how to prove it starts*, and preflight runs that proof.

Nothing here substitutes. PRD 5: if a model cannot load while retaining the
configured VRAM safety margin, it is marked `resource_unavailable` and stays in
the comparison table as unavailable. A configuration is never quietly swapped
for a smaller one, and a resolution is never quietly lowered.

Exit codes
----------
    0   every named configuration is available
    1   at least one configuration is unavailable; the run may still proceed
        for the stages that do not need it
    2   a blocking precondition failed and no run folder was created
"""
from __future__ import annotations

import argparse
import glob as globlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts  # noqa: E402
from pipeline.config_validation import (  # noqa: E402
    PROBE_NONE, PROBE_ONNX_SESSION, PROBE_PYTHON_IMPORT,
    PROBE_SUBPROCESS_VERSION, RESIDENCY_CONCURRENT, ConfigError, ModelConfig,
    load_experiment_config, load_model_dir, missing_comparison_families, resolve,
)
from pipeline.run_manifest import (  # noqa: E402
    AVAILABLE, LAUNCH_FAILED, LICENCE_BLOCKED, MISSING_WEIGHTS, PENDING,
    RESOURCE_UNAVAILABLE, ConfigRecord, EnvironmentRecord, ModelRecord,
    RunManifest, RunStatus, StageStatus, code_revision, probe_environment,
    source_record,
)

# A licence that has not been confirmed cannot feed a publishable comparison
# report, so the configuration is blocked rather than run (PRD 3, PRD 5).
UNCONFIRMED_LICENCES = {"", "UNCONFIRMED", "unknown", "TBD", "tbd"}

# Which execution device a configuration ended up on. Recorded per model,
# because PRD 5 requires latency and VRAM per configuration and a CPU figure is
# not comparable to a GPU one.
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"


def expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = sorted(globlib.glob(pattern))
        out.extend(Path(m) for m in matches) if matches else out.append(Path(pattern))
    return out


def probe_model(config: ModelConfig, env: EnvironmentRecord,
                margin_gb: float) -> ModelRecord:
    """Decide whether one configuration can actually run, and record why not."""
    record = ModelRecord(
        config_id=config.config_id,
        family=config.family,
        state=AVAILABLE,
        licence=config.licence,
        licence_url=config.licence_url,
        input_size=config.input_size,
        precision=config.precision,
    )

    if not config.enabled:
        record.state = RESOURCE_UNAVAILABLE
        record.reason = "disabled in configuration"
        return record

    if (config.licence or "").strip() in UNCONFIRMED_LICENCES:
        record.state = LICENCE_BLOCKED
        record.reason = (
            "licence is unconfirmed; PRD 3 requires licence status recorded "
            "before a configuration may feed the selection report"
        )
        return record

    # Weights, when the configuration names them.
    if config.checkpoint:
        checkpoint = (ROOT / config.checkpoint) if not Path(config.checkpoint).is_absolute() \
            else Path(config.checkpoint)
        record.checkpoint_path = str(checkpoint)
        if not checkpoint.is_file():
            record.state = MISSING_WEIGHTS
            record.reason = f"checkpoint not found: {checkpoint}"
            return record
        record.checkpoint_sha256 = artifacts.sha256_file(checkpoint)
        if config.checkpoint_sha256 and record.checkpoint_sha256 != config.checkpoint_sha256:
            record.state = MISSING_WEIGHTS
            record.reason = (
                f"checkpoint hash mismatch: expected {config.checkpoint_sha256}, "
                f"found {record.checkpoint_sha256}")
            return record

    # Device selection and the VRAM safety margin.
    #
    # A configuration that *requires* a GPU and cannot have one is unavailable,
    # and one that cannot fit beside the margin is unavailable. Neither is ever
    # run at a reduced resolution instead (PRD 5).
    #
    # A configuration that does not require a GPU falls back to CPU and says so.
    # That is not a substitution: it is the same checkpoint, the same input size
    # and the same thresholds, on a slower device, with the device recorded on
    # every timing it produces.
    if env.gpu_usable:
        budget = (env.gpu_vram_total_gb or 0.0) - margin_gb
        if config.vram_gb > budget:
            if config.requires_gpu:
                record.state = RESOURCE_UNAVAILABLE
                record.reason = (
                    f"needs about {config.vram_gb:.1f} GB but only "
                    f"{budget:.1f} GB is available after the {margin_gb:.1f} GB "
                    f"safety margin")
                return record
            record.device = DEVICE_CPU
            record.device_reason = (
                f"{config.vram_gb:.1f} GB needed, {budget:.1f} GB available "
                f"after the safety margin")
        else:
            record.device = DEVICE_CUDA
    elif config.requires_gpu:
        record.state = RESOURCE_UNAVAILABLE
        record.reason = (
            "requires a GPU and no CUDA-capable device is usable from this "
            "process"
            + (f" (nvidia-smi reports {env.gpu_name}, but "
               f"torch.cuda.is_available() is False)" if env.gpu_name else "")
        )
        return record
    else:
        record.device = DEVICE_CPU
        record.device_reason = (
            "no CUDA-capable device usable from this process"
            + (f"; nvidia-smi reports {env.gpu_name} but "
               f"torch.cuda.is_available() is False" if env.gpu_name else "")
        )

    # Launchability.
    if config.probe == PROBE_NONE:
        record.launch_probe = "none required"
    elif config.probe == PROBE_PYTHON_IMPORT:
        assert config.module
        try:
            __import__(config.module)
            record.launch_probe = f"import {config.module}"
        except Exception as exc:  # noqa: BLE001
            record.state = LAUNCH_FAILED
            record.reason = f"import {config.module} failed: {type(exc).__name__}: {exc}"
            return record
    elif config.probe == PROBE_ONNX_SESSION:
        assert config.checkpoint
        try:
            import onnxruntime
            session = onnxruntime.InferenceSession(
                record.checkpoint_path, providers=["CPUExecutionProvider"])
            inputs = ", ".join(i.name for i in session.get_inputs())
            record.launch_probe = f"onnxruntime session created (inputs: {inputs})"
            del session
        except Exception as exc:  # noqa: BLE001
            record.state = LAUNCH_FAILED
            record.reason = f"onnxruntime session failed: {type(exc).__name__}: {exc}"
            return record
    elif config.probe == PROBE_SUBPROCESS_VERSION:
        assert config.binary
        binary = (ROOT / config.binary) if not Path(config.binary).is_absolute() \
            else Path(config.binary)
        if not binary.is_file():
            record.state = MISSING_WEIGHTS
            record.reason = f"binary not found: {binary}"
            return record
        try:
            # Run it from its own directory. The stub loads sibling DLLs, and
            # launching from anywhere else reports the executable as missing.
            proc = subprocess.run(
                [str(binary), *config.binary_args], capture_output=True,
                text=True, timeout=120, cwd=str(binary.parent))
            if proc.returncode != 0:
                record.state = LAUNCH_FAILED
                record.reason = (
                    f"exit {proc.returncode}: "
                    f"{(proc.stderr or proc.stdout).strip()[:200]}")
                return record
            first = (proc.stdout or proc.stderr).strip().splitlines()
            record.launch_probe = first[0][:120] if first else "started"
        except Exception as exc:  # noqa: BLE001
            record.state = LAUNCH_FAILED
            record.reason = f"{type(exc).__name__}: {exc}"
            return record

    return record


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=ROOT / "configs" / "models")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="override the source videos named by the experiment")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--parent-run-id", default=None)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    args = ap.parse_args()

    try:
        experiment = load_experiment_config(args.experiment)
        models = load_model_dir(args.models)
        chosen = resolve(experiment, models)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    source_patterns = args.sources if args.sources is not None else experiment.sources
    sources = expand(source_patterns) if source_patterns else []
    missing_sources = [str(p) for p in sources if not p.is_file()]
    if source_patterns and missing_sources:
        print(f"source video not found: {', '.join(missing_sources)}", file=sys.stderr)
        return 2

    artifact_root = ROOT / experiment.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    env = probe_environment(artifact_root)

    print(f"experiment   {experiment.experiment_id}  ({experiment.config_hash[:12]})")
    print(f"residency    {experiment.residency}, "
          f"{experiment.vram_safety_margin_gb:.1f} GB safety margin")
    print(f"python       {env.python} on {env.platform}")
    print(f"gpu          {env.gpu_name or 'none detected'}"
          + (f", {env.gpu_vram_total_gb:.0f} GB" if env.gpu_vram_total_gb else "")
          + (f", driver {env.driver_version}" if env.driver_version else ""))
    print(f"torch        {env.torch_version or 'absent'}, "
          f"cuda {'available' if env.torch_cuda_available else 'NOT available'}")
    print(f"onnxruntime  {', '.join(env.onnxruntime_providers) or 'absent'}")
    print(f"free disk    {env.free_disk_gb:.1f} GB at {artifact_root}")

    blocking: list[str] = []
    if env.free_disk_gb is not None and env.free_disk_gb < args.min_free_gb:
        blocking.append(
            f"only {env.free_disk_gb:.1f} GB free at {artifact_root}, "
            f"below the {args.min_free_gb:.1f} GB minimum")
    if experiment.residency == RESIDENCY_CONCURRENT and env.gpu_vram_total_gb \
            and env.gpu_vram_total_gb < 30:
        blocking.append(
            f"concurrent residency needs a 32 GB class card; this machine has "
            f"{env.gpu_vram_total_gb:.0f} GB. Use residency: sequential.")
    if blocking:
        for reason in blocking:
            print(f"\nBLOCKED: {reason}", file=sys.stderr)
        return 2

    print("\nconfigurations")
    records = [probe_model(c, env, experiment.vram_safety_margin_gb) for c in chosen]
    width = max(len(r.config_id) for r in records)
    for record in records:
        mark = "OK  " if record.state == AVAILABLE else "--  "
        detail = record.launch_probe if record.state == AVAILABLE else record.reason
        print(f"  {mark}{record.config_id:<{width}}  {record.family:<17} "
              f"{record.state:<21} {detail or ''}")

    unavailable = [r for r in records if r.state != AVAILABLE]
    families_blocked = missing_comparison_families(
        [c for c, r in zip(chosen, records) if r.state == AVAILABLE])

    run_id = args.run_id or artifacts.new_run_id(experiment.experiment_id)
    paths = artifacts.create_run(artifact_root, run_id)
    revision, dirty = code_revision(ROOT)

    manifest = RunManifest(
        run_id=run_id,
        created_utc=artifacts.utc_now(),
        seed=experiment.seed,
        code_revision=revision,
        code_dirty=dirty,
        parent_run_id=args.parent_run_id,
        sources=[source_record(p) for p in sources],
        configs=[ConfigRecord(experiment.experiment_id, str(args.experiment),
                              experiment.config_hash, "experiment")]
        + [ConfigRecord(c.config_id, c.source_path, c.config_hash, c.family)
           for c in chosen],
        models=records,
        environment=env,
        calibration_version=experiment.calibration,
    )

    if not experiment.reference_manifest:
        manifest.notes.append(
            "No reference manifest configured. Per PRD 3 the run proceeds and "
            "object evaluation is marked incomplete; no quality claim may be "
            "made. Stages 09 selection and 12 evaluation are blocked.")
    if not experiment.taxonomy:
        manifest.notes.append(
            "No approved taxonomy configured. PRD 3 blocks detector selection "
            "and evaluation until the product owner approves the class list.")
    if dirty:
        manifest.notes.append(
            "Working tree has uncommitted changes; this run is not exactly "
            "reproducible from its recorded revision.")
    for family in families_blocked:
        manifest.notes.append(
            f"Family {family!r} has fewer than two available configurations, so "
            f"it cannot support a comparison and no winner may be named.")

    manifest.write(paths)
    status = RunStatus(paths, run_id)
    for stage in experiment.stages or [name for name, _ in artifacts.STAGES]:
        status.stages[stage] = StageStatus(stage=stage, state=PENDING)
    status.write()

    print(f"\nrun folder   {paths.dir}")
    print(f"manifest     {len(manifest.available_models)}/{len(records)} "
          f"configurations available")
    for note in manifest.notes:
        print(f"  note: {note}")

    if unavailable:
        print(f"\n{len(unavailable)} configuration(s) unavailable. They remain in "
              f"the comparison table as unavailable and are never substituted "
              f"(PRD 5).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
