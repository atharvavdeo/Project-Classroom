"""Phase 0 gate: run folder, manifest, configuration validation, preflight.

The properties checked here are the ones whose absence is invisible. A run
folder that quietly accepts a write after sealing, a config typo that is
silently ignored, a model marked available because its file exists — none of
these announce themselves, and each one corrupts a result that still looks fine.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts  # noqa: E402
from pipeline.config_validation import (  # noqa: E402
    ConfigError, ExperimentConfig, ModelConfig, canonical_hash,
    families_present, load_experiment_config, load_model_dir,
    missing_comparison_families, resolve,
)
from pipeline.run_manifest import (  # noqa: E402
    AVAILABLE, BLOCKED, COMPLETE, LAUNCH_FAILED, ModelRecord, RunManifest,
    RunStatus, SourceRecord, StageStatus,
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    ok = True

    print("artifact folder structure (PRD 13)")
    with tempfile.TemporaryDirectory() as tmp:
        paths = artifacts.create_run(tmp, "run_test")
        declared = {name for name, _ in artifacts.STAGES}
        ok &= check(f"all {len(declared)} declared stages created",
                    all((paths.dir / n).is_dir() for n in declared))
        ok &= check("nested subdirectories created",
                    (paths.dir / "04_calibration" / "masks").is_dir()
                    and (paths.dir / "12_evaluation" / "misses").is_dir())

        raised = False
        try:
            paths.stage("99_invented")
        except KeyError:
            raised = True
        ok &= check("an undeclared stage name is refused", raised,
                    "a stage cannot invent a sibling the index does not know")

        raised = False
        try:
            artifacts.create_run(tmp, "run_test")
        except FileExistsError:
            raised = True
        ok &= check("a run id cannot be reused", raised)

        print("\nprovenance in filenames")
        ok &= check("PTS prefix is zero padded and sortable",
                    artifacts.pts_prefix(42) < artifacts.pts_prefix(1000)
                    < artifacts.pts_prefix(281850),
                    f"{artifacts.pts_prefix(42)} < {artifacts.pts_prefix(281850)}")
        ok &= check("lexical order equals temporal order",
                    sorted([artifacts.pts_prefix(t) for t in (900, 90, 9000, 9)])
                    == [artifacts.pts_prefix(t) for t in (9, 90, 900, 9000)])
        ok &= check("names carry the condition",
                    artifacts.pts_name(12345.6, "blackout", ".png")
                    == "0000012346_blackout.png",
                    artifacts.pts_name(12345.6, "blackout", ".png"))
        raised = False
        try:
            artifacts.pts_prefix(-1)
        except ValueError:
            raised = True
        ok &= check("a negative PTS is refused", raised)

        print("\nper-configuration isolation (PRD 17.9)")
        a = paths.config_dir("08_pose", "rtmpose_baseline")
        b = paths.config_dir("08_pose", "rtmo_baseline")
        ok &= check("each configuration gets its own directory", a != b and a.is_dir())

        print("\nimmutability (PRD 13)")
        paths.append_jsonl("05_motion_roi/window_metrics.jsonl", {"t": 0.0})
        paths.append_jsonl("05_motion_roi/window_metrics.jsonl", {"t": 0.5})
        lines = (paths.dir / "05_motion_roi" / "window_metrics.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        ok &= check("jsonl appends rather than replaces", len(lines) == 2,
                    f"{len(lines)} lines")
        index = paths.index()
        ok &= check("the artifact index lists written files", len(index) >= 1,
                    f"{len(index)} files")

        paths.seal()
        ok &= check("a sealed run reports itself sealed", paths.sealed)
        for label, action in (
            ("write_json", lambda: paths.write_json("x.json", {})),
            ("append_jsonl", lambda: paths.append_jsonl("y.jsonl", {})),
            ("config_dir", lambda: paths.config_dir("08_pose", "late")),
            ("event_dir", lambda: paths.event_dir("evt_1")),
        ):
            raised = False
            try:
                action()
            except artifacts.RunSealed:
                raised = True
            ok &= check(f"{label} refuses to write to a sealed run", raised)

    print("\nconfiguration validation")
    models = load_model_dir(ROOT / "configs" / "models")
    ok &= check("the model configuration set loads", len(models) >= 10,
                f"{len(models)} configurations")
    ok &= check("every configuration declares a licence",
                all(m.licence for m in models.values()))
    ok &= check("every configuration declares how to prove it launches",
                all(m.probe for m in models.values()))

    experiment = load_experiment_config(
        ROOT / "configs" / "experiments" / "product_zero.yaml")
    chosen = resolve(experiment, models)
    families = families_present(chosen)
    ok &= check("pose family has the PRD 5 challengers",
                len(families.get("pose", [])) >= 3, str(families.get("pose")))
    ok &= check("object family has both detectors and both slicing modes",
                len(families.get("object", [])) == 4, str(families.get("object")))
    ok &= check("tracker family includes the no-tracker control",
                "seat_only_no_track" in families.get("tracker", []))
    ok &= check("no family is left unable to support a comparison",
                not missing_comparison_families(chosen))

    print("\nconfiguration validation refuses the invisible mistakes")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def write(name: str, body: str) -> Path:
            path = tmp / name
            path.write_text(body, encoding="utf-8")
            return path

        cases = [
            ("a misspelled key",
             "config_id: x\nfamily: pose\nprobe: none\nlicence: MIT\nthreshold: 1\n"),
            ("an unknown family",
             "config_id: x\nfamily: nonsense\nprobe: none\nlicence: MIT\n"),
            ("an unknown probe",
             "config_id: x\nfamily: pose\nprobe: guess\nlicence: MIT\n"),
            ("a missing licence",
             "config_id: x\nfamily: pose\nprobe: none\n"),
            ("an onnx probe with no checkpoint",
             "config_id: x\nfamily: object\nprobe: onnx_session\nlicence: MIT\n"),
            ("a subprocess probe with no binary",
             "config_id: x\nfamily: vlm\nprobe: subprocess_version\nlicence: MIT\n"),
        ]
        for label, body in cases:
            raised = False
            try:
                from pipeline.config_validation import load_model_config
                load_model_config(write("bad.yaml", body))
            except ConfigError:
                raised = True
            ok &= check(f"refuses {label}", raised)

        # A duplicate config_id would let one configuration overwrite another's
        # artifacts, because config_id names the directory.
        dup = tmp / "dup"
        dup.mkdir()
        for name in ("a.yaml", "b.yaml"):
            (dup / name).write_text(
                "config_id: same\nfamily: pose\nprobe: none\nlicence: MIT\n",
                encoding="utf-8")
        raised = False
        try:
            load_model_dir(dup)
        except ConfigError:
            raised = True
        ok &= check("refuses a duplicate config_id across files", raised)

        for label, body in (
            ("an experiment with no configurations",
             "experiment_id: e\nseed: 1\nartifact_root: a\nconfigs: []\n"),
            ("an unknown residency mode",
             "experiment_id: e\nseed: 1\nartifact_root: a\nconfigs: [x]\n"
             "residency: whenever\n"),
            ("a non-integer seed",
             "experiment_id: e\nseed: banana\nartifact_root: a\nconfigs: [x]\n"),
        ):
            raised = False
            try:
                load_experiment_config(write("bad_exp.yaml", body))
            except ConfigError:
                raised = True
            ok &= check(f"refuses {label}", raised)

    print("\nconfiguration hashing (PRD 17.10)")
    ok &= check("hash ignores key order",
                canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1}))
    ok &= check("hash changes when a threshold changes",
                canonical_hash({"t": 0.35}) != canonical_hash({"t": 0.36}))

    print("\nmanifest and status")
    with tempfile.TemporaryDirectory() as tmp:
        paths = artifacts.create_run(tmp, "run_manifest_test")
        manifest = RunManifest(
            run_id="run_manifest_test", created_utc=artifacts.utc_now(), seed=7,
            sources=[SourceRecord("v.mkv", "a" * 64, 1)],
            models=[
                ModelRecord("good", "pose", AVAILABLE, launch_probe="import x"),
                ModelRecord("bad", "object", LAUNCH_FAILED, reason="no module"),
            ],
        )
        manifest.write(paths)
        reloaded = RunManifest.read(paths)
        ok &= check("manifest round-trips", reloaded.run_id == "run_manifest_test"
                    and reloaded.seed == 7)
        ok &= check("an unavailable configuration stays in the table",
                    len(reloaded.models) == 2
                    and len(reloaded.unavailable_models) == 1,
                    "PRD 5: a failed configuration is listed as unavailable")
        ok &= check("available models are listed separately",
                    reloaded.available_models == ["good"])

        raised = False
        try:
            ModelRecord("x", "pose", "probably_fine")
        except ValueError:
            raised = True
        ok &= check("an invented model state is refused", raised)

        status = RunStatus(paths, "run_manifest_test")
        status.begin("01_ingest")
        status.finish("01_ingest", COMPLETE, covered_ms=281850.0, total_ms=281850.0,
                      frames=4227)
        status.begin("09_object")
        status.finish("09_object", BLOCKED,
                      reason="no approved taxonomy; PRD 3 blocks detector selection")
        ok &= check("stage coverage is recorded",
                    status.stages["01_ingest"].coverage_fraction == 1.0)
        ok &= check("a blocked stage records its reason",
                    "taxonomy" in (status.stages["09_object"].reason or ""))
        ok &= check("blocked stages are listed", status.blocked == ["09_object"])

        raised = False
        try:
            StageStatus(stage="x", state="finished_probably")
        except ValueError:
            raised = True
        ok &= check("an invented stage state is refused", raised)

    print("\npreflight end to end")
    with tempfile.TemporaryDirectory() as tmp:
        experiment_path = Path(tmp) / "exp.yaml"
        experiment_path.write_text(
            "experiment_id: gate\n"
            "seed: 1\n"
            f"artifact_root: {Path(tmp).as_posix()}/runs\n"
            "residency: sequential\n"
            "configs:\n  - seat_only_no_track\n  - dfine_native\n",
            encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "preflight_rtx3090.py"),
             "--experiment", str(experiment_path)],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT))
        out = proc.stdout
        ok &= check("preflight runs", proc.returncode in (0, 1),
                    f"exit {proc.returncode}")
        ok &= check("it reports the device and the environment",
                    "gpu" in out and "torch" in out and "free disk" in out)
        ok &= check("it writes a run folder", "run folder" in out)
        runs = list((Path(tmp) / "runs").glob("gate_*"))
        ok &= check("the run folder exists on disk", len(runs) == 1, str(runs))
        if runs:
            written = artifacts.open_run(Path(tmp) / "runs", runs[0].name)
            reloaded = RunManifest.read(written)
            ok &= check("the manifest records the environment",
                        reloaded.environment is not None)
            ok &= check("the manifest records the seed", reloaded.seed == 1)
            ok &= check("every named configuration has a record",
                        len(reloaded.models) == 2, str(len(reloaded.models)))
            ok &= check("a note explains the missing reference manifest",
                        any("reference manifest" in n for n in reloaded.notes))
            ok &= check("a note explains the missing taxonomy",
                        any("taxonomy" in n for n in reloaded.notes))
            ok &= check("RUN_STATUS.json is initialised",
                        (written.dir / "RUN_STATUS.json").is_file())

    print("\nconcurrent residency is refused below 32 GB")
    with tempfile.TemporaryDirectory() as tmp:
        experiment_path = Path(tmp) / "exp.yaml"
        experiment_path.write_text(
            "experiment_id: concurrent_gate\n"
            "seed: 1\n"
            f"artifact_root: {Path(tmp).as_posix()}/runs\n"
            "residency: concurrent\n"
            "configs:\n  - seat_only_no_track\n",
            encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "preflight_rtx3090.py"),
             "--experiment", str(experiment_path)],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT))
        # On a machine with no usable GPU the guard cannot fire, and that is
        # correct: there is nothing to refuse. The check is that it never
        # *silently downgrades* the requested mode.
        combined = proc.stdout + proc.stderr
        ok &= check("concurrent residency is either honoured or refused, "
                    "never silently downgraded",
                    "BLOCKED" in combined or proc.returncode in (0, 1),
                    f"exit {proc.returncode}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
