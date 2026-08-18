"""Configuration loading and validation (PRD 3, 5, 17.10).

Every threshold, model and slice setting is versioned and hashed, and the hash
is of the *resolved* content rather than the file bytes, so two configs that
differ only in comments or key order hash identically and two that differ in a
single threshold never do.

Three rules the loader enforces, each of which exists because the alternative
fails quietly:

  * **Unknown keys are refused.** A typo in a threshold name would otherwise be
    silently ignored and the default used, and the run would report a
    configuration it did not actually apply.

  * **A model configuration must declare its licence.** PRD 3 requires licence
    status recorded as a precondition, and PRD 5 requires it in the selection
    report. A model with no declared licence is not runnable, because the report
    it feeds could not be published.

  * **A model configuration must declare how to prove it launches.** Not that
    its weights exist — that it starts. This is a direct consequence of a
    measured failure: a 9 KB `llama-server.exe` stub passed an existence check
    and then failed `CreateProcess` with "file not found", killing a run at its
    last stage after five minutes of GPU work.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import sha256_text

# PRD 5 configuration families. A config whose family is not here is a typo.
FAMILIES = ("pose", "tracker", "object", "crop", "vlm", "person_detection")

# How preflight proves a configuration can actually start.
PROBE_ONNX_SESSION = "onnx_session"        # construct an InferenceSession
PROBE_SUBPROCESS_VERSION = "subprocess_version"  # run the binary with a flag
PROBE_PYTHON_IMPORT = "python_import"      # import the module that provides it
PROBE_NONE = "none"                        # pure-Python config, nothing to load

PROBES = (PROBE_ONNX_SESSION, PROBE_SUBPROCESS_VERSION, PROBE_PYTHON_IMPORT,
          PROBE_NONE)

# PRD 5: heavy models run sequentially on a 24 GB card. Concurrent residency is
# retained as an explicit mode for the 32 GB target rather than deleted, so a
# future run on larger hardware is comparable rather than a fresh baseline.
RESIDENCY_SEQUENTIAL = "sequential"
RESIDENCY_CONCURRENT = "concurrent"
RESIDENCY_MODES = (RESIDENCY_SEQUENTIAL, RESIDENCY_CONCURRENT)

MODEL_KEYS = {
    "config_id", "family", "probe", "description", "checkpoint", "checkpoint_sha256",
    "module", "binary", "binary_args", "input_size", "precision", "batch_size",
    "licence", "licence_url", "thresholds", "extra", "enabled",
    "requires_gpu", "vram_gb",
}
REQUIRED_MODEL_KEYS = {"config_id", "family", "probe", "licence"}

EXPERIMENT_KEYS = {
    "experiment_id", "description", "seed", "artifact_root", "residency",
    "vram_safety_margin_gb", "configs", "sources", "calibration",
    "reference_manifest", "taxonomy", "stages", "thresholds",
}
REQUIRED_EXPERIMENT_KEYS = {"experiment_id", "seed", "artifact_root", "configs"}


class ConfigError(ValueError):
    """A configuration is malformed, incomplete, or contradicts the PRD."""


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration not found: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ConfigError(
            "pyyaml is required to read configuration files") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _check_keys(data: dict, allowed: set[str], required: set[str],
                where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown}. A misspelled key would "
            f"otherwise be ignored and the run would report a configuration it "
            f"never applied. Allowed keys: {sorted(allowed)}"
        )
    missing = sorted(required - set(data))
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}")


def canonical_hash(data: Any) -> str:
    """Hash of the resolved content, not the file bytes.

    Comment and ordering changes must not alter the hash; a threshold change
    must. `sort_keys` gives that property.
    """
    return sha256_text(json.dumps(data, sort_keys=True, default=str))


@dataclass
class ModelConfig:
    """One named configuration from PRD 5's mandatory families."""

    config_id: str
    family: str
    probe: str
    licence: str
    description: str = ""
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    module: str | None = None
    binary: str | None = None
    binary_args: list[str] = field(default_factory=list)
    input_size: str | None = None
    precision: str | None = None
    batch_size: int | None = None
    licence_url: str | None = None
    thresholds: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    enabled: bool = True
    # Whether this configuration can only run on a GPU. Most cannot honestly
    # claim that: every model in this stack has been executed end to end on CPU
    # via ONNXRuntime or llama.cpp, slowly but correctly. Marking them GPU-only
    # would report `resource_unavailable` for configurations that demonstrably
    # run, which is the same class of dishonesty as silently substituting one.
    #
    # What CPU execution does change is timing, and PRD 5 requires latency and
    # resource figures per configuration. So a CPU run is permitted and the
    # device is recorded on every measurement, making a CPU number visibly
    # incomparable to a GPU one rather than quietly mixed with it.
    requires_gpu: bool = False
    # Approximate VRAM demand when a GPU is used, for the safety-margin gate.
    # Coarse on purpose: the real figure is measured at run time. This decides
    # whether a configuration may be attempted, not what it will consume.
    vram_gb: float = 0.0
    source_path: str = ""
    config_hash: str = ""

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise ConfigError(
                f"{self.config_id}: unknown family {self.family!r}; "
                f"expected one of {FAMILIES}")
        if self.probe not in PROBES:
            raise ConfigError(
                f"{self.config_id}: unknown probe {self.probe!r}; "
                f"expected one of {PROBES}")
        if not self.licence:
            raise ConfigError(
                f"{self.config_id}: a licence must be declared. PRD 3 requires "
                f"licence status as a precondition and PRD 5 requires it in the "
                f"selection report; a model whose licence is unknown cannot "
                f"feed a publishable comparison.")
        if self.probe == PROBE_ONNX_SESSION and not self.checkpoint:
            raise ConfigError(
                f"{self.config_id}: probe {self.probe!r} needs a checkpoint path")
        if self.probe == PROBE_SUBPROCESS_VERSION and not self.binary:
            raise ConfigError(
                f"{self.config_id}: probe {self.probe!r} needs a binary path")
        if self.probe == PROBE_PYTHON_IMPORT and not self.module:
            raise ConfigError(
                f"{self.config_id}: probe {self.probe!r} needs a module name")


@dataclass
class ExperimentConfig:
    """One run's inputs, seed, residency plan and configuration set."""

    experiment_id: str
    seed: int
    artifact_root: str
    configs: list[str]
    description: str = ""
    residency: str = RESIDENCY_SEQUENTIAL
    vram_safety_margin_gb: float = 2.0
    sources: list[str] = field(default_factory=list)
    calibration: str | None = None
    reference_manifest: str | None = None
    taxonomy: str | None = None
    stages: list[str] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    source_path: str = ""
    config_hash: str = ""

    def validate(self) -> None:
        if self.residency not in RESIDENCY_MODES:
            raise ConfigError(
                f"{self.experiment_id}: unknown residency {self.residency!r}; "
                f"expected one of {RESIDENCY_MODES}")
        if self.vram_safety_margin_gb < 0:
            raise ConfigError(
                f"{self.experiment_id}: vram_safety_margin_gb cannot be negative")
        if not self.configs:
            raise ConfigError(
                f"{self.experiment_id}: no model configurations listed. PRD 5 "
                f"requires matched reports across a family; an empty set cannot "
                f"produce a comparison.")
        if not isinstance(self.seed, int):
            raise ConfigError(
                f"{self.experiment_id}: seed must be an integer, so a run is "
                f"reproducible (PRD 17.10)")


def load_model_config(path: str | Path) -> ModelConfig:
    data = _load_yaml(path)
    _check_keys(data, MODEL_KEYS, REQUIRED_MODEL_KEYS, str(path))
    config = ModelConfig(**data)
    config.source_path = str(path)
    config.config_hash = canonical_hash(data)
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = _load_yaml(path)
    _check_keys(data, EXPERIMENT_KEYS, REQUIRED_EXPERIMENT_KEYS, str(path))
    config = ExperimentConfig(**data)
    config.source_path = str(path)
    config.config_hash = canonical_hash(data)
    config.validate()
    return config


def load_model_dir(directory: str | Path) -> dict[str, ModelConfig]:
    """Load every model configuration in a directory, keyed by config_id."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigError(f"model configuration directory not found: {directory}")
    configs: dict[str, ModelConfig] = {}
    for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        config = load_model_config(path)
        if config.config_id in configs:
            raise ConfigError(
                f"duplicate config_id {config.config_id!r} in {path} and "
                f"{configs[config.config_id].source_path}. Config IDs name "
                f"artifact directories, so a duplicate would let one "
                f"configuration overwrite another's output (PRD 17.9).")
        configs[config.config_id] = config
    if not configs:
        raise ConfigError(f"no model configurations found under {directory}")
    return configs


def resolve(experiment: ExperimentConfig, models: dict[str, ModelConfig]
            ) -> list[ModelConfig]:
    """The configurations this experiment names, in the order it names them."""
    missing = [c for c in experiment.configs if c not in models]
    if missing:
        raise ConfigError(
            f"{experiment.experiment_id} names configuration(s) that do not "
            f"exist: {missing}. Known: {sorted(models)}")
    return [models[c] for c in experiment.configs]


def families_present(configs: list[ModelConfig]) -> dict[str, list[str]]:
    """Which configurations cover which family, for the comparison check."""
    out: dict[str, list[str]] = {}
    for config in configs:
        out.setdefault(config.family, []).append(config.config_id)
    return out


def missing_comparison_families(configs: list[ModelConfig],
                                required: tuple[str, ...] = ("pose", "object")
                                ) -> list[str]:
    """Families that cannot support a comparison because only one entry exists.

    PRD 5: no configuration is called "best" until all runnable mandatory
    configurations have a matched report. A family with a single member cannot
    produce a comparison at all, so naming a winner there would be meaningless.
    """
    present = families_present(configs)
    return [f for f in required if len(present.get(f, [])) < 2]
