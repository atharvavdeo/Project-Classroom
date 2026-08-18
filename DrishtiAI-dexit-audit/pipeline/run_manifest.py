"""The immutable run manifest and per-stage status (PRD 5, 13, 14, 17.10).

Two files, with different jobs:

  * `RUN_MANIFEST.json` records what the run *was*: source hashes, config
    hashes, checkpoint hashes, GPU and runtime versions, licences, the seed, and
    the code revision. Written once at preflight and never amended.

  * `RUN_STATUS.json` records what the run *did*: one row per stage, its state,
    its coverage, and the reason for anything not completed. Rewritten as the
    run progresses.

Separating them matters. The manifest is the answer to "could this be
reproduced"; the status is the answer to "what actually happened". A single file
mixing the two invites a later stage to amend a record of the inputs.

PRD 17.10 requires that every threshold, input, slice, model and seed is
versioned and hashed. `ConfigRecord` and `ModelRecord` are where that is held,
and `ModelRecord` is deliberately able to represent a model that could *not* be
loaded — PRD 5 requires an unavailable configuration to stay in the comparison
table rather than disappearing from it.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .artifacts import MANIFEST_FILE, STATUS_FILE, RunPaths, sha256_file, utc_now

# Per PRD 5: a configuration that cannot be loaded is marked, never substituted.
AVAILABLE = "available"
RESOURCE_UNAVAILABLE = "resource_unavailable"
MISSING_WEIGHTS = "missing_weights"
LAUNCH_FAILED = "launch_failed"
LICENCE_BLOCKED = "licence_blocked"

MODEL_STATES = (AVAILABLE, RESOURCE_UNAVAILABLE, MISSING_WEIGHTS,
                LAUNCH_FAILED, LICENCE_BLOCKED)

# Per-stage states for RUN_STATUS.json.
PENDING = "pending"
RUNNING = "running"
COMPLETE = "complete"
BLOCKED = "blocked"
FAILED = "failed"
SKIPPED_UNAVAILABLE = "skipped_unavailable"

STAGE_STATES = (PENDING, RUNNING, COMPLETE, BLOCKED, FAILED, SKIPPED_UNAVAILABLE)


@dataclass
class SourceRecord:
    """One source video, hashed before anything reads a pixel of it."""

    path: str
    sha256: str
    size_bytes: int


@dataclass
class ConfigRecord:
    """A configuration file and the hash of its resolved contents."""

    config_id: str
    path: str
    sha256: str
    family: str = ""


@dataclass
class ModelRecord:
    """One named model configuration and whether it can actually run.

    `state` is the important field. PRD 5 is explicit that a failed
    configuration stays in the comparison table as unavailable, so this record
    exists whether or not the model loaded, and `reason` says which.
    """

    config_id: str
    family: str
    state: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    licence: str | None = None
    licence_url: str | None = None
    reason: str | None = None
    launch_probe: str | None = None      # what proved it can start, not just exist
    input_size: str | None = None
    precision: str | None = None
    # The device this configuration will actually execute on, and why. PRD 5
    # requires latency and resource figures per configuration, and a CPU figure
    # is not comparable to a GPU one -- so the device travels with every
    # measurement rather than being assumed from the machine.
    device: str | None = None
    device_reason: str | None = None

    def __post_init__(self) -> None:
        if self.state not in MODEL_STATES:
            raise ValueError(
                f"unknown model state {self.state!r}; expected one of {MODEL_STATES}")


@dataclass
class EnvironmentRecord:
    """The machine, recorded so a result can be attributed to hardware."""

    python: str
    platform: str
    gpu_name: str | None = None
    gpu_vram_total_gb: float | None = None
    driver_version: str | None = None
    cuda_runtime: str | None = None
    torch_version: str | None = None
    torch_cuda_available: bool = False
    onnxruntime_providers: list[str] = field(default_factory=list)
    free_disk_gb: float | None = None
    host_ram_gb: float | None = None

    @property
    def gpu_usable(self) -> bool:
        return bool(self.gpu_name) and self.torch_cuda_available


@dataclass
class RunManifest:
    """Written once. Amending it would defeat the point of having it."""

    run_id: str
    created_utc: str
    seed: int
    code_revision: str | None = None
    code_dirty: bool = False
    parent_run_id: str | None = None
    sources: list[SourceRecord] = field(default_factory=list)
    configs: list[ConfigRecord] = field(default_factory=list)
    models: list[ModelRecord] = field(default_factory=list)
    environment: EnvironmentRecord | None = None
    calibration_version: str | None = None
    reference_manifest_sha256: str | None = None
    notes: list[str] = field(default_factory=list)

    def model(self, config_id: str) -> ModelRecord | None:
        for record in self.models:
            if record.config_id == config_id:
                return record
        return None

    @property
    def available_models(self) -> list[str]:
        return [m.config_id for m in self.models if m.state == AVAILABLE]

    @property
    def unavailable_models(self) -> list[ModelRecord]:
        return [m for m in self.models if m.state != AVAILABLE]

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, paths: RunPaths) -> Path:
        return paths.write_json(MANIFEST_FILE, self.to_dict())

    @classmethod
    def read(cls, paths: RunPaths) -> "RunManifest":
        data = json.loads((paths.dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        env = data.pop("environment", None)
        manifest = cls(
            **{k: v for k, v in data.items()
               if k not in ("sources", "configs", "models")},
            sources=[SourceRecord(**s) for s in data.get("sources", [])],
            configs=[ConfigRecord(**c) for c in data.get("configs", [])],
            models=[ModelRecord(**m) for m in data.get("models", [])],
        )
        if env:
            manifest.environment = EnvironmentRecord(**env)
        return manifest


@dataclass
class StageStatus:
    stage: str
    state: str = PENDING
    started_utc: str | None = None
    finished_utc: str | None = None
    reason: str | None = None
    # PRD 18: the complete decodable duration is processed, or every missing
    # range is explicitly failed. Coverage is how that claim is evidenced.
    covered_ms: float | None = None
    total_ms: float | None = None
    counts: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in STAGE_STATES:
            raise ValueError(
                f"unknown stage state {self.state!r}; expected one of {STAGE_STATES}")

    @property
    def coverage_fraction(self) -> float | None:
        if not self.total_ms:
            return None
        return (self.covered_ms or 0.0) / self.total_ms


class RunStatus:
    """Live per-stage state, rewritten as the run progresses."""

    def __init__(self, paths: RunPaths, run_id: str):
        self.paths = paths
        self.run_id = run_id
        self.stages: dict[str, StageStatus] = {}

    def begin(self, stage: str) -> StageStatus:
        status = StageStatus(stage=stage, state=RUNNING, started_utc=utc_now())
        self.stages[stage] = status
        self.write()
        return status

    def finish(self, stage: str, state: str = COMPLETE, reason: str | None = None,
               covered_ms: float | None = None, total_ms: float | None = None,
               **counts) -> StageStatus:
        status = self.stages.get(stage) or StageStatus(stage=stage)
        status.state = state
        status.reason = reason
        status.finished_utc = utc_now()
        if covered_ms is not None:
            status.covered_ms = covered_ms
        if total_ms is not None:
            status.total_ms = total_ms
        status.counts.update(counts)
        if state not in STAGE_STATES:
            raise ValueError(f"unknown stage state {state!r}")
        self.stages[stage] = status
        self.write()
        return status

    @property
    def blocked(self) -> list[str]:
        return [s.stage for s in self.stages.values() if s.state == BLOCKED]

    def write(self) -> Path:
        return self.paths.write_json(STATUS_FILE, {
            "run_id": self.run_id,
            "updated_utc": utc_now(),
            "stages": [asdict(s) for s in self.stages.values()],
        })


# ------------------------------------------------------------- environment --

def code_revision(root: str | Path) -> tuple[str | None, bool]:
    """Git revision and whether the tree is dirty.

    A dirty tree is recorded rather than refused: refusing would make the tool
    unusable mid-development. But a result produced from uncommitted code must
    say so, or it cannot be reproduced.
    """
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=15)
        if rev.returncode != 0:
            return None, False
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                               capture_output=True, text=True, timeout=15)
        return rev.stdout.strip(), bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, False


def probe_environment(artifact_root: str | Path) -> EnvironmentRecord:
    """Measure the machine. Never guess a capability that was not observed."""
    from .artifacts import free_bytes

    record = EnvironmentRecord(
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        free_disk_gb=round(free_bytes(artifact_root) / 1e9, 1),
    )

    try:
        # numpy first: in an Anaconda environment two copies of libiomp5md.dll
        # are reachable, and importing torch into a process that has not already
        # loaded the MKL copy aborts with OpenMP error #15 before any of our
        # code runs. Measured directly on this machine.
        import numpy  # noqa: F401
        import torch

        record.torch_version = torch.__version__
        record.torch_cuda_available = bool(torch.cuda.is_available())
        if record.torch_cuda_available:
            record.gpu_name = torch.cuda.get_device_name(0)
            record.gpu_vram_total_gb = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            record.cuda_runtime = getattr(torch.version, "cuda", None)
    except Exception:  # noqa: BLE001 - absence is a finding, not a crash
        pass

    # nvidia-smi answers even when torch is a CPU-only build, which is exactly
    # the case on the development machine. Reporting "no GPU" there would be
    # wrong; reporting "GPU present but torch cannot use it" is the truth.
    try:
        smi = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if smi.returncode == 0 and smi.stdout.strip():
            name, total_mib, driver = [
                p.strip() for p in smi.stdout.strip().splitlines()[0].split(",")]
            record.gpu_name = record.gpu_name or name
            record.gpu_vram_total_gb = record.gpu_vram_total_gb or round(
                float(total_mib) / 1024.0, 1)
            record.driver_version = driver
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    try:
        import onnxruntime
        record.onnxruntime_providers = list(onnxruntime.get_available_providers())
    except Exception:  # noqa: BLE001
        pass

    try:
        import psutil
        record.host_ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass

    return record


def source_record(path: str | Path) -> SourceRecord:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"source video not found: {path}")
    return SourceRecord(str(path), sha256_file(path), path.stat().st_size)
