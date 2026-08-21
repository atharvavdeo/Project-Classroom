"""The immutable run folder (PRD 13).

One directory per run, holding every intermediate and final artifact. Two
properties are enforced rather than encouraged:

  * **Every generated image carries source provenance.** PRD 6.1 requires that
    no artifact exists without a source PTS/frame mapping, so filenames are
    built from the source timestamp and the accompanying sidecar records the
    video hash. An image that cannot say where it came from cannot be used as
    evidence.

  * **A completed run is immutable.** Reprocessing creates a new run ID linked
    to its parent (PRD 13). `seal()` writes the marker and `assert_open()`
    refuses writes afterwards, so a later stage cannot quietly rewrite the
    output a report was generated from.

Filenames use zero-padded millisecond PTS prefixes so that lexical sort equals
temporal sort. Ten digits covers 10^10 ms, about 115 days of recording, which is
comfortably beyond any single examination file.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# The stage directories of PRD 13, in execution order. Held as data rather than
# created ad hoc, so a stage cannot invent a sibling directory that the artifact
# index does not know about.
STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source", ("contact_sheets",)),
    ("01_ingest", ("anomalies",)),
    ("02_health", ()),
    ("03_alignment", ()),
    ("04_calibration", ("overlays", "masks")),
    ("05_motion_roi", ("heatmaps", "overlays")),
    ("06_segmentation", ()),
    ("07_tracking", ()),
    ("08_pose", ()),
    ("09_object", ()),
    ("10_evidence", ("events",)),
    ("11_gemma", ()),
    ("12_evaluation", ("misses", "false_positives")),
    ("13_reports", ()),
    # Continuous per-person records over the whole recording, as opposed
    # to the candidate-gated stages above. Numbered after the reports
    # because it was added later; it depends on nothing they produce.
    ("14_person_timeline", ("annotated",)),
    ("logs", ()),
)

SEAL_FILE = "RUN_SEALED"
MANIFEST_FILE = "RUN_MANIFEST.json"
STATUS_FILE = "RUN_STATUS.json"


class RunSealed(RuntimeError):
    """A write was attempted against a completed run."""


def pts_prefix(pts_ms: float) -> str:
    """Zero-padded millisecond prefix, so lexical order equals temporal order."""
    if pts_ms < 0:
        raise ValueError(f"negative PTS: {pts_ms}")
    return f"{int(round(pts_ms)):010d}"


def pts_name(pts_ms: float, label: str, suffix: str) -> str:
    """`0000012345_blackout.png` — sortable, and self-describing in a listing."""
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return f"{pts_prefix(pts_ms)}_{clean}{suffix}"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


@dataclass
class RunPaths:
    """Handles for one run directory."""

    root: Path
    run_id: str
    _stage_index: dict[str, Path] = field(default_factory=dict, repr=False)

    @property
    def dir(self) -> Path:
        return self.root / self.run_id

    @property
    def sealed(self) -> bool:
        return (self.dir / SEAL_FILE).exists()

    def assert_open(self) -> None:
        if self.sealed:
            raise RunSealed(
                f"{self.run_id} is sealed. Reprocessing must create a new run "
                f"linked to this one as its parent, never overwrite it."
            )

    def stage(self, name: str) -> Path:
        """Return a stage directory, refusing names not declared in PRD 13."""
        if name not in self._stage_index:
            known = ", ".join(sorted(self._stage_index))
            raise KeyError(f"unknown stage {name!r}; declared stages are {known}")
        return self._stage_index[name]

    def config_dir(self, stage: str, config_id: str) -> Path:
        """Per-configuration subdirectory.

        Every model configuration writes into its own directory. PRD 17.9: model
        configurations cannot overwrite one another's artifacts, and the cheapest
        way to guarantee that is to give each one a namespace it cannot escape.
        """
        self.assert_open()
        path = self.stage(stage) / config_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def event_dir(self, event_id: str) -> Path:
        self.assert_open()
        path = self.stage("10_evidence") / "events" / event_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str | Path, payload: dict | list) -> Path:
        self.assert_open()
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def append_jsonl(self, relative: str | Path, record: dict) -> Path:
        """Append one record. JSONL is the format for anything per-row.

        Append-only by design: a stage that re-runs writes a new run, so a
        rewritten line would mean a stage had silently replaced its own history.
        """
        self.assert_open()
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return path

    def log(self, name: str, text: str) -> Path:
        self.assert_open()
        path = self.stage("logs") / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip("\n") + "\n")
        return path

    def seal(self) -> Path:
        """Mark the run complete. Nothing may be written afterwards."""
        marker = self.dir / SEAL_FILE
        marker.write_text(utc_now(), encoding="utf-8")
        return marker

    def index(self) -> list[dict]:
        """Every file in the run, with size and stage. Feeds the artifact index."""
        rows = []
        for path in sorted(self.dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.dir)
                rows.append({
                    "path": relative.as_posix(),
                    "stage": relative.parts[0] if relative.parts else "",
                    "bytes": path.stat().st_size,
                })
        return rows


def create_run(root: str | Path, run_id: str | None = None) -> RunPaths:
    """Build the PRD 13 skeleton and return handles to it."""
    root = Path(root)
    run_id = run_id or new_run_id()
    paths = RunPaths(root=root, run_id=run_id)
    if paths.dir.exists():
        raise FileExistsError(
            f"{paths.dir} already exists. Run IDs are immutable; pick a new one."
        )
    for name, children in STAGES:
        stage_path = paths.dir / name
        stage_path.mkdir(parents=True)
        for child in children:
            (stage_path / child).mkdir(parents=True)
        paths._stage_index[name] = stage_path
    return paths


def open_run(root: str | Path, run_id: str) -> RunPaths:
    """Reopen an existing run directory, sealed or not."""
    paths = RunPaths(root=Path(root), run_id=run_id)
    if not paths.dir.is_dir():
        raise FileNotFoundError(f"no run directory at {paths.dir}")
    for name, _children in STAGES:
        stage_path = paths.dir / name
        # Index declared stages whether or not the directory is already there.
        # A run folder created before a stage existed would otherwise refuse
        # that stage forever, which makes every existing run un-extendable the
        # moment the layout grows. Sealed runs stay untouched: `assert_open`
        # and the write helpers are what enforce immutability, not the absence
        # of a directory.
        if not stage_path.is_dir() and not paths.sealed:
            stage_path.mkdir(parents=True, exist_ok=True)
        if stage_path.is_dir():
            paths._stage_index[name] = stage_path
    return paths


def free_bytes(path: str | Path) -> int:
    """Writable capacity at a path, for the PRD 3 capacity precondition."""
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free
