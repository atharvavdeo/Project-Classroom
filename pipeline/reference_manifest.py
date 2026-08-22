"""The reference manifest, and the wall around it (PRD 14, PRD 15).

PRD 15 states the isolation rule:

> Reference manifest becomes readable only in stage 12. Add an automated test
> that fails if a candidate/crop row includes `reference_id` or reference
> annotations.

This module is the only place in the pipeline that reads reference data, and
`assert_isolated` is the check that nothing upstream has. Both halves matter.
Restricting *reading* stops the obvious leak, where a candidate stage consults
the answers. Checking the *artifacts* catches the subtle one, where reference
fields are attached to rows early "just for debugging" and quietly become an
input to a threshold three commits later.

### Invalid rows are recorded, not skipped

PRD 14:

> Invalid source hash/PTS/class/out-of-bounds box is written to
> `validation_errors.json`; evaluation is incomplete, not silently skipped.

The distinction is the difference between "the detector missed 3 of 40" and "the
detector missed 3 of 40 that we checked, having quietly dropped 12 malformed
rows". A manifest with rejected rows produces an evaluation explicitly marked
incomplete, with the count of what could not be checked.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

VISIBILITY = ("visible", "partially_occluded", "heavily_occluded", "too_small",
              "blurred", "unknown")

REQUIRED_FIELDS = ("reference_id", "source_video_sha256", "source_pts_ms",
                   "class", "annotation_source")


class ReferenceIsolationError(RuntimeError):
    """Reference data has reached a stage that must not see it."""


@dataclass
class ReferenceRow:
    """One independently identified phone/chit sighting (PRD 14)."""

    reference_id: str
    source_video_sha256: str
    source_pts_ms: float
    cls: str
    annotation_source: str
    frame_index: int | None = None
    bbox_xyxy_source: tuple[float, float, float, float] | None = None
    seat_id: str | None = None
    visibility: str = "unknown"
    notes: str = ""

    @property
    def localised(self) -> bool:
        return self.bbox_xyxy_source is not None

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class ValidationError:
    reference_id: str
    field: str
    detail: str

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class ReferenceManifest:
    """Validated rows, rejected rows, and the reason for every rejection."""

    rows: list[ReferenceRow] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)
    source_sha256: str = ""
    path: str = ""
    approved_classes: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether every supplied row could be checked.

        False makes every metric derived from this manifest incomplete, and the
        evaluator says so rather than reporting a clean number over a subset.
        """
        return not self.errors

    def by_class(self, cls: str) -> list[ReferenceRow]:
        return [r for r in self.rows if r.cls == cls]

    def summary(self) -> dict:
        classes: dict[str, int] = {}
        visibility: dict[str, int] = {}
        for row in self.rows:
            classes[row.cls] = classes.get(row.cls, 0) + 1
            visibility[row.visibility] = visibility.get(row.visibility, 0) + 1
        return {
            "path": self.path,
            "accepted_rows": len(self.rows),
            "rejected_rows": len(self.errors),
            "complete": self.complete,
            "by_class": dict(sorted(classes.items())),
            "by_visibility": dict(sorted(visibility.items())),
            "localised_rows": sum(1 for r in self.rows if r.localised),
            "note": "every metric derived from this manifest is complete"
            if self.complete else
            f"{len(self.errors)} row(s) could not be validated. Every metric "
            f"derived from this manifest is INCOMPLETE; the rejected rows were "
            f"recorded, not skipped (PRD 14).",
        }


def _parse_box(value) -> tuple[float, float, float, float] | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith("[") \
            else [float(v) for v in value.replace(",", " ").split()]
    box = tuple(float(v) for v in value)
    if len(box) != 4:
        raise ValueError(f"expected 4 coordinates, got {len(box)}")
    return box


def load(path: str | Path, source_sha256: str = "",
         approved_classes: tuple[str, ...] = (),
         frame_w: int | None = None, frame_h: int | None = None,
         pts_span_ms: tuple[float, float] | None = None) -> ReferenceManifest:
    """Read and validate a JSONL or CSV reference manifest.

    Every check in PRD 14 is applied, and a failing row goes to `errors` with
    the field that failed rather than being dropped.
    """
    path = Path(path)
    manifest = ReferenceManifest(source_sha256=source_sha256, path=str(path),
                                 approved_classes=tuple(approved_classes))

    raw: list[dict] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    seen: set[str] = set()
    for index, row in enumerate(raw):
        reference_id = str(row.get("reference_id", "") or f"row_{index}")

        missing = [f for f in REQUIRED_FIELDS if not row.get(f)]
        if missing:
            manifest.errors.append(ValidationError(
                reference_id, ",".join(missing),
                f"required field(s) absent: {missing}"))
            continue

        if reference_id in seen:
            manifest.errors.append(ValidationError(
                reference_id, "reference_id",
                "duplicate; PRD 14 requires a unique immutable ID"))
            continue
        seen.add(reference_id)

        if source_sha256 and row["source_video_sha256"] != source_sha256:
            manifest.errors.append(ValidationError(
                reference_id, "source_video_sha256",
                f"{row['source_video_sha256'][:12]} does not match the source "
                f"{source_sha256[:12]}; this row describes a different file"))
            continue

        try:
            pts = float(row["source_pts_ms"])
        except (TypeError, ValueError):
            manifest.errors.append(ValidationError(
                reference_id, "source_pts_ms",
                f"{row['source_pts_ms']!r} is not a timestamp"))
            continue
        if pts < 0:
            manifest.errors.append(ValidationError(
                reference_id, "source_pts_ms", f"{pts} is negative"))
            continue
        if pts_span_ms and not (pts_span_ms[0] <= pts <= pts_span_ms[1]):
            manifest.errors.append(ValidationError(
                reference_id, "source_pts_ms",
                f"{pts} is outside the decoded span "
                f"{pts_span_ms[0]}-{pts_span_ms[1]}"))
            continue

        cls = str(row["class"])
        if approved_classes and cls not in approved_classes:
            manifest.errors.append(ValidationError(
                reference_id, "class",
                f"{cls!r} is not in the approved taxonomy {approved_classes}"))
            continue

        try:
            box = _parse_box(row.get("bbox_xyxy_source"))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            manifest.errors.append(ValidationError(
                reference_id, "bbox_xyxy_source", str(exc)))
            continue

        if box is not None:
            if box[2] <= box[0] or box[3] <= box[1]:
                manifest.errors.append(ValidationError(
                    reference_id, "bbox_xyxy_source",
                    f"degenerate box {box}"))
                continue
            if frame_w and frame_h and not (
                    0 <= box[0] and 0 <= box[1]
                    and box[2] <= frame_w and box[3] <= frame_h):
                manifest.errors.append(ValidationError(
                    reference_id, "bbox_xyxy_source",
                    f"box {box} lies outside {frame_w}x{frame_h}"))
                continue

        visibility = str(row.get("visibility") or "unknown")
        if visibility not in VISIBILITY:
            manifest.errors.append(ValidationError(
                reference_id, "visibility",
                f"{visibility!r} is not one of {VISIBILITY}"))
            continue

        frame_index = row.get("frame_index")
        manifest.rows.append(ReferenceRow(
            reference_id=reference_id,
            source_video_sha256=str(row["source_video_sha256"]),
            source_pts_ms=pts,
            cls=cls,
            annotation_source=str(row["annotation_source"]),
            frame_index=int(frame_index) if frame_index not in (None, "") else None,
            bbox_xyxy_source=box,
            seat_id=str(row["seat_id"]) if row.get("seat_id") else None,
            visibility=visibility,
            notes=str(row.get("notes") or ""),
        ))

    manifest.rows.sort(key=lambda r: (r.source_pts_ms, r.reference_id))
    return manifest


# ------------------------------------------------------------ isolation --

# Stages that must never contain reference annotations. Everything up to and
# including the VLM review; only stage 12 may hold them.
ISOLATED_STAGES = ("05_motion_roi", "06_segmentation", "07_tracking", "08_pose",
                   "09_object", "10_evidence", "11_gemma")

# Field-name fragments that indicate reference data.
MARKERS = ("reference_id", "reference_class", "reference_bbox", "ground_truth",
           "groundtruth", "is_true_positive", "label_source", "annotation_id")


def leaked_fields(payload) -> list[str]:
    """Every reference-looking key anywhere in a nested structure."""
    found: list[str] = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if any(marker in lowered for marker in MARKERS) \
                        or lowered.startswith("reference"):
                    found.append(f"{path}{key}")
                walk(value, f"{path}{key}.")
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(payload)
    return sorted(set(found))


def assert_isolated(run_dir: str | Path,
                    stages: tuple[str, ...] = ISOLATED_STAGES) -> list[str]:
    """Scan a run's pre-evaluation artifacts for reference leakage.

    Returns the violations rather than raising, so a caller can report every one
    instead of the first. `tests/test_reference_isolation.py` fails the build on
    a non-empty result, which is the automated test PRD 15 requires.
    """
    run_dir = Path(run_dir)
    violations: list[str] = []

    for stage in stages:
        directory = run_dir / stage
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix not in (".json", ".jsonl") or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(run_dir).as_posix()
            if path.suffix == ".jsonl":
                for number, line in enumerate(text.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for leaked in leaked_fields(payload):
                        violations.append(f"{relative}:{number} carries {leaked!r}")
            else:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                for leaked in leaked_fields(payload):
                    violations.append(f"{relative} carries {leaked!r}")

    return sorted(set(violations))
