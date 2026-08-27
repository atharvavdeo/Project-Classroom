"""One evidence record per proposal. The output contract of this system.

Everything the pipeline can eventually say about a candidate is assembled from
these. A record is created the moment a detector proposes an object and is
never destroyed -- not when a gate rejects it, not when SAM 3 calls it a
keyboard, not when a reviewer dismisses it. Suppression is written *into* the
record as a reason code; it does not delete the record.

### Why records survive their own rejection

Because the interesting question is almost always about the ones that died. The
measured history of this corpus is a list of suppressions that turned out to
matter: 729 detections dropped for being away from a wrist (correct), 402
dropped as keyboards (correct), and an unknown number of true chits dropped by
SAM 3 (incorrect -- it missed frames 2 and 5 of the reference set). Only the
first two of those are visible today, and only because they were recounted by
hand. A record that keeps its own rejection makes precision and recall
computable later, from artifacts already on disk, the moment ground-truth
labels exist.

### The three-layer split, and what it protects

    Layer                mutable?   who writes it
    -----                --------   -------------
    Detector, Geometry   frozen     the detector, once
    reason_codes         append     the gates, SAM 3, the fusion machine
    Review               append     a human, through the dashboard

`Detector` and `Geometry` are frozen dataclasses. This is not decoration: it is
the mechanical guarantee that no later stage can quietly rewrite what the
detector originally said. SAM 3 disagreeing with the paper detector must show
up as *two* stored claims plus a reason code, never as an edited class name.

### What a record deliberately does not carry

No mask arrays, no crops, no `rle_mask`, no `points`. A four-hour recording
sampled at 4 Hz produces on the order of 10^5 records; masks would make the
artifact unopenable and nothing downstream reads them. The record carries the
box and the frame time, which is enough to re-cut the crop from the source
video when a reviewer asks to see it.

### What a record may never carry

A field named `cheating`, `guilty`, `violation` or any synonym, and any number
presented as a probability of one. The record holds observations and the
reasons attached to them. The policy state lives in `pipeline/fusion.py`, and
the two states that assert a violation -- `human_confirmed`, `human_dismissed`
-- can only be written by `Review`, by a person.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from pipeline import reason_codes as rc

# Bump on any change that alters the meaning of an existing field. Consumers
# (the dashboard, any evaluation script) must refuse a contract they do not
# know, rather than misread a renamed field.
CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class Provenance:
    """Where and when. Enough to re-cut the crop from the source video."""

    run_id: str = ""
    video: str = ""
    camera: str = ""           # camera identity; falls back to the video stem
    pts_ms: float = 0.0        # decoded presentation timestamp, the only clock
    stage: str = ""            # the pipeline stage that produced the proposal


@dataclass(frozen=True)
class Detector:
    """What a model said, exactly as it said it. Frozen on purpose."""

    name: str = ""
    version: str = ""
    class_name: str = ""
    confidence: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}" if self.version else self.name


@dataclass(frozen=True)
class Geometry:
    """Boxes in source-image pixels, plus the scale they must be read against.

    `size_ratio_to_person` is the figure the size gate uses, and it is stored
    rather than recomputed because the gate's threshold is per-camera: 0.12 is
    the 63rd percentile on 1280x720 footage and the 25th on 360x450. A record
    that carries only a pass/fail cannot be re-gated when that is corrected.
    """

    box: tuple = (0.0, 0.0, 0.0, 0.0)          # x1, y1, x2, y2
    image_size: tuple = (0, 0)                  # w, h of the source frame
    person_box: tuple = (0.0, 0.0, 0.0, 0.0)
    size_ratio_to_person: float = 0.0
    torso_scale_px: float = 0.0


@dataclass
class Sam3Claim:
    """One segment SAM 3 returned for one prompted phrase."""

    phrase: str = ""
    confidence: float = 0.0
    overlap_with_proposal: float = 0.0
    box: tuple = (0.0, 0.0, 0.0, 0.0)


@dataclass
class Sam3Response:
    """The verifier's whole answer, including the hard negatives.

    `responded` separates "SAM 3 said nothing matched" from "SAM 3 never
    answered". Collapsing those two is the single most dangerous thing this
    contract prevents: an outage that reads as an all-clear.
    """

    attempted: bool = False
    responded: bool = False
    error: str = ""
    prompt: str = ""
    model: str = ""
    request_id: str = ""
    latency_ms: float = 0.0
    target_claims: list = field(default_factory=list)     # Sam3Claim
    negative_claims: list = field(default_factory=list)   # Sam3Claim

    @property
    def is_abstention(self) -> bool:
        return self.attempted and not self.responded


@dataclass
class Association:
    """Who this object belongs to, and how sure we are of that."""

    track_id: int = -1
    seat: str = "unattributed"
    seat_calibrated: bool = False
    wrist_resolved: bool = False
    wrist_distance_norm: float = -1.0    # in person-box widths; -1 = unknown
    nearest_wrist: str = ""              # "left" | "right" | ""
    competing_tracks: int = 0            # >0 means ownership is contested


@dataclass
class Quality:
    """Whether the system could see well enough to be believed."""

    person_box_px: int = 0
    head_resolvable: bool = False
    hands_resolvable: bool = False
    occluded: bool = False
    track_coverage: float = 0.0


@dataclass
class ReviewEntry:
    """One human action. Append-only: corrections are new entries."""

    decision: str = ""      # confirmed | dismissed | needs_better_view
    reviewer: str = ""
    reason: str = ""
    at: str = ""            # ISO 8601


@dataclass
class EvidenceRecord:
    record_id: str = ""
    contract_version: str = CONTRACT_VERSION

    provenance: Provenance = field(default_factory=Provenance)
    detector: Detector = field(default_factory=Detector)
    geometry: Geometry = field(default_factory=Geometry)
    sam3: Sam3Response = field(default_factory=Sam3Response)
    association: Association = field(default_factory=Association)
    quality: Quality = field(default_factory=Quality)

    # Append-only. Order is the order decisions were taken.
    reason_codes: list = field(default_factory=list)
    review: list = field(default_factory=list)

    def add_reason(self, code: str) -> "EvidenceRecord":
        """Attach a vocabulary code. Never replaces, never deduplicates away
        an earlier code that argued the other way."""
        rc.get(code)                      # raises on anything invented
        if code not in self.reason_codes:
            self.reason_codes.append(code)
        return self

    def add_review(self, entry: ReviewEntry) -> "EvidenceRecord":
        self.review.append(entry)
        return self

    def reasons(self) -> list:
        return [rc.get(c) for c in self.reason_codes]

    def has(self, code: str) -> bool:
        return code in self.reason_codes

    def routes(self) -> list:
        """The states this record's reasons argue for, no weighting applied."""
        return [r.route for r in self.reasons()]

    def assign_id(self) -> "EvidenceRecord":
        """Stable id from provenance + geometry + detector.

        Content-addressed rather than sequential so that re-running a stage
        produces the same ids, and so two records that differ only in their
        reason codes collide -- which is what we want, because that is the same
        proposal seen twice.
        """
        seed = json.dumps([
            self.provenance.run_id, self.provenance.video,
            round(self.provenance.pts_ms, 1), self.detector.label,
            self.detector.class_name, [round(v, 1) for v in self.geometry.box],
            self.association.track_id,
        ], sort_keys=True)
        self.record_id = hashlib.sha1(seed.encode()).hexdigest()[:16]
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def write_jsonl(records, path) -> int:
    """Write records to a JSONL file. Returns the count written."""
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")
            written += 1
    return written


def _rebuild(cls, blob):
    return cls(**blob) if isinstance(blob, dict) else cls()


def read_jsonl(path) -> list:
    """Read records back, refusing a contract version we do not understand."""
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            blob = json.loads(line)
            seen = blob.get("contract_version", "")
            if seen != CONTRACT_VERSION:
                raise ValueError(
                    f"{path}: record {blob.get('record_id')} declares contract "
                    f"{seen!r}, this build speaks {CONTRACT_VERSION!r}. "
                    f"Refusing to guess at renamed fields.")
            sam3 = blob.get("sam3") or {}
            out.append(EvidenceRecord(
                record_id=blob.get("record_id", ""),
                provenance=_rebuild(Provenance, blob.get("provenance")),
                detector=_rebuild(Detector, blob.get("detector")),
                geometry=_rebuild(Geometry, blob.get("geometry")),
                sam3=Sam3Response(
                    **{**sam3,
                       "target_claims": [Sam3Claim(**c) for c in
                                         sam3.get("target_claims", [])],
                       "negative_claims": [Sam3Claim(**c) for c in
                                           sam3.get("negative_claims", [])]}),
                association=_rebuild(Association, blob.get("association")),
                quality=_rebuild(Quality, blob.get("quality")),
                reason_codes=list(blob.get("reason_codes") or []),
                review=[ReviewEntry(**e) for e in (blob.get("review") or [])],
            ))
    return out
