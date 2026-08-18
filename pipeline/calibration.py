"""Dense spatial calibration, approval and seat attribution (PRD 7).

The governing principle is PRD 7's own: **a tracker ID is not a student identity
in an occluded 30-60 person room.** Anonymous seat and desk context is the stable
identifier, and a track supports a seat only while the association is clearly
enough evidenced. Everything here follows from that.

Three properties are enforced rather than encouraged:

**Approval is a human act with a named reviewer.** Automatic polygon proposal
may assist and never approves (PRD 7). `validate()` produces issues;
`approve()` refuses while any blocking issue stands and records who approved
what. An unapproved profile cannot attribute a seat — it can still support
health diagnostics, which is exactly the split PRD 3 asks for.

**Editing a profile creates a new version, never an edit in place.** PRD 7:
modifying a calibration requires reprocessing and old events are never silently
rewritten. `revise()` returns a new profile carrying its parent's identity; the
old one is marked superseded and stays readable.

**Attribution refuses to guess.** PRD 7 forbids assigning by nearest box centre
and forbids forcing a seat across a shared boundary. When the evidence does not
separate two candidate seats, the answer is `ambiguous_seat` with every
candidate score retained — not the higher score silently chosen.

An epoch is the unit of geometric validity. A sustained change in camera
geometry begins a new epoch (PRD 6.2, `camera_repositioned`), and seat
attribution is blocked in the new epoch until a profile for it is approved.
Without that, a bumped camera keeps attributing motion to seats that have moved
in the frame.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------ kinds --

# Zones belong to a seat and subdivide it (PRD 7).
SEAT_CONTEXT = "seat_context"
DESK_WORK_SURFACE = "desk_work_surface"
UPPER_BODY = "upper_body"
LAP_BAG = "lap_bag"
AISLE = "aisle"
STANDING = "standing"

ZONE_KINDS = (SEAT_CONTEXT, DESK_WORK_SURFACE, UPPER_BODY, LAP_BAG, AISLE, STANDING)
REQUIRED_ZONES = (SEAT_CONTEXT, DESK_WORK_SURFACE, UPPER_BODY, LAP_BAG)

# Masks are frame-global (PRD 7).
MONITOR_EXCLUSION = "monitor_exclusion"
ENVIRONMENT = "environment"
REFLECTION = "reflection"
INVIGILATOR_PATH = "invigilator_path"
STATIC_BACKGROUND = "static_background"
OVERLAY = "overlay"

MASK_KINDS = (MONITOR_EXCLUSION, ENVIRONMENT, REFLECTION, INVIGILATOR_PATH,
              STATIC_BACKGROUND, OVERLAY)

# Masks whose regions are never scored for candidate motion. `static_background`
# is the exception: it marks where alignment may look, not where scoring may
# not, so it is deliberately absent from this set.
EXCLUDED_FROM_SCORING = frozenset({
    MONITOR_EXCLUSION, ENVIRONMENT, REFLECTION, INVIGILATOR_PATH, OVERLAY})

# Approval lifecycle.
DRAFT = "draft"
SUBMITTED = "submitted"
APPROVED = "approved"
REJECTED = "rejected"
SUPERSEDED = "superseded"
APPROVAL_STATES = (DRAFT, SUBMITTED, APPROVED, REJECTED, SUPERSEDED)

# Review conditions PRD 7 requires images for.
REVIEW_CONDITIONS = ("crowded", "sparse", "changed_light", "aisle_reflection")

# Attribution outcomes (PRD 7).
AMBIGUOUS_SEAT = "ambiguous_seat"
UNATTRIBUTED = "unattributed"

BLOCKING = "blocking"
WARNING = "warning"


class CalibrationError(ValueError):
    """A profile is malformed, or an operation would violate PRD 7."""


@dataclass
class ValidationIssue:
    code: str
    severity: str
    detail: str
    seat_id: str | None = None

    def __str__(self) -> str:
        where = f" [{self.seat_id}]" if self.seat_id else ""
        return f"{self.severity.upper()}{where} {self.code}: {self.detail}"


@dataclass
class Zone:
    kind: str
    polygon: list[tuple[float, float]]

    def validate(self) -> None:
        if self.kind not in ZONE_KINDS:
            raise CalibrationError(
                f"unknown zone kind {self.kind!r}; expected one of {ZONE_KINDS}")


@dataclass
class Seat:
    """One anonymous seat. The label is a physical placard, never a person."""

    seat_id: str
    zones: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    desk_line_y: float | None = None
    # Seats sharing a desk or an aisle edge. Declared, because an undeclared
    # shared boundary is the case PRD 7 forbids forcing an attribution across.
    adjacent: list[str] = field(default_factory=list)
    # Boundaries the operator has explicitly marked as unresolvable. Motion here
    # is `ambiguous_seat` by construction rather than by failing a threshold.
    ambiguous_with: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def polygon(self) -> list[tuple[float, float]]:
        return self.zones.get(SEAT_CONTEXT, [])


@dataclass
class MaskRegion:
    kind: str
    polygon: list[tuple[float, float]]
    note: str = ""

    def validate(self) -> None:
        if self.kind not in MASK_KINDS:
            raise CalibrationError(
                f"unknown mask kind {self.kind!r}; expected one of {MASK_KINDS}")


@dataclass
class CameraEpoch:
    """A span over which the camera geometry is believed unchanged."""

    epoch_id: str
    camera_id: str
    start_pts_ms: float
    end_pts_ms: float | None = None
    reason: str = "initial"
    reference_pts_ms: float = 0.0
    reference_frame_sha256: str | None = None

    def covers(self, pts_ms: float) -> bool:
        if pts_ms < self.start_pts_ms:
            return False
        return self.end_pts_ms is None or pts_ms <= self.end_pts_ms


@dataclass
class CalibrationProfile:
    """One reviewed profile, valid for one camera epoch (PRD 7)."""

    camera_id: str
    epoch_id: str
    version: int
    frame_w: int
    frame_h: int
    source_sha256: str | None = None
    reference_pts_ms: float = 0.0
    operator: str = ""
    reviewer: str = ""
    approval_state: str = DRAFT
    approved_utc: str | None = None
    parent_version: int | None = None
    seats: list[Seat] = field(default_factory=list)
    masks: list[MaskRegion] = field(default_factory=list)
    review_images: dict[str, str] = field(default_factory=dict)
    confidence_notes: str = ""
    proposal_source: str = "manual"   # "manual" or the proposer that assisted

    # ------------------------------------------------------------ helpers --

    @property
    def approved(self) -> bool:
        return self.approval_state == APPROVED

    def seat(self, seat_id: str) -> Seat | None:
        return next((s for s in self.seats if s.seat_id == seat_id), None)

    def masks_of(self, kind: str) -> list[MaskRegion]:
        return [m for m in self.masks if m.kind == kind]

    @property
    def version_id(self) -> str:
        return f"{self.camera_id}/{self.epoch_id}/v{self.version}"

    # --------------------------------------------------------- validation --

    def validate(self) -> list[ValidationIssue]:
        """Every reason this profile could not be approved.

        PRD 7 names five rejections explicitly -- zero area, out of frame,
        unresolved overlap, missing background support, unlabelled boundaries --
        and each is checked here. Warnings are recorded but do not block.
        """
        issues: list[ValidationIssue] = []

        if not self.seats:
            issues.append(ValidationIssue(
                "no_seats", BLOCKING, "a profile with no seats cannot attribute"))
        if not self.operator:
            issues.append(ValidationIssue(
                "no_operator", BLOCKING, "the operator who authored this is unrecorded"))

        seen: set[str] = set()
        for seat in self.seats:
            if seat.seat_id in seen:
                issues.append(ValidationIssue(
                    "duplicate_seat_id", BLOCKING,
                    "seat ids name artifact directories and must be unique",
                    seat.seat_id))
            seen.add(seat.seat_id)

            missing = [z for z in REQUIRED_ZONES if z not in seat.zones]
            if missing:
                issues.append(ValidationIssue(
                    "missing_zone", BLOCKING,
                    f"required zone(s) absent: {missing}", seat.seat_id))

            for kind, polygon in seat.zones.items():
                if kind not in ZONE_KINDS:
                    issues.append(ValidationIssue(
                        "unknown_zone_kind", BLOCKING, f"{kind!r}", seat.seat_id))
                    continue
                issues += self._polygon_issues(polygon, f"zone {kind}", seat.seat_id)

            if seat.desk_line_y is not None and not (0 <= seat.desk_line_y <= self.frame_h):
                issues.append(ValidationIssue(
                    "desk_line_out_of_frame", BLOCKING,
                    f"y={seat.desk_line_y} outside 0..{self.frame_h}", seat.seat_id))

            for other in seat.adjacent:
                if other not in {s.seat_id for s in self.seats}:
                    issues.append(ValidationIssue(
                        "adjacency_unknown_seat", BLOCKING,
                        f"declared adjacent to {other!r}, which does not exist",
                        seat.seat_id))

        for mask in self.masks:
            if mask.kind not in MASK_KINDS:
                issues.append(ValidationIssue(
                    "unknown_mask_kind", BLOCKING, f"{mask.kind!r}"))
                continue
            issues += self._polygon_issues(mask.polygon, f"mask {mask.kind}")

        if not self.masks_of(STATIC_BACKGROUND):
            issues.append(ValidationIssue(
                "missing_background_support", BLOCKING,
                "alignment needs a static-background region that excludes seats, "
                "aisles, monitors and reflections (PRD 6.3.1)"))

        issues += self._overlap_issues()

        missing_reviews = [c for c in REVIEW_CONDITIONS if c not in self.review_images]
        if missing_reviews:
            issues.append(ValidationIssue(
                "missing_review_images", WARNING,
                f"no review image for: {missing_reviews}"))
        if not self.reviewer and self.approval_state in (SUBMITTED, APPROVED):
            issues.append(ValidationIssue(
                "no_reviewer", BLOCKING, "approval requires a named reviewer"))

        return issues

    def _polygon_issues(self, polygon, what: str, seat_id: str | None = None
                        ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if len(polygon) < 3:
            issues.append(ValidationIssue(
                "degenerate_polygon", BLOCKING,
                f"{what} has {len(polygon)} points; a region needs at least 3",
                seat_id))
            return issues
        if polygon_area(polygon) <= 0.0:
            issues.append(ValidationIssue(
                "zero_area", BLOCKING, f"{what} encloses no area", seat_id))
        outside = [(x, y) for x, y in polygon
                   if not (0 <= x <= self.frame_w and 0 <= y <= self.frame_h)]
        if outside:
            issues.append(ValidationIssue(
                "out_of_frame", BLOCKING,
                f"{what} has {len(outside)} point(s) outside "
                f"{self.frame_w}x{self.frame_h}", seat_id))
        return issues

    def _overlap_issues(self) -> list[ValidationIssue]:
        """Seat footprints may not overlap unless the boundary is declared.

        An undeclared overlap is the case PRD 7 forbids forcing an attribution
        across: motion in that region genuinely belongs to either seat, and
        silently picking one is the failure. Declaring it makes the region
        `ambiguous_seat` by construction.
        """
        issues: list[ValidationIssue] = []
        for i, a in enumerate(self.seats):
            for b in self.seats[i + 1:]:
                if not a.polygon or not b.polygon:
                    continue
                overlap = polygon_overlap_px(a.polygon, b.polygon,
                                             self.frame_w, self.frame_h)
                if overlap <= 0:
                    continue
                declared = (b.seat_id in a.ambiguous_with
                            or a.seat_id in b.ambiguous_with)
                if not declared:
                    issues.append(ValidationIssue(
                        "unlabelled_boundary", BLOCKING,
                        f"{a.seat_id} and {b.seat_id} overlap by {overlap:.0f} px "
                        f"with no declared ambiguous boundary",
                        a.seat_id))
        return issues

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        return [i for i in self.validate() if i.severity == BLOCKING]

    # ----------------------------------------------------------- lifecycle --

    def submit(self, operator: str | None = None) -> None:
        if operator:
            self.operator = operator
        if self.approval_state not in (DRAFT, REJECTED):
            raise CalibrationError(
                f"cannot submit from state {self.approval_state!r}")
        self.approval_state = SUBMITTED

    def approve(self, reviewer: str, when_utc: str) -> None:
        """Approve, or refuse and say exactly why.

        PRD 7: automatic polygon proposal may assist but never approves. This is
        the only path to APPROVED, it requires a named human, and it will not
        run while a blocking issue stands.
        """
        if not reviewer:
            raise CalibrationError("approval requires a named reviewer")
        if self.approval_state not in (SUBMITTED, DRAFT):
            raise CalibrationError(
                f"cannot approve from state {self.approval_state!r}")
        self.reviewer = reviewer
        blocking = self.blocking_issues
        if blocking:
            raise CalibrationError(
                "approval refused; resolve these first:\n  "
                + "\n  ".join(str(i) for i in blocking))
        self.approval_state = APPROVED
        self.approved_utc = when_utc

    def reject(self, reviewer: str, reason: str) -> None:
        self.reviewer = reviewer
        self.approval_state = REJECTED
        self.confidence_notes = (self.confidence_notes + "\n" if self.confidence_notes
                                 else "") + f"rejected: {reason}"

    def revise(self) -> "CalibrationProfile":
        """A new version carrying this one's identity.

        PRD 7 requires that modifying a calibration creates a new version and
        that old events are never silently rewritten. Editing in place would
        make every event produced under the old geometry unexplainable.
        """
        child = CalibrationProfile(
            camera_id=self.camera_id, epoch_id=self.epoch_id,
            version=self.version + 1, frame_w=self.frame_w, frame_h=self.frame_h,
            source_sha256=self.source_sha256,
            reference_pts_ms=self.reference_pts_ms,
            operator=self.operator, approval_state=DRAFT,
            parent_version=self.version,
            seats=[Seat(s.seat_id, {k: list(v) for k, v in s.zones.items()},
                        s.desk_line_y, list(s.adjacent), list(s.ambiguous_with),
                        s.notes) for s in self.seats],
            masks=[MaskRegion(m.kind, list(m.polygon), m.note) for m in self.masks],
            review_images=dict(self.review_images),
            confidence_notes=self.confidence_notes,
            proposal_source=self.proposal_source,
        )
        self.approval_state = SUPERSEDED
        return child

    # ---------------------------------------------------------------- io --

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k not in ("seats", "masks")},
            "seats": [asdict(s) for s in self.seats],
            "masks": [asdict(m) for m in self.masks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationProfile":
        seats = [Seat(**{**s, "zones": {k: [tuple(p) for p in v]
                                        for k, v in s["zones"].items()}})
                 for s in data.get("seats", [])]
        masks = [MaskRegion(m["kind"], [tuple(p) for p in m["polygon"]],
                            m.get("note", "")) for m in data.get("masks", [])]
        rest = {k: v for k, v in data.items() if k not in ("seats", "masks")}
        return cls(**rest, seats=seats, masks=masks)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "CalibrationProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------- geometry --

def polygon_area(polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    pts = np.asarray(polygon, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def fill(polygon, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
    return mask


def polygon_overlap_px(a, b, w: int, h: int) -> float:
    """Shared area in pixels, ignoring a shared edge.

    `fillPoly` includes the boundary, so two seats drawn flush against each
    other intersect along that edge and would be reported as overlapping. Both
    masks are eroded by one pixel first, which removes the shared edge and keeps
    any genuine overlap. Measured on real calibration: this was a false positive
    that blocked otherwise valid profiles.
    """
    kernel = np.ones((3, 3), np.uint8)
    ma = cv2.erode(fill(a, w, h), kernel, iterations=1)
    mb = cv2.erode(fill(b, w, h), kernel, iterations=1)
    return float((ma & mb).sum())


def scoring_mask(profile: CalibrationProfile) -> np.ndarray:
    """1 where candidate motion may be scored, 0 where a mask excludes it."""
    allowed = np.ones((profile.frame_h, profile.frame_w), dtype=np.uint8)
    for mask in profile.masks:
        if mask.kind in EXCLUDED_FROM_SCORING:
            allowed &= 1 - fill(mask.polygon, profile.frame_w, profile.frame_h)
    return allowed


def background_support(profile: CalibrationProfile) -> np.ndarray:
    """Where global alignment may look for static structure (PRD 6.3.1).

    The declared static-background regions, minus every seat, aisle, monitor,
    reflection and environment region. Alignment fitted on pixels containing
    people measures the people, not the camera.
    """
    w, h = profile.frame_w, profile.frame_h
    support = np.zeros((h, w), dtype=np.uint8)
    for mask in profile.masks_of(STATIC_BACKGROUND):
        support |= fill(mask.polygon, w, h)
    for seat in profile.seats:
        for polygon in seat.zones.values():
            support &= 1 - fill(polygon, w, h)
    for mask in profile.masks:
        if mask.kind in EXCLUDED_FROM_SCORING:
            support &= 1 - fill(mask.polygon, w, h)
    return support


# ---------------------------------------------------------- attribution --

@dataclass
class Attribution:
    """Which seat a box belongs to, or an explicit refusal to say."""

    seat_id: str | None
    state: str                      # a seat id, AMBIGUOUS_SEAT, or UNATTRIBUTED
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.seat_id is not None


@dataclass
class AttributionConfig:
    # A candidate seat must contain at least this share of the box's lower
    # region before it is considered at all.
    min_overlap: float = 0.15
    # The best candidate must beat the runner-up by this margin, or the answer
    # is ambiguous. PRD 7 forbids picking the higher score when the evidence
    # does not separate them.
    min_margin: float = 0.15
    # The fraction of a person box treated as their seated footprint. A seated
    # person's box includes head and shoulders that lean well outside the seat,
    # so the bottom third is what actually indicates where they are sitting.
    bottom_fraction: float = 0.33


def attribute_box(box: tuple[float, float, float, float],
                  profile: CalibrationProfile,
                  cfg: AttributionConfig | None = None) -> Attribution:
    """Attribute a person box to a seat, or refuse.

    Uses the overlap of the box's **lower** region with each seat footprint, not
    the box centre. PRD 7: never assign by nearest box centre alone. A leaning
    candidate's centre routinely sits over a neighbour while their seated
    footprint does not move at all.
    """
    cfg = cfg or AttributionConfig()
    if not profile.approved:
        return Attribution(
            None, UNATTRIBUTED,
            reason=f"calibration {profile.version_id} is {profile.approval_state}, "
                   f"not approved; seat attribution is blocked (PRD 3)")

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return Attribution(None, UNATTRIBUTED, reason="degenerate box")

    lower_y1 = y2 - (y2 - y1) * cfg.bottom_fraction
    footprint = fill([(x1, lower_y1), (x2, lower_y1), (x2, y2), (x1, y2)],
                     profile.frame_w, profile.frame_h)
    total = float(footprint.sum())
    if total <= 0:
        return Attribution(None, UNATTRIBUTED, reason="box falls outside the frame")

    scores: dict[str, float] = {}
    for seat in profile.seats:
        if not seat.polygon:
            continue
        seat_mask = fill(seat.polygon, profile.frame_w, profile.frame_h)
        share = float((footprint & seat_mask).sum()) / total
        if share >= cfg.min_overlap:
            scores[seat.seat_id] = round(share, 4)

    if not scores:
        return Attribution(None, UNATTRIBUTED, scores={},
                           reason="no seat overlaps the box footprint")

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_id, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    # An explicitly declared ambiguous boundary is ambiguous by construction,
    # regardless of how the numbers fall.
    if len(ranked) > 1:
        seat = profile.seat(best_id)
        second_id = ranked[1][0]
        if seat and second_id in seat.ambiguous_with:
            return Attribution(
                None, AMBIGUOUS_SEAT, scores,
                reason=f"{best_id} and {second_id} share a boundary the operator "
                       f"marked unresolvable")

    if best - runner_up < cfg.min_margin:
        return Attribution(
            None, AMBIGUOUS_SEAT, scores,
            reason=f"{best_id} leads {ranked[1][0]} by {best - runner_up:.3f}, "
                   f"under the {cfg.min_margin} margin")

    return Attribution(best_id, best_id, scores,
                       reason=f"footprint overlap {best:.3f}")
