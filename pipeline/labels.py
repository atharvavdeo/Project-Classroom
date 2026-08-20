"""Every label the system can emit, in one registry, for UI and API.

A frontend that hardcodes its own strings drifts from the pipeline the first
time a class is renamed, and the drift is silent: the UI shows a stale label
next to a live number. So every class, state, event kind and disposition is
declared once here, with the display name, severity and colour the UI should
use, and exported as JSON by `tools/export_labels.py`.

### Severity is about reviewer attention, not guilt

`severity` orders the reviewer's queue. It never means "how likely is this
cheating" -- nothing in this system estimates that, and a UI that renders
severity as a guilt meter would be misrepresenting the pipeline. The wording of
every `description` is deliberately observational for the same reason: these
strings are what an investigator reads, and PS #1 requires alerts that
"highlight specific observed behaviors rather than making direct accusations of
cheating".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

INFO = "info"              # context, not a finding
LOW = "low"
MEDIUM = "medium"
HIGH = "high"
BLOCKED = "blocked"        # withheld from the queue by a gate
SEVERITIES = (BLOCKED, INFO, LOW, MEDIUM, HIGH)


@dataclass(frozen=True)
class Label:
    key: str
    display: str
    group: str
    severity: str
    colour: str
    description: str

    def to_row(self) -> dict:
        return asdict(self)


def _L(key, display, group, severity, colour, description) -> Label:
    return Label(key, display, group, severity, colour, description)


# --------------------------------------------------------- object classes --
OBJECT_LABELS = (
    _L("phone", "Mobile phone", "object", HIGH, "#ef4444",
       "A detector reported a phone-like object at this location and time."),
    _L("secondary_paper_chit", "Paper / chit", "object", HIGH, "#f97316",
       "Reported by the detector's paper class. On this corpus that class "
       "fires on answer sheets and printed material as well as on notes."),
    _L("book_or_notebook_like_object", "Book or notebook", "object", MEDIUM,
       "#f59e0b",
       "A bound or printed paper object. Exam material and prohibited "
       "material look alike at this resolution."),
    _L("stationery", "Stationery", "object", LOW, "#eab308",
       "Pen, pencil or similar. Expected in an examination."),
    _L("answer_sheet", "Answer sheet", "object", INFO, "#84cc16",
       "Expected examination material."),
    _L("calculator", "Calculator", "object", LOW, "#a3e635",
       "May be permitted or prohibited depending on the examination."),
    _L("bag", "Bag", "object", LOW, "#22d3ee",
       "A bag within reach of the candidate."),
    _L("monitor_or_bezel", "Monitor", "object", INFO, "#64748b",
       "Fixed equipment. Counted as a hard negative for phone detection: a "
       "dark rectangle is the shape a phone detector confuses most."),
    _L("keyboard", "Keyboard", "object", INFO, "#64748b",
       "Fixed equipment, and this detector's most frequent false call on "
       "desk texture."),
    _L("mouse", "Mouse", "object", INFO, "#64748b", "Fixed equipment."),
    _L("hand", "Hand", "object", INFO, "#94a3b8", "An observed hand."),
    _L("unknown_object", "Unclassified object", "object", INFO, "#94a3b8",
       "Something was detected but matched no class in the taxonomy."),
    _L("insufficient_visual_evidence", "Insufficient evidence", "object",
       BLOCKED, "#475569",
       "The source region was below the resolution floor, so no call was "
       "made. An abstention is a result, not a failure."),
)

# --------------------------------------------- object confirmation status --
OBJECT_STATUS_LABELS = (
    _L("temporally_supported_object_evidence", "Confirmed across frames",
       "object_status", HIGH, "#ef4444",
       "The same object was found at the same place in several frames."),
    _L("single_frame_object_candidate", "Single frame only", "object_status",
       LOW, "#f59e0b",
       "Seen in one frame. Retained, not discarded: a singleton is a status."),
    _L("conflicting_object_evidence", "Conflicting", "object_status", MEDIUM,
       "#f97316",
       "Frames disagreed about what the object is. Conflict is reported, "
       "never silently resolved to the majority."),
    _L("insufficient_visual_evidence", "Insufficient", "object_status",
       BLOCKED, "#475569", "Below the resolution floor."),
)

# ------------------------------------------------------- head and posture --
FACING_LABELS = (
    _L("frontal", "Facing forward", "facing", INFO, "#22c55e",
       "Head oriented toward the candidate's own workstation."),
    _L("turned_left", "Turned left", "facing", MEDIUM, "#f59e0b",
       "Head yaw to the candidate's own left, beyond their resting posture."),
    _L("turned_right", "Turned right", "facing", MEDIUM, "#f59e0b",
       "Head yaw to the candidate's own right, beyond their resting posture."),
    _L("downward", "Looking down", "facing", LOW, "#eab308",
       "Head pitched down. Normal while writing or reading."),
    _L("not_resolvable_at_source_scale", "Not resolvable", "facing", BLOCKED,
       "#475569",
       "The head was too small in the source for any orientation claim."),
)

EVENT_LABELS = (
    _L("look_down", "Prolonged look down", "head_event", MEDIUM, "#eab308",
       "Head held below this candidate's own resting pitch for a sustained "
       "period. Expected during written work; length is what matters."),
    _L("yaw_left", "Sustained turn left", "head_event", MEDIUM, "#f59e0b",
       "Head held turned to the candidate's left beyond their own baseline."),
    _L("yaw_right", "Sustained turn right", "head_event", MEDIUM, "#f59e0b",
       "Head held turned to the candidate's right beyond their own baseline."),
    _L("look_up", "Look up", "head_event", LOW, "#a3e635",
       "Head raised above resting pitch. PS #1 names raising the head while "
       "thinking as normal behaviour."),
    _L("over_shoulder", "Over-shoulder turn", "head_event", HIGH, "#ef4444",
       "Yaw far enough that one ear is occluded by the turn: the candidate "
       "is looking behind, not merely aside."),
    _L("repeated_same_direction", "Repeated glances, one direction",
       "head_event", HIGH, "#ef4444",
       "Several sustained turns the same way within a short window. This is "
       "the pattern that separates a stretch from sustained attention "
       "elsewhere."),
    _L("toward_occupied_neighbour", "Turn toward an occupied seat",
       "head_event", HIGH, "#ef4444",
       "The direction of the turn points at an adjacent seat that a person "
       "occupies, rather than at an aisle, a wall or a partition."),
    _L("lean_toward_neighbour", "Lean toward neighbour", "posture", HIGH,
       "#ef4444",
       "Torso displaced toward an adjacent occupied seat beyond this "
       "candidate's own resting position."),
    _L("left_seat", "Away from seat", "posture", MEDIUM, "#f97316",
       "No person was associated with this seat for a sustained period."),
    _L("returned_to_seat", "Returned to seat", "posture", INFO, "#22c55e",
       "A person was associated with this seat again after an absence."),
    _L("seat_change", "Seat change", "posture", HIGH, "#ef4444",
       "A track that was associated with one seat became associated with a "
       "different seat. Requires an approved calibration."),
)

# ------------------------------------------------------------ observability --
OBSERVABILITY_LABELS = (
    _L("visible", "Visible", "observability", INFO, "#22c55e",
       "The keypoint was resolved."),
    _L("occluded", "Occluded", "observability", INFO, "#f59e0b",
       "Something blocked it. On this corpus that is usually a monitor, a "
       "chair back or a partition."),
    _L("not_detected", "Not detected", "observability", INFO, "#94a3b8",
       "The model returned nothing here."),
    _L("not_resolvable_at_source_scale", "Below source resolution",
       "observability", BLOCKED, "#475569",
       "Too few source pixels for a claim. Distinct from 'not detected': the "
       "model was never entitled to answer."),
)

# ------------------------------------------------------------ dispositions --
DISPOSITION_LABELS = (
    _L("high_potential_corroborated", "High potential", "disposition", HIGH,
       "#dc2626",
       "Object evidence confirmed across frames AND sustained pose context "
       "AND a resolved seat AND no quality or geometry penalty. Four "
       "independent things agree. Open first."),
    _L("flagged_for_review", "Flagged for review", "disposition", MEDIUM,
       "#f59e0b",
       "Partial corroboration, or among the strongest motion in this "
       "recording. The working queue."),
    _L("retained_low_priority", "Retained", "disposition", LOW, "#64748b",
       "Real but weak. Searchable, not queued."),
    _L("rejected_not_offered", "Withheld", "disposition", BLOCKED, "#475569",
       "A quality gate withheld this from the queue. It is NOT a finding "
       "that nothing happened: the evidence keeps its score and stays "
       "searchable."),
)

# ------------------------------------------------------------ health states --
HEALTH_LABELS = (
    _L("blur", "Blurred", "health", INFO, "#f59e0b", "Reduced sharpness."),
    _L("exposure_change", "Exposure change", "health", INFO, "#f59e0b",
       "Illumination shifted; motion estimates over this span are suspect."),
    _L("compression_noise", "Compression noise", "health", INFO, "#f59e0b",
       "Blocking artefacts at a level that can imitate small-object detail."),
    _L("whiteout", "Whiteout", "health", BLOCKED, "#475569", "Saturated."),
    _L("blackout", "Blackout", "health", BLOCKED, "#475569", "Near black."),
    _L("frozen_video", "Frozen video", "health", BLOCKED, "#475569",
       "Consecutive identical frames; the recording stalled."),
    _L("camera_motion_uncertain", "Camera motion uncertain", "health", INFO,
       "#f97316",
       "Global motion could not be compensated to the required residual, so "
       "raw evidence is retained and rank is penalised."),
    _L("environmental_motion", "Environmental motion", "health", INFO,
       "#94a3b8", "Fans, curtains or lighting, not a person."),
)

ALL_GROUPS = {
    "object": OBJECT_LABELS,
    "object_status": OBJECT_STATUS_LABELS,
    "facing": FACING_LABELS,
    "head_event": EVENT_LABELS,
    "observability": OBSERVABILITY_LABELS,
    "disposition": DISPOSITION_LABELS,
    "health": HEALTH_LABELS,
}

BY_KEY = {label.key: label
          for group in ALL_GROUPS.values() for label in group}


def describe(key: str) -> Label | None:
    return BY_KEY.get(key)


def registry() -> dict:
    """The whole registry, shaped for a frontend to consume directly."""
    return {
        "severities": list(SEVERITIES),
        "groups": {name: [label.to_row() for label in labels]
                   for name, labels in ALL_GROUPS.items()},
        "severity_note":
            "severity orders reviewer attention. It is not a probability that "
            "misconduct occurred, and no part of this system estimates one. A "
            "UI must not render it as a guilt score.",
        "not_measured": [
            "eye gaze or iris direction -- impossible at 4.7-10.1 px "
            "interocular distance on this corpus",
            "face identity -- never computed, by design",
            "any assertion that a candidate cheated",
        ],
    }
