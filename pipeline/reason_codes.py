"""The closed vocabulary of reasons this system is allowed to give.

Every route, suppression and abstention in the pipeline must name one of these
codes. Nothing downstream may invent a reason string, and nothing may overwrite
a code once it has been attached to an evidence record -- codes accumulate, so
the record shows the whole argument, including the parts that argued the other
way.

### Why a closed vocabulary at all

The failure this prevents is the one already measured on this corpus: a run
produced 1,817 paper detections under a single label, `paper_like_object`, with
no field anywhere saying *why* any of them survived or died. When 402 of them
turned out to be keyboards there was no way to ask the artifact "which ones did
you drop, and on what grounds" -- the answer had to be recomputed from scratch.
A code per decision makes that question answerable from the file.

### The `implemented` field is the honest part

Roughly a third of the codes below are `implemented=False`. They are in the
contract because the behaviour is in the operating matrix and the dashboard has
to have somewhere to put it; they are marked because a reviewer must never see
a metric panel that silently reads zero for a thing the pipeline cannot detect.
Zero seat-vacancy events currently means "there is no seat model", not "nobody
left their seat".

### `route` is an argument, not a verdict

A code's `route` is the state it argues *for*. The fusion machine in
`pipeline/fusion.py` weighs every code on a track and decides. A single
`SAM3_SUPPORTED` does not make a review candidate; a single `SAM3_UNAVAILABLE`
does force the track out of any negative finding.
"""
from __future__ import annotations

from dataclasses import dataclass

# Domains. These are the rows of the operating matrix, collapsed to the ones a
# reason can actually belong to.
D_INGEST = "ingest"
D_CALIBRATION = "calibration"
D_TRACKING = "tracking"
D_SEAT = "seat"
D_OBJECT = "object"
D_ORIENTATION = "orientation"
D_HANDS = "hands"
D_PRESENCE = "presence"
D_SYSTEM = "system"
D_POLICY = "policy"

# Routes -- the fusion states a code argues for. Kept as bare strings rather
# than imported from `fusion`, because `fusion` imports this module.
R_NO_ACTION = "no_action"
R_CONTEXT = "context_observation"
R_BETTER_VIEW = "needs_better_view"
R_REVIEW = "review_candidate"


@dataclass(frozen=True)
class Reason:
    code: str
    domain: str
    route: str
    meaning: str        # what the system observed, in words a reviewer reads
    guardrail: str      # the thing this code must never be read as saying
    implemented: bool   # can the pipeline emit this today?


def _r(code, domain, route, meaning, guardrail, implemented=True):
    return Reason(code, domain, route, meaning, guardrail, implemented)


# --- ingest and image quality ------------------------------------------------
VIDEO_FRAME_UNDECODABLE = _r(
    "VIDEO_FRAME_UNDECODABLE", D_INGEST, R_BETTER_VIEW,
    "the frame could not be decoded",
    "a frame that was never read is not a frame in which nothing happened",
    implemented=False)

TARGET_BELOW_RESOLUTION = _r(
    "TARGET_BELOW_RESOLUTION", D_INGEST, R_BETTER_VIEW,
    "the object crop was too small to classify",
    "too small to see is not the same as not there")

# --- calibration and seats ---------------------------------------------------
SEAT_UNCALIBRATED = _r(
    "SEAT_UNCALIBRATED", D_CALIBRATION, R_CONTEXT,
    "this camera has no seat polygons, so the subject is a tracked person "
    "rather than a seat number",
    "an unattributed track must never be reported as a named candidate")

SEAT_ASSOCIATION_AMBIGUOUS = _r(
    "SEAT_ASSOCIATION_AMBIGUOUS", D_SEAT, R_BETTER_VIEW,
    "more than one person could own this object or this seat",
    "never assign to the nearest person when affinity is ambiguous",
    implemented=False)

ROLE_UNCERTAIN = _r(
    "ROLE_UNCERTAIN", D_SEAT, R_CONTEXT,
    "the person is not seated and their role is not established",
    "standing is not evidence of being an invigilator, nor of being a "
    "candidate",
    implemented=False)

# --- tracking ----------------------------------------------------------------
TRACK_FRAGMENT = _r(
    "TRACK_FRAGMENT", D_TRACKING, R_BETTER_VIEW,
    "the track covers too little of the recording to profile",
    "a fragment is probably one of several tracks for the same person; it is "
    "not a person who was briefly present")

TRACK_IDENTITY_UNVERIFIED = _r(
    "TRACK_IDENTITY_UNVERIFIED", D_TRACKING, R_BETTER_VIEW,
    "the track was lost and resumed, and no re-identification exists",
    "the person who reappeared is not verified to be the person who left")

# --- object proposals --------------------------------------------------------
OBJECT_PROPOSAL_PAPER = _r(
    "OBJECT_PROPOSAL_PAPER", D_OBJECT, R_CONTEXT,
    "the paper detector proposed a paper-like object",
    "the detector has one class and cannot tell approved paper from a chit")

OBJECT_PROPOSAL_PHONE = _r(
    "OBJECT_PROPOSAL_PHONE", D_OBJECT, R_CONTEXT,
    "a detector proposed a phone at this person's hands",
    "a proposal is not a finding until SAM 3 verifies it and it is associated "
    "with this person. The proposer today is stock COCO, which fires on mice, "
    "keyboards, paper edges and a hand raised to an ear")

DFINE_OBJECT_CONTEXT = _r(
    "DFINE_OBJECT_CONTEXT", D_OBJECT, R_CONTEXT,
    "stock COCO detection of workstation equipment or paperwork",
    "COCO's `cell phone` class may never route a person to review -- it is "
    "context about the desk, not evidence about the candidate")

OBJECT_AT_WRIST = _r(
    "OBJECT_AT_WRIST", D_OBJECT, R_CONTEXT,
    "the proposed object sits within reach of a resolved wrist",
    "proximity to a hand is not handling, and a keyboard is also at the wrist")

OBJECT_AWAY_FROM_WRIST = _r(
    "OBJECT_AWAY_FROM_WRIST", D_OBJECT, R_NO_ACTION,
    "the proposed object is nowhere near either wrist",
    "this suppresses the proposal, not the person")

OBJECT_TOO_LARGE = _r(
    "OBJECT_TOO_LARGE", D_OBJECT, R_NO_ACTION,
    "the proposal is too large relative to the person to be a handheld object",
    "the size cap is a per-camera figure and does not transfer between "
    "resolutions -- see open question Q7")

OBJECT_NO_WRIST_RESOLVED = _r(
    "OBJECT_NO_WRIST_RESOLVED", D_OBJECT, R_BETTER_VIEW,
    "no wrist was resolvable, so the object could not be associated",
    "an unassociated object is unassessed, not cleared")

OBJECT_STATIONARY_ON_DESK = _r(
    "OBJECT_STATIONARY_ON_DESK", D_OBJECT, R_CONTEXT,
    "the object stayed in place across the observed window",
    "persistence makes it desk furniture; it does not clear a genuine chit "
    "that happens to be lying still")

OBJECT_HANDLING_SUSTAINED = _r(
    "OBJECT_HANDLING_SUSTAINED", D_OBJECT, R_REVIEW,
    "the object was at the hands continuously beyond the handling threshold",
    "duration is one feature; it argues for review, never for a violation")

OBJECT_HANDLING_RECURRENT = _r(
    "OBJECT_HANDLING_RECURRENT", D_OBJECT, R_REVIEW,
    "handling recurred in separate episodes rather than in one burst",
    "episodes are deduplicated object tracks, never repeated frames")

# --- SAM 3 verification ------------------------------------------------------
SAM3_SUPPORTED = _r(
    "SAM3_SUPPORTED", D_OBJECT, R_REVIEW,
    "SAM 3 independently segmented the claimed object at this location",
    "a second model agreeing raises confidence in one object indicator; it is "
    "not a second, independent modality")

SAM3_EQUIPMENT_CONTEXT = _r(
    "SAM3_EQUIPMENT_CONTEXT", D_OBJECT, R_CONTEXT,
    "SAM 3 named the region as workstation equipment -- keyboard, mouse, "
    "monitor or desk",
    "this suppresses the one proposal it overlaps, not every signal on the "
    "person")

SAM3_NOT_CONFIRMED = _r(
    "SAM3_NOT_CONFIRMED", D_OBJECT, R_BETTER_VIEW,
    "SAM 3 returned no segment matching the claim, and no equipment either",
    "NOT CONFIRMED IS NOT FALSE. SAM 3 missed known true positives on this "
    "corpus -- frames 2 and 5 of the reference set. This code may never be "
    "rendered, counted or summarised as `clear`, `safe` or `no cheating`")

SAM3_PHONE_NAMED = _r(
    "SAM3_PHONE_NAMED", D_OBJECT, R_REVIEW,
    "SAM 3 phone-supported the object",
    "still a proposal about an object, not a finding about a person; seat and "
    "hand association are still required")

SAM3_MASK_UNSTABLE = _r(
    "SAM3_MASK_UNSTABLE", D_OBJECT, R_BETTER_VIEW,
    "the segment changed size or shape too much across the temporal window",
    "instability is an abstention, not a rejection",
    implemented=False)

SAM3_UNAVAILABLE = _r(
    "SAM3_UNAVAILABLE", D_SYSTEM, R_BETTER_VIEW,
    "the SAM 3 endpoint errored, timed out, or was not reachable",
    "an outage may never produce a negative verdict; the source proposal "
    "survives untouched")

SAM3_NOT_RUN = _r(
    "SAM3_NOT_RUN", D_SYSTEM, R_BETTER_VIEW,
    "this run was never adjudicated by SAM 3",
    "unrefereed object evidence is unverified, not verified-negative")

# --- orientation -------------------------------------------------------------
ORIENTATION_NOT_ASSESSABLE = _r(
    "ORIENTATION_NOT_ASSESSABLE", D_ORIENTATION, R_BETTER_VIEW,
    "too few frames resolved this person's head to establish a baseline",
    "no baseline means not measured. It must never be summarised as normal "
    "behaviour")

ORIENTATION_AWAY_SUSTAINED = _r(
    "ORIENTATION_AWAY_SUSTAINED", D_ORIENTATION, R_REVIEW,
    "the head was held away from this person's own resting posture for a "
    "sustained share of the frames where it could be measured",
    "orientation is not gaze. At this resolution the system may not say what "
    "the person was looking at, only which way they faced")

ORIENTATION_CHANGE_FREQUENT = _r(
    "ORIENTATION_CHANGE_FREQUENT", D_ORIENTATION, R_CONTEXT,
    "direction changed often relative to this person's own baseline",
    "fidgeting is normal; frequency alone may not escalate")

ORIENTATION_DOWN_PROLONGED = _r(
    "ORIENTATION_DOWN_PROLONGED", D_ORIENTATION, R_CONTEXT,
    "the head was pitched below the resting level for a prolonged period",
    "this is the posture of writing as much as of reading something hidden")

ORIENTATION_TOWARD_NEIGHBOUR = _r(
    "ORIENTATION_TOWARD_NEIGHBOUR", D_ORIENTATION, R_REVIEW,
    "orientation toward an adjacent seat, sustained and repeated",
    "requires seat geometry; a single turn of the head is not this",
    implemented=False)

TALKING_NOT_MEASURABLE = _r(
    "TALKING_NOT_MEASURABLE", D_ORIENTATION, R_BETTER_VIEW,
    "mouth state cannot be resolved from this footage",
    "the system does not detect talking, and must never display a talking "
    "metric reading zero")

# --- hands -------------------------------------------------------------------
HAND_OCCLUDED_LONG = _r(
    "HAND_OCCLUDED_LONG", D_HANDS, R_BETTER_VIEW,
    "hands were unresolvable for an extended period",
    "an observability warning. It escalates only alongside an independent "
    "object or action signal, never on its own")

HAND_BELOW_DESK_LINE = _r(
    "HAND_BELOW_DESK_LINE", D_HANDS, R_CONTEXT,
    "wrists were below the calibrated desk line",
    "hands in the lap is the resting posture of most seated people")

HAND_INTERACTION_RECURRENT = _r(
    "HAND_INTERACTION_RECURRENT", D_HANDS, R_CONTEXT,
    "repeated reaching episodes toward desk, pocket or bag zones",
    "requires zones this system has not calibrated, and writing looks like "
    "this",
    implemented=False)

# --- presence ----------------------------------------------------------------
PRESENCE_GAP = _r(
    "PRESENCE_GAP", D_PRESENCE, R_CONTEXT,
    "the person was absent from frame and later reappeared",
    "track loss, occlusion and walking out of shot are indistinguishable "
    "here; this is not `left the seat`")

SEAT_VACANCY_SUSTAINED = _r(
    "SEAT_VACANCY_SUSTAINED", D_PRESENCE, R_REVIEW,
    "the seat polygon was reliably empty for longer than the policy allows",
    "requires seat calibration and an occupancy model; neither exists yet",
    implemented=False)

# --- policy ------------------------------------------------------------------
POLICY_MATERIAL_PERMITTED = _r(
    "POLICY_MATERIAL_PERMITTED", D_POLICY, R_CONTEXT,
    "the exam policy permits this material",
    "a technical detection cannot override the configured policy",
    implemented=False)


ALL = tuple(v for v in list(globals().values()) if isinstance(v, Reason))
BY_CODE = {r.code: r for r in ALL}

IMPLEMENTED = tuple(r for r in ALL if r.implemented)
DECLARED_ONLY = tuple(r for r in ALL if not r.implemented)


def get(code: str) -> Reason:
    """Look up a code, refusing anything outside the vocabulary.

    Deliberately raises. A typo that silently becomes a new reason string is
    how a closed vocabulary stops being closed.
    """
    try:
        return BY_CODE[code]
    except KeyError:
        raise KeyError(
            f"{code!r} is not in the reason vocabulary. Add it to "
            f"pipeline/reason_codes.py with a route and a guardrail, or use an "
            f"existing code -- reasons may not be invented at the call site."
        ) from None
