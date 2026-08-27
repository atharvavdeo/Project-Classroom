"""The classification ladder: from measurements to a reviewable verdict.

Implements the specification in `docs/CLASSIFICATION_AND_OPEN_QUESTIONS.md`.
Read that first -- this module is the code, that document is the argument.

### The ladder

    0  observation        a measurement was taken                  not shown
    1  signal             departs from this person's own baseline  not shown alone
    2  indicator          survives every gate                      AMBER card
    3  incident_candidate two INDEPENDENT indicators coincide      RED card
    4  confirmed_incident a human watched it and agreed            human only

**Nothing here may ever write rung 4.** An automated system that declares a
candidate a cheat creates a liability we cannot discharge, and our own
measurements say why: the strongest object signal we have fired on seat
placards, shirt fabric and 368 keyboards before it was refereed.

### The independence rule, and why it is the whole design

Two indicators count as independent only if they come from **different sensing
modalities**. Two chit detections 250 ms apart are one indicator, not two. A
sustained head turn plus an object at the hands is two.

Without that rule, rung 3 degenerates into "the detector fired a lot", which is
exactly the failure already measured: 1,817 chit detections on one 88-second
video, 972 of them nowhere near a hand.

The modalities available today:

    MODALITY_ORIENTATION   AlphaPose facial keypoints -> head yaw/pitch
    MODALITY_OBJECT        chit detector + SAM 3 referee, at the wrists
    MODALITY_PRESENCE      tracking continuity: absences, coverage

Note what is deliberately *not* a separate modality: the chit detector and SAM 3
are both object evidence, so their agreement strengthens **one** indicator
rather than creating two. That agreement is still the most valuable signal we
have -- on 12_paper it put 133 of 136 corroborations on one man -- but it raises
confidence in the object indicator, it does not escalate on its own.

### Guarding

Every escalation records the figure that produced it, and every abstention
records why it abstained. A card that cannot say why it is a card does not get
shown. `low_confidence_reasons` is not decoration: a reviewer who cannot see
that a person's head was never resolvable will read "no orientation signal" as
"behaved normally", which is the opposite of what the data says.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Rungs.
OBSERVATION = "observation"
SIGNAL = "signal"
INDICATOR = "indicator"
INCIDENT_CANDIDATE = "incident_candidate"
CONFIRMED = "confirmed_incident"        # written by humans, never by this code

RUNG_ORDER = {OBSERVATION: 0, SIGNAL: 1, INDICATOR: 2,
              INCIDENT_CANDIDATE: 3, CONFIRMED: 4}

# Sensing modalities. Independence is defined across these, not across detections.
MODALITY_ORIENTATION = "head_orientation"
MODALITY_OBJECT = "object_at_hands"
MODALITY_PRESENCE = "presence"


@dataclass
class Guard:
    """Every threshold that can promote or block a finding.

    Each carries its provenance in the comment: `measured` means it came from
    this corpus, `declared` means it is a policy choice nobody has validated
    yet. A reader must be able to tell those apart without leaving the file.
    """

    # --- orientation (B8/B9) -------------------------------------------------
    # measured: matches HeadPoseConfig.min_departure_from_baseline, tuned when
    # an absolute threshold flagged 45 resting postures on video 04.
    away_fraction_indicator: float = 0.40
    # declared: brief turns are universal, so a person must spend a real share
    # of their observed time turned away before it means anything.
    away_fraction_strong: float = 0.60
    # declared: four episodes in the recording, same direction dominant.
    repeat_episodes_indicator: int = 4

    # --- object at hands (B12/B14) ------------------------------------------
    # measured: on 12_paper the subject's longest continuous handling was 56.5 s
    # and the next-ranked false positive was 5.0 s.
    handling_ms_indicator: float = 5000.0
    handling_ms_strong: float = 20000.0
    # A SAM 3 corroboration is worth more than raw count, because it is a
    # different model with a different vocabulary. measured: 133 of 136
    # corroborations on 12_paper fell on one man.
    min_corroborations_indicator: int = 3
    # Episode sampling caps the number of referee calls, so on a track with few
    # representatives a count floor of 3 silently becomes "all of them must
    # agree". The rate below is an ADDITIONAL way to clear the bar, never a
    # replacement for the count -- see `_corroborated()`.
    #
    # It must stay additive. Measured on run 1510/12_paper, SAM 3's support
    # rate on the two known paper incidents is 5 of 19 (26%) for track 118 and
    # 5 of 29 (17%) for track 53. Requiring a 67% rate *instead of* the count
    # dropped both out of review, along with track 18 on 01_phone. SAM 3 is a
    # conservative referee on this corpus and its support rate on genuine
    # handling is low; a rate-only test therefore deletes true positives.
    min_corroboration_rate_indicator: float = 2 / 3
    min_corroboration_samples_indicator: int = 2
    # A phone is prohibited outright, so a single refereed phone is an
    # indicator on its own -- but only when SAM 3 named it, never when the
    # single-class chit detector merely fired.
    phone_reclassifications_indicator: int = 1

    # --- presence (B7/B15) ---------------------------------------------------
    # declared: an absence long enough to be the person leaving.
    absence_ms_indicator: float = 10000.0

    # --- abstention ----------------------------------------------------------
    # Below this coverage a track is a fragment, not a person, and no verdict
    # above `observation` may be issued for it. measured: the news clip produced
    # 515 tracks for ~22 people, median coverage 1.96%.
    min_coverage: float = 0.05
    # A profile with no resolvable head cannot contribute an orientation
    # indicator. It must say so rather than scoring zero.
    min_resolvable_samples: int = 8


@dataclass
class Indicator:
    """One reason a person is on the list."""

    modality: str
    kind: str
    detail: str           # the figure that produced it, in plain words
    strength: float       # 0..1, for ordering only -- never shown as a probability


@dataclass
class Verdict:
    """What the system is willing to say about one track."""

    track_id: int
    seat: str = "unattributed"
    rung: str = OBSERVATION

    first_seen_ms: float = 0.0
    last_seen_ms: float = 0.0
    coverage: float = 0.0

    indicators: list = field(default_factory=list)
    modalities: list = field(default_factory=list)

    # Why this verdict is weaker than it looks. Always rendered on the card.
    caveats: list = field(default_factory=list)

    # For ranking the queue. Not a probability and must never be shown as one.
    priority: float = 0.0

    @property
    def is_reportable(self) -> bool:
        return RUNG_ORDER[self.rung] >= RUNG_ORDER[INDICATOR]


def corroborated_enough(supported: int, adjudicated: int,
                        guard: Guard | None = None) -> bool:
    """Has SAM 3 backed this object claim enough to count as verified?

    Two independent ways to clear the bar, whichever fires first:

    1. **Count.** `supported >= min_corroborations_indicator`. This is the
       original rule and it still carries the well-adjudicated tracks.
    2. **Rate.** On a track whose episodes yielded only a handful of referee
       calls, the count floor silently becomes "all of them must agree". A
       high support rate over a small but non-trivial denominator clears it
       instead.

    The second is deliberately an OR, so this function can only ever be more
    permissive than the count alone. Making it an AND -- or replacing the
    count with the rate -- deletes known true positives: SAM 3 supported only
    26% of adjudications on 12_paper track 118 and 17% on track 53, both of
    which are real paper handling.

    `fusion.py` and this module must agree on this, so both call it here.
    """
    guard = guard or Guard()
    if adjudicated <= 0:
        return False
    if supported >= guard.min_corroborations_indicator:
        return True
    return (adjudicated >= guard.min_corroboration_samples_indicator
            and supported / adjudicated >= guard.min_corroboration_rate_indicator)


def _orientation(profile: dict, guard: Guard) -> tuple:
    """Indicators and caveats from head orientation."""
    indicators, caveats = [], []
    resolvable = int(profile.get("resolvable_samples") or 0)
    if resolvable < guard.min_resolvable_samples:
        caveats.append(
            f"head orientation never resolved ({resolvable} usable samples) — "
            f"this person's gaze was not measured, not measured-as-normal")
        return indicators, caveats

    away = float(profile.get("fraction_away_from_baseline") or 0.0)
    if away >= guard.away_fraction_indicator:
        indicators.append(Indicator(
            MODALITY_ORIENTATION, "sustained_attention_away",
            f"turned away from their own resting posture in {away:.0%} of "
            f"the frames where their head could be measured",
            min(away / max(guard.away_fraction_strong, 1e-6), 1.0)))

    left = int(profile.get("samples_turned_left") or 0)
    right = int(profile.get("samples_turned_right") or 0)
    if min(left, right) >= guard.repeat_episodes_indicator:
        indicators.append(Indicator(
            MODALITY_ORIENTATION, "repeated_orientation_change",
            f"turned both ways repeatedly ({left} left, {right} right)",
            0.5))
    return indicators, caveats


def _object(evidence: dict | None, sam3: dict | None, guard: Guard) -> tuple:
    """Indicators and caveats from object evidence.

    The chit detector and SAM 3 are one modality. SAM 3 raises the *strength*
    of the object indicator and can escalate its kind to a phone; it does not
    create a second, independent indicator.
    """
    indicators, caveats = [], []
    if not evidence:
        return indicators, caveats

    corroborated = int((sam3 or {}).get("corroborated", 0))
    phones = int((sam3 or {}).get("reclassified_as_phone", 0))
    suppressed = int((sam3 or {}).get("suppressed", 0))
    unsupported = int((sam3 or {}).get("unsupported", 0))
    adjudicated = corroborated + phones + suppressed + unsupported

    if phones >= guard.phone_reclassifications_indicator:
        indicators.append(Indicator(
            MODALITY_OBJECT, "phone_handled",
            f"SAM 3 identified a mobile phone in their hands "
            f"({phones} frame(s)) — a phone is prohibited outright",
            1.0))
    elif corroborated_enough(corroborated, adjudicated, guard):
        longest = float(evidence.get("longest_episode_ms") or 0.0)
        indicators.append(Indicator(
            MODALITY_OBJECT, "object_handled_corroborated",
            f"handled a small pale object; a second model (SAM 3) "
            f"independently agreed in {corroborated} of {adjudicated} "
            f"sampled frame(s), longest "
            f"continuous stretch {longest / 1000:.1f}s",
            min(corroborated / 20.0, 1.0)))
    elif adjudicated and not corroborated and not phones:
        # Adjudicated without support. The wording below is load-bearing and
        # was corrected once already: "confirmed none" reads as a finding of
        # absence, and SAM 3 is not entitled to one. It missed known true
        # positives on this corpus (frames 2 and 5 of the reference set), so
        # `unsupported` means *not confirmed*, never *not there*. Only the
        # equipment suppressions are a positive statement, and only about the
        # detections they overlap.
        if suppressed:
            caveats.append(
                f"a second model (SAM 3) named {suppressed} of their "
                f"{adjudicated} object detections as workstation equipment "
                f"(keyboard, mouse, monitor)")
        if unsupported:
            caveats.append(
                f"SAM 3 could not confirm {unsupported} of their object "
                f"detections either way — unverified, not cleared")
        return indicators, caveats
    elif not adjudicated:
        longest = float(evidence.get("longest_episode_ms") or 0.0)
        if longest >= guard.handling_ms_indicator:
            indicators.append(Indicator(
                MODALITY_OBJECT, "object_handled_unrefereed",
                f"handled a small pale object for {longest / 1000:.1f}s "
                f"continuously — NOT independently reviewed",
                0.4))
            caveats.append(
                "object evidence has not been refereed by SAM 3; the "
                "single-class detector cannot tell paper from a keyboard")
    return indicators, caveats


def _presence(profile: dict, guard: Guard) -> tuple:
    indicators, caveats = [], []
    longest = float(profile.get("longest_absence_ms") or 0.0)
    if longest >= guard.absence_ms_indicator:
        indicators.append(Indicator(
            MODALITY_PRESENCE, "left_and_returned",
            f"absent from frame for {longest / 1000:.0f}s and reappeared",
            min(longest / 60000.0, 1.0)))
        caveats.append(
            "re-identification does not exist, so the person who returned is "
            "not verified to be the person who left")
    return indicators, caveats


def assess(profile: dict, evidence: dict | None = None,
           sam3: dict | None = None, guard: Guard | None = None) -> Verdict:
    """Build one verdict from one track's measurements."""
    guard = guard or Guard()
    out = Verdict(
        track_id=int(profile.get("track_id", -1)),
        seat=str(profile.get("seat_state") or "unattributed"),
        first_seen_ms=float(profile.get("first_seen_ms") or 0.0),
        last_seen_ms=float(profile.get("last_seen_ms") or 0.0),
        coverage=float(profile.get("coverage_of_recording") or 0.0),
    )

    if out.coverage < guard.min_coverage:
        out.rung = OBSERVATION
        out.caveats.append(
            f"seen for only {out.coverage:.1%} of the recording — too "
            f"fragmentary to profile; likely one of several tracks for the "
            f"same person")
        return out

    for source in (_orientation(profile, guard),
                   _object(evidence, sam3, guard),
                   _presence(profile, guard)):
        found, notes = source
        out.indicators.extend(found)
        out.caveats.extend(notes)

    out.modalities = sorted({i.modality for i in out.indicators})

    if len(out.modalities) >= 2:
        out.rung = INCIDENT_CANDIDATE
    elif out.indicators:
        out.rung = INDICATOR
    else:
        out.rung = OBSERVATION

    # Priority orders the queue. Modality count dominates deliberately: two
    # weak indicators from different senses beat one strong one from a single
    # sense, because that is what the independence rule is for.
    out.priority = (len(out.modalities) * 10.0
                    + sum(i.strength for i in out.indicators))
    if out.seat == "unattributed":
        out.caveats.append(
            "no seat calibration for this camera, so this is a tracked person, "
            "not a seat number")
    return out


def rank(verdicts: list) -> list:
    """Reportable verdicts, most urgent first."""
    return sorted([v for v in verdicts if v.is_reportable],
                  key=lambda v: (-RUNG_ORDER[v.rung], -v.priority))
