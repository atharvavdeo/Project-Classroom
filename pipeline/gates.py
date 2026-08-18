"""The five-state gate protocol (PRD 8).

**Gates route evidence; they do not erase it.** That single sentence is the
whole design. Every other video-analysis pipeline this replaces had a boolean
somewhere that discarded a window, and the discarded windows were invisible
afterwards — you could not tell a span that was examined and dismissed from a
span that was never examined at all.

So there is no boolean. Every decision produces a record with a state, the
inputs and scores that produced it, the thresholds and config that judged it,
a reason in words, and pointers to its artifacts:

    pass                        adequate reliability; route and rank normally
    deprioritise                weak or environmental but usable; retain, penalise
    uncertain                   conflicting or insufficient classification;
                                retain and expose for review or reprocessing
    suppress_from_auto_ranking  a known integrity issue makes automatic ranking
                                unreliable; retain and keep searchable, exclude
                                only from automatic priority
    failed                      data unavailable or corrupt; preserve the gap and
                                claim no analysis

The distinction between `suppress_from_auto_ranking` and `failed` is the one
that is easy to collapse and expensive to get wrong. Suppressed evidence exists
and is searchable — a reviewer can find it deliberately. Failed evidence does
not exist, and the honest output is a hole with a reason attached, never an
interpolation across it.

`GateAudit` measures what PRD 8 requires: correct and false passes, correct and
false suppressions, uncertain retention and failed coverage. A gate that has
never been audited against references is a gate nobody has checked, and its
numbers say so rather than defaulting to a flattering zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# ----------------------------------------------------------------- states --

PASS = "pass"
DEPRIORITISE = "deprioritise"
UNCERTAIN = "uncertain"
SUPPRESS_FROM_AUTO_RANKING = "suppress_from_auto_ranking"
FAILED = "failed"

STATES = (PASS, DEPRIORITISE, UNCERTAIN, SUPPRESS_FROM_AUTO_RANKING, FAILED)

# States whose evidence still reaches automatic ranking.
RANKABLE = frozenset({PASS, DEPRIORITISE})

# States whose evidence is retained and searchable. Everything except `failed`,
# because failed evidence is absent rather than demoted.
RETAINED = frozenset({PASS, DEPRIORITISE, UNCERTAIN, SUPPRESS_FROM_AUTO_RANKING})

# Ranking multiplier per state. `deprioritise` keeps evidence in the list and
# below anything unpenalised; it does not remove it.
PENALTY = {
    PASS: 1.0,
    DEPRIORITISE: 0.35,
    UNCERTAIN: 0.0,                   # retained, but never auto-ranked
    SUPPRESS_FROM_AUTO_RANKING: 0.0,
    FAILED: 0.0,
}


class GateError(ValueError):
    """A gate record is malformed."""


@dataclass
class GateRecord:
    """One gate decision, with everything needed to re-derive it (PRD 8)."""

    gate: str                          # which gate made this decision
    condition: str                     # what it was judging
    state: str
    start_pts_ms: float
    end_pts_ms: float
    reason: str
    inputs: dict = field(default_factory=dict)      # the measured values
    thresholds: dict = field(default_factory=dict)  # what they were judged against
    config_id: str = ""
    artifacts: list[str] = field(default_factory=list)
    seat_id: str | None = None

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise GateError(
                f"unknown gate state {self.state!r}; expected one of {STATES}")
        if self.end_pts_ms < self.start_pts_ms:
            raise GateError(
                f"{self.gate}: interval ends before it starts "
                f"({self.start_pts_ms} -> {self.end_pts_ms})")
        if not self.reason:
            raise GateError(
                f"{self.gate}: every gate decision must carry a reason in words. "
                f"A state with no explanation cannot be reviewed or appealed.")

    @property
    def duration_ms(self) -> float:
        return self.end_pts_ms - self.start_pts_ms

    @property
    def retained(self) -> bool:
        return self.state in RETAINED

    @property
    def rankable(self) -> bool:
        return self.state in RANKABLE

    @property
    def penalty(self) -> float:
        return PENALTY[self.state]

    def overlaps(self, start_ms: float, end_ms: float) -> bool:
        return self.start_pts_ms < end_ms and self.end_pts_ms > start_ms

    def to_row(self) -> dict:
        return {
            **asdict(self),
            "duration_ms": round(self.duration_ms, 3),
            "retained": self.retained,
            "rankable": self.rankable,
            "penalty": self.penalty,
        }


class GateLog:
    """Every gate decision for one run, and the questions asked of them."""

    def __init__(self) -> None:
        self.records: list[GateRecord] = []

    def add(self, record: GateRecord) -> GateRecord:
        self.records.append(record)
        return record

    def route(self, gate: str, condition: str, state: str, start_ms: float,
              end_ms: float, reason: str, **kw) -> GateRecord:
        return self.add(GateRecord(gate, condition, state, start_ms, end_ms,
                                   reason, **kw))

    def over(self, start_ms: float, end_ms: float) -> list[GateRecord]:
        return [r for r in self.records if r.overlaps(start_ms, end_ms)]

    def worst(self, start_ms: float, end_ms: float) -> str:
        """The most restrictive state applying to an interval.

        Ordered by how much they restrict use, not by severity of the underlying
        problem: `failed` means the data is absent, which restricts more than
        `suppress_from_auto_ranking`, which restricts more than `uncertain`.
        """
        order = [FAILED, SUPPRESS_FROM_AUTO_RANKING, UNCERTAIN, DEPRIORITISE, PASS]
        states = {r.state for r in self.over(start_ms, end_ms)}
        for state in order:
            if state in states:
                return state
        return PASS

    def multiplier(self, start_ms: float, end_ms: float) -> float:
        """Combined ranking penalty for an interval.

        Multiplicative rather than a minimum: two independent weak signals over
        the same span are weaker than either alone, and a single hard stop takes
        it to zero regardless of what else applies.
        """
        value = 1.0
        for record in self.over(start_ms, end_ms):
            value *= record.penalty
        return value

    def counts(self) -> dict[str, int]:
        out = {state: 0 for state in STATES}
        for record in self.records:
            out[record.state] += 1
        return out

    def coverage_ms(self, state: str) -> float:
        """Union of time in one state. Overlapping records count once."""
        intervals = sorted((r.start_pts_ms, r.end_pts_ms)
                           for r in self.records if r.state == state)
        if not intervals:
            return 0.0
        total, current_start, current_end = 0.0, *intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end
        return total + (current_end - current_start)

    def to_rows(self) -> list[dict]:
        return [r.to_row() for r in self.records]

    def summary(self, total_ms: float | None = None) -> dict:
        out = {
            "records": len(self.records),
            "by_state": self.counts(),
            "coverage_ms": {s: round(self.coverage_ms(s), 3) for s in STATES},
            "retained_records": sum(1 for r in self.records if r.retained),
            "discarded_records": 0,
        }
        if total_ms:
            out["failed_fraction"] = round(self.coverage_ms(FAILED) / total_ms, 5)
        return out


# ------------------------------------------------------------- gate audit --

@dataclass
class AuditOutcome:
    """One reference interval, and what the gates did with it."""

    reference_id: str
    start_pts_ms: float
    end_pts_ms: float
    relevant: bool               # does a reviewer consider this worth surfacing
    state: str
    correct: bool
    note: str = ""


class GateAudit:
    """Measures whether the gates routed correctly (PRD 8, PRD 15).

    Four quantities, named as PRD 8 names them:

      * **correct pass** — relevant evidence that reached ranking
      * **false pass** — irrelevant evidence that reached ranking
      * **correct suppression** — irrelevant evidence held back
      * **false suppression** — relevant evidence held back, which is the
        expensive error and the one this whole protocol exists to make visible

    Plus uncertain retention and failed coverage. With no references the audit
    reports `unaudited` rather than a flattering zero: a gate nobody has checked
    is not a gate with no errors.
    """

    def __init__(self, log: GateLog):
        self.log = log
        self.outcomes: list[AuditOutcome] = []

    def judge(self, reference_id: str, start_ms: float, end_ms: float,
              relevant: bool, note: str = "") -> AuditOutcome:
        state = self.log.worst(start_ms, end_ms)
        reached_ranking = state in RANKABLE
        # A relevant reference is handled correctly when it reaches ranking; an
        # irrelevant one when it does not.
        correct = reached_ranking if relevant else not reached_ranking
        outcome = AuditOutcome(reference_id, start_ms, end_ms, relevant, state,
                               correct, note)
        self.outcomes.append(outcome)
        return outcome

    @property
    def audited(self) -> bool:
        return bool(self.outcomes)

    def metrics(self, total_ms: float | None = None) -> dict:
        if not self.audited:
            return {
                "status": "unaudited",
                "detail": "no reference intervals were supplied, so gate "
                          "routing has not been checked. This is not a "
                          "zero-error result.",
                "failed_coverage_ms": round(self.log.coverage_ms(FAILED), 3),
                "uncertain_retained": self.log.counts()[UNCERTAIN],
            }

        relevant = [o for o in self.outcomes if o.relevant]
        irrelevant = [o for o in self.outcomes if not o.relevant]
        correct_pass = sum(1 for o in relevant if o.state in RANKABLE)
        false_suppression = len(relevant) - correct_pass
        false_pass = sum(1 for o in irrelevant if o.state in RANKABLE)
        correct_suppression = len(irrelevant) - false_pass

        out = {
            "status": "audited",
            "references": len(self.outcomes),
            "correct_pass": correct_pass,
            "false_pass": false_pass,
            "correct_suppression": correct_suppression,
            "false_suppression": false_suppression,
            # The headline. A gate that suppresses relevant evidence is failing
            # at the one job it must not fail.
            "false_suppression_rate": round(false_suppression / len(relevant), 4)
            if relevant else None,
            "false_pass_rate": round(false_pass / len(irrelevant), 4)
            if irrelevant else None,
            "uncertain_retained": self.log.counts()[UNCERTAIN],
            "failed_coverage_ms": round(self.log.coverage_ms(FAILED), 3),
            "by_state": self.log.counts(),
        }
        if total_ms:
            out["failed_coverage_fraction"] = round(
                self.log.coverage_ms(FAILED) / total_ms, 5)
        return out

    def failures(self) -> list[AuditOutcome]:
        """Every reference the gates got wrong, for rendering."""
        return [o for o in self.outcomes if not o.correct]
