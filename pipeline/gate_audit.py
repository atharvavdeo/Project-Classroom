"""Gate audit against references and sampled intervals (PRD 8, PRD 15).

PRD 8 requires the audit to measure four things on references **plus** sampled
normal and disturbance intervals:

> Gate audit must measure correct/false pass, correct/false suppression,
> uncertain retention, and failed coverage on references plus sampled
> normal/disturbance intervals.

The "plus sampled intervals" is the part that is easy to skip and expensive to
lose. Auditing only against references measures the gates on the moments already
known to be interesting, which is precisely the population where a permissive
gate looks good. A gate that passes everything scores a perfect false-suppression
rate on references and is useless. Sampled *normal* intervals -- spans nobody
flagged -- are what reveal it, because a gate that passes those too is passing
everything.

`pipeline.gates.GateAudit` already holds the counting logic and reports
`unaudited` rather than a flattering zero when nothing has been checked. This
module supplies the populations: references from stage 12, and sampled intervals
drawn from the run's own timeline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .gates import GateAudit, GateLog

NORMAL = "sampled_normal"
DISTURBANCE = "sampled_disturbance"
REFERENCE = "reference"


@dataclass
class SampleConfig:
    # Intervals drawn from spans nobody flagged. These are the population that
    # exposes a gate which passes everything.
    normal_samples: int = 40

    # Intervals drawn from spans a health condition covers. These expose the
    # opposite failure: a gate that suppresses whenever anything is imperfect.
    disturbance_samples: int = 20

    interval_ms: float = 2000.0
    seed: int = 20260819


@dataclass
class AuditPopulation:
    """The intervals an audit was run over, and where each came from."""

    intervals: list[tuple[str, str, float, float, bool]] = field(
        default_factory=list)
    # Populated by `sample_intervals` when it could not draw what it asked for.
    shortfall: dict = field(default_factory=dict)

    def add(self, source: str, identifier: str, start_ms: float, end_ms: float,
            relevant: bool) -> None:
        self.intervals.append((source, identifier, start_ms, end_ms, relevant))

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for source, *_ in self.intervals:
            out[source] = out.get(source, 0) + 1
        return out


def sample_intervals(duration_ms: float,
                     disturbance_spans: list[tuple[float, float, str]],
                     reference_times: list[float],
                     cfg: SampleConfig | None = None) -> AuditPopulation:
    """Draw normal and disturbance intervals from the run's own timeline.

    Normal intervals deliberately exclude anything near a reference. An interval
    that happens to contain a real sighting is not a normal interval, and
    counting a pass there as a false pass would penalise the gate for being
    right.
    """
    cfg = cfg or SampleConfig()
    rng = random.Random(cfg.seed)
    population = AuditPopulation()

    def in_disturbance(start: float, end: float) -> str | None:
        for lo, hi, condition in disturbance_spans:
            if start < hi and end > lo:
                return condition
        return None

    def near_reference(start: float, end: float) -> bool:
        return any(start - cfg.interval_ms <= t <= end + cfg.interval_ms
                   for t in reference_times)

    attempts = 0
    normal = 0
    while normal < cfg.normal_samples and attempts < cfg.normal_samples * 50:
        attempts += 1
        start = rng.uniform(0.0, max(duration_ms - cfg.interval_ms, 0.0))
        end = start + cfg.interval_ms
        if in_disturbance(start, end) or near_reference(start, end):
            continue
        # Not relevant: no reference, no disturbance. A gate should pass these,
        # and one that suppresses them is over-suppressing quiet footage.
        population.add(NORMAL, f"normal_{normal}", start, end, False)
        normal += 1

    # Falling short is reported, never passed over. Normal intervals are the
    # only population that can expose a gate which passes everything, so an
    # audit that quietly obtained none of them is an audit missing its most
    # informative half -- and it would otherwise look like a complete audit.
    population.shortfall = {
        "normal_requested": cfg.normal_samples,
        "normal_obtained": normal,
        "attempts": attempts,
        "reason": "" if normal >= cfg.normal_samples else
        f"only {normal} of {cfg.normal_samples} normal intervals could be "
        f"drawn in {attempts} attempts. Disturbance spans and reference "
        f"neighbourhoods cover most of this recording, so there is little "
        f"undisturbed footage to sample. The gate's behaviour on quiet "
        f"footage is therefore UNMEASURED here, not measured as good.",
    }

    for index, (lo, hi, condition) in enumerate(disturbance_spans[:cfg.disturbance_samples]):
        # Relevant only if a reference falls inside. A disturbance span with no
        # reference in it is a span the gates are *right* to hold back.
        relevant = any(lo <= t <= hi for t in reference_times)
        population.add(DISTURBANCE, f"{condition}_{index}", lo, hi, relevant)

    return population


def audit(log: GateLog, population: AuditPopulation,
          duration_ms: float | None = None) -> dict:
    """Run the audit over a mixed population and report it by source.

    Reported per source as well as overall, because the two populations answer
    different questions and one aggregate number hides both. A gate can have
    zero false suppressions on references and suppress half of normal footage;
    only the split shows it.
    """
    auditor = GateAudit(log)
    for source, identifier, start_ms, end_ms, relevant in population.intervals:
        auditor.judge(identifier, start_ms, end_ms, relevant, note=source)

    overall = auditor.metrics(duration_ms)
    if overall.get("status") == "unaudited":
        return {"overall": overall, "by_source": {},
                "population": population.counts(),
                "shortfall": population.shortfall}

    by_source: dict[str, dict] = {}
    for source in {s for s, *_ in population.intervals}:
        rows = [o for o in auditor.outcomes if o.note == source]
        relevant = [o for o in rows if o.relevant]
        irrelevant = [o for o in rows if not o.relevant]
        from .gates import RANKABLE
        correct_pass = sum(1 for o in relevant if o.state in RANKABLE)
        false_pass = sum(1 for o in irrelevant if o.state in RANKABLE)
        by_source[source] = {
            "intervals": len(rows),
            "relevant": len(relevant),
            "correct_pass": correct_pass,
            "false_suppression": len(relevant) - correct_pass,
            "false_pass": false_pass,
            "correct_suppression": len(irrelevant) - false_pass,
            "false_suppression_rate": round(
                (len(relevant) - correct_pass) / len(relevant), 4)
            if relevant else None,
        }

    return {
        "overall": overall,
        "by_source": by_source,
        "population": population.counts(),
        "shortfall": population.shortfall,
        "failures": [
            {"reference_id": o.reference_id, "source": o.note,
             "start_pts_ms": o.start_pts_ms, "end_pts_ms": o.end_pts_ms,
             "relevant": o.relevant, "state": o.state}
            for o in auditor.failures()],
        "note": "audited on references plus sampled normal and disturbance "
                "intervals (PRD 8). References alone cannot expose a gate that "
                "passes everything: on that population it scores a perfect "
                "false-suppression rate.",
    }
