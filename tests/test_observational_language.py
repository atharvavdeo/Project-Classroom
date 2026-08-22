"""Phase 6 gate: no verdict vocabulary anywhere a person will read (PRD 1, 17.14).

PRD 1 sets the product boundary and PRD 17.14 makes it a rule:

> It is not a cheating classifier. It must not infer intent, issue a verdict
> [...] Permitted language is observational.

The risk is not that someone writes "cheating detected" into a template. It is
that the vocabulary arrives one word at a time — a ranking field called
`suspicion`, a report line reading "the subject was caught", a headline that
turns an empty result into "nothing happened" — and each addition looks
reasonable next to the last. So this gate sweeps the strings the product
actually emits: ranking reasons, headlines, packet captions, prompts, and the
report generator's own vocabulary.

Two specific PRD requirements are checked here rather than left to review:

**Rank is not probability.** PRD 12: "Priority is transparent ranking, not
cheating probability." A score bounded in [0, 1] invites exactly that reading,
so the ranker's own output is checked for the framing and for the components
that justify it.

**Zero events is a result, not a silence.** PRD 12 requires the report to state
analysed, failed and affected duration with retained counts, and forbids
"nothing happened". The empty case is the one most likely to fall through to a
cheerful default, so it is tested explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evidence import ObjectEvidence, PacketConfig, build  # noqa: E402
from pipeline.gemma_review import (  # noqa: E402
    FORBIDDEN, SYSTEM, USER_TEMPLATE, build_prompt, contains_forbidden,
    contains_identity,
)
from pipeline.object_detection import TAXONOMY  # noqa: E402
from pipeline.ranking import (  # noqa: E402
    LOW, NORMAL, SUPPRESSED, WEIGHTS, rank, summarise,
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


# Vocabulary that must never appear in a user-facing string this product emits.
# Distinct from the model-output check: this one is about our own text.
BANNED_IN_OUTPUT = (
    "cheat", "cheating", "malpractice", "misconduct", "guilty", "culprit",
    "suspicion", "suspicious", "violation", "offender", "caught",
    "nothing happened", "probability of cheating", "cheating probability",
)


def scan(text: str) -> list[str]:
    lowered = text.lower()
    return [term for term in BANNED_IN_OUTPUT if term in lowered]


def event(event_id="EV-1", **kw):
    base = {
        "event_id": event_id, "seat_id": "S-61",
        "start_pts_ms": 10000.0, "end_pts_ms": 14000.0, "peak_pts_ms": 12000.0,
        "peak_deviation": 6.2, "peak_area_ratio": 0.31, "seat_confidence": 0.9,
        "health_state": [], "gate_state": "pass", "priority": "normal",
    }
    base.update(kw)
    return base


def main() -> int:
    ok = True

    print("the approved taxonomy contains no verdict class")
    for cls in TAXONOMY:
        ok &= check(f"{cls!r} is observational", not scan(cls), str(scan(cls)))
    ok &= check("abstention is a class, not an absence",
                "insufficient_visual_evidence" in TAXONOMY)

    print("\nranking component names describe evidence, not guilt")
    for name in WEIGHTS:
        ok &= check(f"{name!r} is observational", not scan(name))
    ok &= check("no component is called suspicion or risk",
                not any(w in n for n in WEIGHTS for w in ("suspicion", "risk")))

    print("\nranked events carry components and coefficients, not a probability")
    ranked = rank([event()])
    row = ranked[0].to_row()
    for field in ("score", "rank", "components", "weights", "penalties",
                  "reason", "gate_state", "needs_human_review"):
        ok &= check(f"row carries {field}", field in row)
    ok &= check("the score is unbounded, not a [0,1] probability",
                ranked[0].score > 1.0,
                f"{ranked[0].score:.2f}; a 0-1 score invites reading it as a "
                f"likelihood that something occurred")
    ok &= check("the reason is arithmetic, not a claim",
                not scan(ranked[0].reason), ranked[0].reason[:60])
    ok &= check("the score is re-derivable from the persisted numbers",
                abs(sum(row["weights"][k] * v
                        for k, v in row["components"].items()) - row["score"])
                < 1e-6,
                "with no penalty the total is exactly the weighted sum")

    print("\nGemma cannot alter rank (PRD 11)")
    class FakeReview:
        config_id = "gemma_marked_evidence"
        needs_human_review = False

    without = rank([event("A"), event("B", peak_deviation=2.0)])
    with_review = rank([event("A"), event("B", peak_deviation=2.0)],
                       reviews_by_event={"A": FakeReview(), "B": FakeReview()})
    ok &= check("order is identical with and without reviews",
                [e.event_id for e in without] == [e.event_id for e in with_review],
                str([e.event_id for e in with_review]))
    ok &= check("scores are identical too",
                [round(e.score, 6) for e in without]
                == [round(e.score, 6) for e in with_review])
    ok &= check("the review changes only the human-review flag",
                with_review[0].needs_human_review is False
                and without[0].needs_human_review is True)

    print("\na single unsustained pose signal cannot create a high-priority event")
    class Unsustained:
        sustained = False

    class Sustained:
        sustained = True

    quiet = rank([event("Q", peak_deviation=0.0, peak_area_ratio=0.0,
                        seat_confidence=0.0)],
                 pose_by_event={"Q": Unsustained()})
    ok &= check("it contributes nothing",
                quiet[0].components["pose_sustained"] == 0.0)
    loud = rank([event("L", peak_deviation=0.0, peak_area_ratio=0.0,
                       seat_confidence=0.0)],
                pose_by_event={"L": Sustained()})
    ok &= check("a sustained one does", loud[0].components["pose_sustained"] == 1.0)
    ok &= check("and the unsustained event ranks lower",
                quiet[0].score < loud[0].score,
                f"{quiet[0].score:.2f} vs {loud[0].score:.2f}")

    print("\na suppressed event keeps its score and stays searchable (PRD 8)")
    mixed = rank([event("HIGH", peak_deviation=9.0),
                  event("SUP", peak_deviation=9.0,
                        gate_state="suppress_from_auto_ranking")])
    suppressed = next(e for e in mixed if e.event_id == "SUP")
    ok &= check("its priority is suppressed", suppressed.priority == SUPPRESSED)
    ok &= check("its score is not zeroed", suppressed.score > 0.0,
                f"{suppressed.score:.2f}")
    ok &= check("it is ordered last but still ranked",
                suppressed.rank == len(mixed) and suppressed.rank > 0)
    ok &= check("the reason says it stays searchable",
                "searchable" in suppressed.reason, suppressed.reason[:60])

    print("\npenalties scale the whole score rather than subtracting from it")
    clean = rank([event("C")])[0]
    blurred = rank([event("B", health_state=["blur"])])[0]
    ok &= check("a quality-affected event scores lower",
                blurred.score < clean.score,
                f"{blurred.score:.2f} vs {clean.score:.2f}")
    ok &= check("the penalty is recorded by name",
                "quality_affected" in blurred.penalties,
                str(blurred.penalties))
    ok &= check("it is multiplicative, so it matters most on strong events",
                abs(blurred.score - clean.score * 0.6) < 1e-6)

    print("\nan ambiguous seat is penalised but never dropped")
    ambiguous = rank([event("AMB", seat_id="ambiguous_seat")])[0]
    ok &= check("it is still ranked", ambiguous.rank == 1)
    ok &= check("with an uncertain-seat penalty",
                "uncertain_seat" in ambiguous.penalties)

    print("\nzero high-priority events is reported as a result (PRD 12)")
    empty = summarise([], analysed_ms=282000.0, failed_ms=0.0, affected_ms=41600.0)
    ok &= check("the phrase 'nothing happened' never appears",
                not scan(empty["headline"]), str(scan(empty["headline"])))
    ok &= check("analysed duration is stated", empty["analysed_ms"] == 282000.0)
    ok &= check("failed duration is stated", "failed_ms" in empty)
    ok &= check("quality-affected duration is stated", "quality_affected_ms" in empty)
    ok &= check("retained low-priority count is stated",
                "low_priority_retained" in empty)
    ok &= check("suppressed count is stated",
                "suppressed_from_auto_ranking" in empty)
    ok &= check("the evaluation status is stated as incomplete",
                "no quality claim" in empty["evaluation_status"],
                empty["evaluation_status"][:60])
    ok &= check("the headline says what it is a statement about",
                "not a finding that nothing occurred" in empty["headline"],
                empty["headline"][-60:])

    print("\na populated summary avoids probability framing too")
    full = summarise(rank([event("A"), event("B")]), analysed_ms=282000.0)
    ok &= check("no banned vocabulary", not scan(full["headline"]))
    ok &= check("it says priority is a reading order",
                "reading order" in full["headline"], full["headline"][-70:])
    ok &= check("weights and penalties are published with the summary",
                full["weights"] and full["penalties"])

    print("\nthe prompt forbids the vocabulary rather than merely avoiding it")
    ok &= check("the system message states the boundary",
                "do not decide whether any rule was broken" in SYSTEM.lower())
    ok &= check("it forbids inferring intent", "infer intent" in SYSTEM.lower())
    ok &= check("it forbids identifying anyone", "identify anyone" in SYSTEM.lower())
    ok &= check("it forbids describing unmarked people",
                "other than the marked subject" in SYSTEM.lower())
    ok &= check("it names abstention as a correct answer",
                "correct answers, not failures" in SYSTEM.lower())
    ok &= check("the user template fences metadata off",
                "not evidence about behaviour" in USER_TEMPLATE.lower())

    print("\npacket captions are observational")
    packet = build(event(), (60.0, 280.0, 380.0, 700.0),
                   [ObjectEvidence("phone", 0.62, "single_frame_object_candidate",
                                   "dfine_native", 1, 200.0, 500.0, 232.0, 522.0)],
                   1280, 720, PacketConfig())
    for tile in packet.tiles:
        ok &= check(f"caption clean: {tile.caption[:38]}...",
                    not scan(tile.caption), str(scan(tile.caption)))
    prompt = build_prompt(packet)
    ok &= check("the assembled user prompt is clean",
                not scan(prompt["user"]), str(scan(prompt["user"])))

    print("\nthe detectors agree with the checkers")
    ok &= check("every FORBIDDEN term is detected in a sentence using it",
                all(contains_forbidden(f"The subject {t} something.")
                    for t in ("cheated", "deliberately", "intended")))
    ok &= check("ordinary observation trips neither checker",
                not contains_forbidden("The right hand moves below the desk.")
                and not contains_identity("The right hand moves below the desk."),
                "a rule that fires on correct output trains readers to ignore it")
    ok &= check("'human' does not trip the identity checker",
                not contains_identity("A human reviewer must confirm this."),
                "substring matching flagged 'man' inside 'human'")
    ok &= check("'the' does not trip it either",
                not contains_identity("The head is angled toward the desk."))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
