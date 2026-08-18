"""Phase 6 gate: the constrained schema, and the refusal to repair (PRD 11).

> Schema-invalid output stays invalid, never repaired into evidence.

The temptation this guards against is strong and reasonable-sounding. A model
returns almost-valid JSON with a trailing comma, or an enum value one character
off, and fixing it costs three lines. But the fixed object is not the model's
assessment — it is the pipeline's guess at what the model meant — and once it
reaches `parsed.json` nothing downstream can tell the difference. So there is no
repair path, and this test proves there is none by handing the validator every
near-miss that a lenient parser would happily accept.

The grammar is tested too, against the specific llama.cpp parser constraints
measured in an earlier phase. Each of those bounds cost a live failure to find:
an unbounded list looped until the token budget cut it off mid-string; an
unbounded whitespace rule let the model emit blank lines until it had produced
nothing at all; and a string rule with no lower bound produced a list padded
with the literal entry `" ,"`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evidence import ObjectEvidence, PacketConfig, build  # noqa: E402
from pipeline.gemma_review import (  # noqa: E402
    EVIDENCE_QUALITY, GRAMMAR, MAX_ACTIONS, MAX_NOTE, MAX_TEXT, MIN_TEXT,
    OBJECT_ASSESSMENT, OBJECT_CERTAINTY, review, review_packet, summarise,
    unavailable, validate,
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


VALID = {
    "subject_marker_visible": True,
    "evidence_quality": "limited",
    "observable_actions": [
        "The marked subject's right hand moves below the desk surface.",
        "The head is angled downward toward the desk.",
    ],
    "object_assessment": "uncertain",
    "object_certainty": "possible",
    "quality_limitations": [
        "The subject occupies few pixels and the image is heavily compressed.",
    ],
    "needs_human_review": True,
    "reasoning_note": "A small dark shape is visible near the right hand but "
                      "cannot be distinguished from the mouse on the desk.",
}


def packet():
    return build({"event_id": "EV-1", "seat_id": "S-61",
                  "start_pts_ms": 10000.0, "end_pts_ms": 14000.0,
                  "peak_pts_ms": 12000.0, "source_video_sha256": "abc"},
                 (60.0, 280.0, 380.0, 700.0),
                 [ObjectEvidence("phone", 0.62, "single_frame_object_candidate",
                                 "dfine_native", 1, 200.0, 500.0, 232.0, 522.0)],
                 1280, 720, PacketConfig())


def main() -> int:
    ok = True

    print("a well-formed response validates")
    result = validate(VALID)
    ok &= check("valid", result.valid, str(result.errors))

    print("\nevery required field is required")
    for key in VALID:
        missing = {k: v for k, v in VALID.items() if k != key}
        outcome = validate(missing)
        ok &= check(f"missing {key} is invalid", not outcome.valid)

    print("\nenums are closed, and a near-miss is not accepted")
    for key, bad, allowed in (
            ("evidence_quality", "sufficent", EVIDENCE_QUALITY),
            ("object_assessment", "phone", OBJECT_ASSESSMENT),
            ("object_certainty", "likely", OBJECT_CERTAINTY)):
        outcome = validate({**VALID, key: bad})
        ok &= check(f"{key}={bad!r} rejected", not outcome.valid,
                    f"a lenient parser would map this onto {allowed[0]!r}")

    print("\nbooleans must be booleans, not strings")
    ok &= check("subject_marker_visible='true' rejected",
                not validate({**VALID, "subject_marker_visible": "true"}).valid)
    ok &= check("needs_human_review=1 rejected",
                not validate({**VALID, "needs_human_review": 1}).valid)

    print("\nlist bounds hold in both directions")
    ok &= check("an over-long list is rejected",
                not validate({**VALID, "observable_actions":
                              ["x" * 20] * (MAX_ACTIONS + 1)}).valid)
    ok &= check("a padded short entry is rejected",
                not validate({**VALID, "observable_actions": [" ,"]}).valid,
                "a live run padded a list with the literal entry ' ,'")
    ok &= check("an entry at the floor is accepted",
                validate({**VALID, "observable_actions": ["x" * MIN_TEXT]}).valid)
    ok &= check("an entry over the ceiling is rejected",
                not validate({**VALID, "observable_actions":
                              ["x" * (MAX_TEXT + 1)]}).valid)
    ok &= check("an empty list is accepted",
                validate({**VALID, "observable_actions": []}).valid,
                "seeing nothing describable is a real answer")
    ok &= check("a non-list is rejected",
                not validate({**VALID, "observable_actions": "a string"}).valid)
    ok &= check("an over-long note is rejected",
                not validate({**VALID, "reasoning_note": "x" * (MAX_NOTE + 1)}).valid)

    print("\nverdict and intent vocabulary invalidates a response (PRD 17.14)")
    for phrase in ("The subject was cheating during the examination period.",
                   "This looks like clear misconduct by the marked subject.",
                   "The subject deliberately concealed the object from view.",
                   "The subject was trying to read from a hidden note.",
                   "This is suspicious behaviour that warrants discipline."):
        outcome = validate({**VALID, "reasoning_note": phrase})
        ok &= check(f"rejected: {phrase[:42]}...", not outcome.valid,
                    str(outcome.forbidden_terms))

    print("\nand it is caught in the action list too, not only the note")
    ok &= check("forbidden term inside observable_actions",
                not validate({**VALID, "observable_actions":
                              ["The subject is cheating using a phone here."]}).valid)

    print("\nperson-identifying vocabulary invalidates a response (PRD 11)")
    for phrase in ("The man in the second row reaches toward his lap area.",
                   "She lowers her right hand below the desk surface.",
                   "A young student with dark hair leans to the left side."):
        ok &= check(f"rejected: {phrase[:42]}...",
                    not validate({**VALID, "reasoning_note": phrase}).valid)

    print("\nobservational language is accepted")
    for phrase in ("The right hand moves below the desk surface and returns.",
                   "A small dark rectangle is visible beside the keyboard.",
                   "The head is angled downward for several consecutive frames."):
        ok &= check(f"accepted: {phrase[:42]}...",
                    validate({**VALID, "reasoning_note": phrase}).valid,
                    str(validate({**VALID, "reasoning_note": phrase}).errors))

    print("\nmalformed JSON is retained verbatim and never repaired")
    for broken, label in (
            ('{"subject_marker_visible": true,}', "trailing comma"),
            ('{"evidence_quality": "limited"', "truncated mid-object"),
            ('not json at all', "not JSON"),
            ('["a", "list"]', "a JSON array, not an object")):
        outcome = review_packet(packet(), lambda _p, b=broken: b)
        ok &= check(f"{label}: not usable", not outcome.usable)
        ok &= check(f"{label}: raw response retained verbatim",
                    outcome.raw_response == broken, outcome.raw_response[:30])
        ok &= check(f"{label}: nothing was parsed into evidence",
                    outcome.parsed is None or not outcome.usable)
        ok &= check(f"{label}: it forces human review",
                    outcome.needs_human_review)

    print("\na schema-valid response round-trips through the runner")
    good = review_packet(packet(), lambda _p: json.dumps(VALID))
    ok &= check("usable", good.usable, str(good.validation.errors))
    ok &= check("latency is recorded", good.latency_ms >= 0.0)
    ok &= check("the prompt version is recorded", bool(good.prompt_version))
    ok &= check("the contact sheet path is recorded",
                good.contact_sheet.startswith("11_gemma/"), good.contact_sheet)
    ok &= check("needs_human_review comes from the model when valid",
                good.needs_human_review is True)
    off = {**VALID, "needs_human_review": False}
    ok &= check("and it can be False when the model says so",
                not review_packet(packet(), lambda _p: json.dumps(off))
                .needs_human_review)

    print("\na runner that raises is recorded, not swallowed")
    def broken_runner(_p):
        raise RuntimeError("llama-server refused the connection")
    failed = review_packet(packet(), broken_runner)
    ok &= check("state is launch_failed", failed.state == "launch_failed",
                failed.state)
    ok &= check("the exception text survives",
                "refused the connection" in failed.reason)
    ok &= check("it forces human review", failed.needs_human_review)

    print("\nthe grammar carries every measured llama.cpp constraint")
    ok &= check("one rule per line",
                all(line.count("::=") == 1
                    for line in GRAMMAR.strip().split("\n")),
                "the pretty-printed form is rejected with 'failed to parse grammar'")
    ok &= check("no backslash inside a character class",
                "[^\"\\\\]" not in GRAMMAR,
                "the conventional JSON string rule does not parse")
    ok &= check("the action list is bounded",
                "{0,4}" in GRAMMAR,
                "an unbounded list loops until the budget cuts it mid-string")
    ok &= check("strings have a floor and a ceiling",
                "{12,240}" in GRAMMAR,
                "no floor produced a list padded with the literal ' ,'")
    ok &= check("whitespace is bounded",
                "ws ::= [ \\t\\n]{0,4}" in GRAMMAR,
                "an unbounded run is a token sink that produces no content")
    for enum in ("sufficient", "not_visible", "supported", "phone_like_object"):
        ok &= check(f"the grammar enumerates {enum!r}", f'\\"{enum}\\"' in GRAMMAR)
    ok &= check("every schema field appears in the grammar",
                all(f'\\"{k}\\":' in GRAMMAR for k in VALID),
                "a field the grammar cannot emit can never validate")

    print("\nthe summary reports invalidity and asserts nothing was repaired")
    reviews = [good, review_packet(packet(), lambda _p: "broken")]
    report = summarise(reviews)
    ok &= check("both counted", report["reviews"] == 2)
    ok &= check("one schema-invalid", report["schema_invalid"] == 1)
    ok &= check("repaired is reported as zero", report["repaired"] == 0)
    ok &= check("the note explains why repair is refused",
                "not the model's" in report["note"], report["note"][:60])
    ok &= check("an unavailable VLM still yields one row per packet",
                len(unavailable([packet()], "resource_unavailable", "no server")) == 1)
    ok &= check("and those rows force human review",
                unavailable([packet()], "resource_unavailable", "x")[0]
                .needs_human_review)

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
