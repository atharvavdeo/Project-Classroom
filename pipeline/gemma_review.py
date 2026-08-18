"""Bounded Gemma review of flagged evidence (PRD 11).

PRD 11 bounds the VLM tightly, and every bound is here as code rather than as an
instruction the model is asked to follow:

> Gemma is called only after flagged/ambiguous motion/pose/object evidence
> exists. It cannot create candidates, alter boundaries, alter rank, identify
> unmarked people, or state intent.

`review()` takes packets and returns reviews. It has no path to the candidate
manifest, no path to segmentation, and no return value that ranking consults for
ordering -- `ranking.py` reads `needs_human_review` and nothing else. A model
that cannot reach a thing cannot alter it.

### Schema-invalid output stays invalid

> Schema-invalid output stays invalid, never repaired into evidence.

There is no repair path in this module. Not a lenient parser, not a retry with a
patched prompt, not a "close enough" fallback. A response that does not validate
is stored with its validation failure and the packet is marked as unreviewed by
the VLM. The reason is that repair is indistinguishable from fabrication once it
reaches the artifact: a JSON object assembled by the pipeline from a malformed
response is not the model's assessment, but nothing downstream can tell.

### The grammar

Grammar-constrained decoding makes a malformed response unrepresentable rather
than merely discouraged. The GBNF constraints below were measured against
llama.cpp b10472 in an earlier phase and are carried forward verbatim, because
each cost a live failure to find:

  * one rule per line -- the pretty-printed form is rejected outright;
  * no backslash inside a character class -- the conventional JSON string rule
    does not parse;
  * repetition bounds on every list, string and whitespace run, or the model
    loops until the token budget cuts it off mid-string and nothing parses.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

PROMPT_VERSION = "product-zero-1"

# PRD 11's enumerated fields.
EVIDENCE_QUALITY = ("sufficient", "limited", "insufficient")
OBJECT_CERTAINTY = ("not_visible", "possible", "supported")
OBJECT_ASSESSMENT = ("phone_like_object", "paper_like_object", "other_object",
                     "no_object_visible", "uncertain")

# Bounds on the free-text list, matching the grammar below.
MAX_ACTIONS = 5
MIN_TEXT, MAX_TEXT = 12, 240
MAX_NOTE = 400


# ------------------------------------------------------------- grammar --

GRAMMAR = (
    'root ::= "{" ws '
    '"\\"subject_marker_visible\\":" ws boolean ws "," ws '
    '"\\"evidence_quality\\":" ws quality ws "," ws '
    '"\\"observable_actions\\":" ws actions ws "," ws '
    '"\\"object_assessment\\":" ws assessment ws "," ws '
    '"\\"object_certainty\\":" ws certainty ws "," ws '
    '"\\"quality_limitations\\":" ws actions ws "," ws '
    '"\\"needs_human_review\\":" ws boolean ws "," ws '
    '"\\"reasoning_note\\":" ws note ws "}"\n'
    'boolean ::= "true" | "false"\n'
    'quality ::= "\\"sufficient\\"" | "\\"limited\\"" | "\\"insufficient\\""\n'
    # Bounded, deliberately. An unbounded list lets the model loop until the
    # token budget cuts it off, and a response truncated mid-string is
    # unparseable however correct the grammar is.
    'actions ::= "[" ws (string (ws "," ws string){0,4})? ws "]"\n'
    'assessment ::= "\\"phone_like_object\\"" | "\\"paper_like_object\\"" '
    '| "\\"other_object\\"" | "\\"no_object_visible\\"" | "\\"uncertain\\""\n'
    'certainty ::= "\\"not_visible\\"" | "\\"possible\\"" | "\\"supported\\""\n'
    # The lower bound on a string is not decoration. A live run produced a list
    # whose first entry was a real sentence and whose remaining entries were the
    # literal string " ," -- the model padding a list a grammar accepted at any
    # length. A floor makes the padding unrepresentable.
    'string ::= "\\"" [^"\\n\\r\\t]{12,240} "\\""\n'
    'note ::= "\\"" [^"\\n\\r\\t]{0,400} "\\""\n'
    # Whitespace is bounded too, and this one is easy to miss: an unbounded run
    # is a token sink. The model can emit blank lines between fields until the
    # budget is gone, having produced no content at all.
    'ws ::= [ \\t\\n]{0,4}\n'
)


# -------------------------------------------------------------- prompt --

SYSTEM = """You are an evidence reviewer for recorded examination-hall CCTV. You describe what is visible in still images. You do not decide whether any rule was broken; a human reviewer does that, using your description.

The subject under review is marked with a red box and a seat label in every full-frame tile, and shown again in a magnified crop. Describe only that marked subject. Other people are visible in these images; they are not under review, and you must not describe them.

Every desk in this hall carries a monitor, a keyboard and a mouse. A small dark rectangle near a hand is far more often one of those than a phone. Weigh what you see against what is normally on the desk.

The camera is distant and the footage is compressed. Where the image does not support a call, say so: `insufficient` evidence quality and `not_visible` certainty are correct answers, not failures.

You must not:
- state or imply that anyone was cheating, breaking a rule, or acting dishonestly;
- infer intent, motive, or state of mind;
- identify anyone by name, appearance, gender, or any personal characteristic;
- describe any person other than the marked subject;
- treat the motion figures in the metadata as evidence about behaviour. They describe pixel change, nothing more.

Report only what is observable: hand and arm positions, where a hand moved to, head orientation relative to the desk, and whether an object is visible and what it resembles."""

USER_TEMPLATE = """{legend}

Metadata (reference only; not evidence about behaviour):
- subject seat: {seat}
- interval: {start:.2f}s to {end:.2f}s, peak at {peak:.2f}s
- source quality: {health}
- object observations from the detector: {objects}

Describe what is visible for the marked subject. Return one JSON object matching the required schema. Keep each observation to one short sentence."""


def legend(packet) -> str:
    """A frame legend built from the packet's own tiles.

    Generated, never asserted. An earlier version described five tiles in prose
    while the sheet held six, so the model was told the last tile was a
    magnified crop when it was another timeline frame. Building the legend from
    the same tiles that make the sheet means the two cannot disagree.
    """
    from .evidence import FULL_FRAME, OBJECT_CROP, SUBJECT_CROP

    frames = [t for t in packet.tiles if t.role == FULL_FRAME]
    subjects = [t for t in packet.tiles if t.role == SUBJECT_CROP]
    objects = [t for t in packet.tiles if t.role == OBJECT_CROP]

    parts = [f"This contact sheet holds {len(packet.tiles)} numbered tiles, read "
             f"left to right and top to bottom."]
    parts.append(
        f"Tiles 1-{len(frames)} are full frames at "
        + ", ".join(f"{t.pts_ms / 1000.0:.2f}s" for t in frames)
        + ". The red box and label mark the subject seat in each.")
    if subjects:
        first = len(frames) + 1
        parts.append(
            f"Tile {first} is the same subject magnified "
            f"{subjects[0].magnification:.1f}x from "
            f"{subjects[0].native_w:.0f}x{subjects[0].native_h:.0f} source pixels.")
    if objects:
        first = len(frames) + len(subjects) + 1
        parts.append(
            f"Tiles {first}-{first + len(objects) - 1} are magnified crops of "
            f"regions where the detector reported an object.")
    return " ".join(parts)


def build_prompt(packet) -> dict:
    """The full prompt for one packet, with everything needed to reproduce it."""
    objects = ", ".join(
        f"{o.cls} at {o.confidence:.2f} ({o.status})" for o in packet.objects
    ) or "none reported"
    user = USER_TEMPLATE.format(
        legend=legend(packet),
        seat=packet.seat_state,
        start=packet.start_pts_ms / 1000.0,
        end=packet.end_pts_ms / 1000.0,
        peak=packet.peak_pts_ms / 1000.0,
        health=", ".join(packet.health_state) or "no condition recorded",
        objects=objects)
    return {
        "prompt_version": PROMPT_VERSION,
        "system": SYSTEM,
        "user": user,
        "grammar": GRAMMAR,
        "event_id": packet.event_id,
    }


# ---------------------------------------------------------- validation --

# Vocabulary that turns an observation into a verdict. Checked on output, not
# only asked for in the prompt: PRD 17.14 forbids user-facing verdict language,
# and a rule enforced only by instruction is a rule the model may decline.
FORBIDDEN = (
    "cheat", "cheating", "cheated", "malpractice", "misconduct", "dishonest",
    "guilty", "innocent", "violation", "violated", "caught", "culprit",
    "offence", "offense", "suspicious", "suspect", "illegal", "breaking the rule",
    "broke the rule", "should be punished", "disciplinary", "intent", "intended",
    "trying to", "attempting to", "deliberately", "on purpose",
)

# Words describing a person rather than their observable action.
#
# Matched on **word boundaries**, not as substrings. Substring matching looked
# fine and was badly wrong: `"he "` matches inside "t*he* right hand", so every
# ordinary observational sentence was rejected as person-identifying, and
# `"man"` matches inside "human". A rule that fires on correct output is worse
# than no rule -- it trains whoever reads the artifacts to ignore the flag.
IDENTITY = ("man", "men", "woman", "women", "boy", "girl", "male", "female",
            "he", "she", "his", "her", "hers", "him", "gentleman", "lady",
            "young", "elderly", "beard", "moustache", "skin", "complexion",
            "ethnicity")

_FORBIDDEN_RE = None
_IDENTITY_RE = None


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return asdict(self)


def _compiled(terms: tuple[str, ...]):
    """One alternation of whole-word patterns, longest first.

    Longest first so that a multi-word phrase is reported as itself rather than
    as whichever single word inside it happened to match earlier.
    """
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(
        r"\b(" + "|".join(re.escape(t.strip()) for t in ordered) + r")\b",
        re.IGNORECASE)


def contains_forbidden(text: str) -> list[str]:
    global _FORBIDDEN_RE
    if _FORBIDDEN_RE is None:
        _FORBIDDEN_RE = _compiled(FORBIDDEN)
    return sorted({m.group(0).lower() for m in _FORBIDDEN_RE.finditer(text)})


def contains_identity(text: str) -> list[str]:
    global _IDENTITY_RE
    if _IDENTITY_RE is None:
        _IDENTITY_RE = _compiled(IDENTITY)
    return sorted({m.group(0).lower() for m in _IDENTITY_RE.finditer(text)})


def validate(payload: dict) -> ValidationResult:
    """Validate a parsed response against PRD 11's schema. No repair.

    Every failure is collected rather than short-circuited, so the artifact
    records everything wrong with a response instead of only the first thing.
    """
    errors: list[str] = []

    required = ("subject_marker_visible", "evidence_quality",
                "observable_actions", "object_assessment", "object_certainty",
                "quality_limitations", "needs_human_review", "reasoning_note")
    for key in required:
        if key not in payload:
            errors.append(f"missing required field {key!r}")

    if not isinstance(payload.get("subject_marker_visible"), bool):
        errors.append("subject_marker_visible must be a boolean")
    if not isinstance(payload.get("needs_human_review"), bool):
        errors.append("needs_human_review must be a boolean")

    quality = payload.get("evidence_quality")
    if quality not in EVIDENCE_QUALITY:
        errors.append(f"evidence_quality {quality!r} is not one of {EVIDENCE_QUALITY}")

    assessment = payload.get("object_assessment")
    if assessment not in OBJECT_ASSESSMENT:
        errors.append(f"object_assessment {assessment!r} is not one of "
                      f"{OBJECT_ASSESSMENT}")

    certainty = payload.get("object_certainty")
    if certainty not in OBJECT_CERTAINTY:
        errors.append(f"object_certainty {certainty!r} is not one of "
                      f"{OBJECT_CERTAINTY}")

    for key in ("observable_actions", "quality_limitations"):
        value = payload.get(key)
        if not isinstance(value, list):
            errors.append(f"{key} must be a list")
            continue
        if len(value) > MAX_ACTIONS:
            errors.append(f"{key} has {len(value)} entries, over the "
                          f"{MAX_ACTIONS} bound")
        for entry in value:
            if not isinstance(entry, str):
                errors.append(f"{key} contains a non-string entry")
            elif not (MIN_TEXT <= len(entry) <= MAX_TEXT):
                errors.append(
                    f"{key} entry of {len(entry)} characters is outside "
                    f"{MIN_TEXT}-{MAX_TEXT}; short entries are list padding and "
                    f"long ones truncate mid-word")

    note = payload.get("reasoning_note")
    if not isinstance(note, str):
        errors.append("reasoning_note must be a string")
    elif len(note) > MAX_NOTE:
        errors.append(f"reasoning_note of {len(note)} characters is over "
                      f"{MAX_NOTE}")

    # A response asserting no marker was visible is not invalid -- it is a
    # correct and important answer, and it forces human review below.
    text = " ".join(
        [str(payload.get("reasoning_note", ""))]
        + [str(v) for v in payload.get("observable_actions", []) if isinstance(v, str)]
        + [str(v) for v in payload.get("quality_limitations", []) if isinstance(v, str)])

    forbidden = contains_forbidden(text)
    if forbidden:
        errors.append(
            f"verdict or intent vocabulary present: {forbidden}. PRD 1 and 17.14 "
            f"forbid it; this response is not evidence and is not repaired")
    identity = contains_identity(text)
    if identity:
        errors.append(
            f"person-identifying vocabulary present: {identity}. PRD 11 forbids "
            f"identifying anyone; describe the action, not the person")

    return ValidationResult(valid=not errors, errors=errors,
                            forbidden_terms=forbidden + identity)


# -------------------------------------------------------------- review --

@dataclass
class Review:
    """One VLM review, valid or not, with everything PRD 11 requires retained."""

    event_id: str
    config_id: str = "gemma_marked_evidence"
    prompt_version: str = PROMPT_VERSION
    checkpoint_sha256: str = ""
    raw_response: str = ""
    parsed: dict | None = None
    validation: ValidationResult | None = None
    latency_ms: float = 0.0
    contact_sheet: str = ""
    state: str = "available"
    reason: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.validation and self.validation.valid)

    @property
    def needs_human_review(self) -> bool:
        """True whenever a human must look, including when the model failed.

        An invalid response is a reason for a human to look, not a reason to
        skip the event. Defaulting it to False would let a malformed response
        quietly demote evidence.
        """
        if not self.usable:
            return True
        return bool(self.parsed.get("needs_human_review", True))

    def to_row(self) -> dict:
        return {
            "event_id": self.event_id,
            "config_id": self.config_id,
            "prompt_version": self.prompt_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "raw_response": self.raw_response,
            "parsed": self.parsed,
            "validation": self.validation.to_row() if self.validation else None,
            "latency_ms": round(self.latency_ms, 2),
            "contact_sheet": self.contact_sheet,
            "state": self.state,
            "reason": self.reason,
            "usable": self.usable,
            "needs_human_review": self.needs_human_review,
        }


def review_packet(packet, runner, checkpoint_sha256: str = "") -> Review:
    """Review one packet. Refuses to call the model on an invalid packet."""
    problems = packet.validate()
    if problems:
        return Review(
            event_id=packet.event_id, state="blocked",
            reason="packet does not meet PRD 11: " + "; ".join(problems),
            checkpoint_sha256=checkpoint_sha256,
            validation=ValidationResult(valid=False, errors=problems))

    prompt = build_prompt(packet)
    started = time.perf_counter()
    try:
        raw = runner(prompt)
    except Exception as exc:                            # noqa: BLE001
        return Review(event_id=packet.event_id, state="launch_failed",
                      reason=f"{type(exc).__name__}: {exc}",
                      checkpoint_sha256=checkpoint_sha256,
                      latency_ms=(time.perf_counter() - started) * 1000.0)
    latency = (time.perf_counter() - started) * 1000.0

    review = Review(event_id=packet.event_id, raw_response=raw,
                    checkpoint_sha256=checkpoint_sha256, latency_ms=latency,
                    contact_sheet=f"11_gemma/{packet.event_id}/contact_sheet.png")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Not repaired. Not partially parsed. Stored as it arrived, with why.
        review.validation = ValidationResult(
            valid=False,
            errors=[f"response is not valid JSON: {exc}. It is retained "
                    f"verbatim and never repaired into evidence (PRD 11)"])
        return review

    if not isinstance(parsed, dict):
        review.validation = ValidationResult(
            valid=False, errors=["response is not a JSON object"])
        return review

    review.parsed = parsed
    review.validation = validate(parsed)
    return review


def review(packets: list, runner, checkpoint_sha256: str = "") -> list[Review]:
    """Review every flagged packet. Order is the packets' order; nothing is reranked."""
    return [review_packet(p, runner, checkpoint_sha256) for p in packets]


def unavailable(packets: list, state: str, reason: str) -> list[Review]:
    """Record every packet as unreviewed when the VLM cannot run."""
    return [Review(event_id=p.event_id, state=state, reason=reason)
            for p in packets]


def summarise(reviews: list[Review]) -> dict:
    if not reviews:
        return {"reviews": 0,
                "note": "no packets were reviewed. Gemma cannot create "
                        "candidates, so an empty review set means no flagged "
                        "evidence existed, not that nothing happened."}
    # A review that never ran is not a review that failed validation. Counting
    # them together makes an absent model look like a broken one, and hides the
    # fact that no assessment was made at all.
    attempted = [r for r in reviews if r.state == "available"]
    not_attempted = [r for r in reviews if r.state not in ("available", "blocked")]
    valid = [r for r in attempted if r.usable]

    error_counts: dict[str, int] = {}
    for review_row in attempted:
        if review_row.validation and not review_row.validation.valid:
            for error in review_row.validation.errors:
                key = error.split(":")[0][:60]
                error_counts[key] = error_counts.get(key, 0) + 1

    assessments: dict[str, int] = {}
    certainties: dict[str, int] = {}
    for row in valid:
        assessments[row.parsed["object_assessment"]] = \
            assessments.get(row.parsed["object_assessment"], 0) + 1
        certainties[row.parsed["object_certainty"]] = \
            certainties.get(row.parsed["object_certainty"], 0) + 1

    latencies = sorted(r.latency_ms for r in reviews if r.latency_ms)
    return {
        "reviews": len(reviews),
        "attempted": len(attempted),
        "not_attempted": len(not_attempted),
        "not_attempted_reason": not_attempted[0].reason if not_attempted else "",
        "schema_valid": len(valid),
        "schema_invalid": len(attempted) - len(valid),
        "repaired": 0,
        "blocked_packets": sum(1 for r in reviews if r.state == "blocked"),
        "validation_errors": error_counts,
        "object_assessment": dict(sorted(assessments.items())),
        "object_certainty": dict(sorted(certainties.items())),
        "subject_marker_not_visible": sum(
            1 for r in valid if not r.parsed.get("subject_marker_visible")),
        "needs_human_review": sum(1 for r in reviews if r.needs_human_review),
        "median_latency_ms": round(latencies[len(latencies) // 2], 1)
        if latencies else None,
        "note": "no invalid response was repaired. A JSON object assembled by "
                "the pipeline from a malformed response is not the model's "
                "assessment, and nothing downstream could tell (PRD 11).",
    }


def llamacpp_runner(url: str = "http://127.0.0.1:8080/v1/chat/completions",
                    model: str = "gemma", images_of=None, timeout: float = 180.0):
    """Adapter over a running llama-server with grammar-constrained decoding."""
    import urllib.error
    import urllib.request

    def run(prompt: dict) -> str:
        content = [{"type": "text", "text": prompt["user"]}]
        if images_of is not None:
            for data_url in images_of(prompt["event_id"]):
                content.append({"type": "image_url",
                                "image_url": {"url": data_url}})
        body = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": prompt["system"]},
                         {"role": "user", "content": content}],
            "grammar": prompt["grammar"],
            "temperature": 0.0,
            "n_predict": 512,
        }).encode()
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        return payload["choices"][0]["message"]["content"]

    return run
