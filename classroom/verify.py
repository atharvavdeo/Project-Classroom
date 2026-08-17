"""Phase 7 - visual verification.

The verifier sees only a handful of ambiguous events. It may re-rank one or mark
it uncertain; it may not create a high-priority event, delete evidence, or
declare misconduct (PRD 9.1).

Two design decisions carry most of the safety here:

  * Output is constrained by a GBNF grammar, not by asking politely in the
    prompt. Schema validity becomes structural rather than probabilistic, which
    is what makes the 100% valid-output acceptance criterion meaningful.

  * The model is prompted for *posture and interaction*, not object
    identification. At the measured ~30x20 px phone footprint it cannot reliably
    identify the object either, and asking it to would manufacture exactly the
    confident-and-wrong output the abstention rules exist to prevent.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from . import db
from .config import VerifyConfig

OBJECT_VALUES = ("phone_like", "paper_like", "other", "not_visible", "uncertain")
INTERACTION_VALUES = ("none", "self_action", "neighbouring_seat", "uncertain")
QUALITY_VALUES = ("sufficient", "limited", "insufficient")

# Grammar-constrained decoding. Every field is enumerated, so a malformed or
# creative response is not representable rather than merely discouraged.
#
# Three constraints of llama.cpp's GBNF parser, all found by probing it directly
# (llama-server b10472) rather than by reading the spec:
#
#   * One rule per line, no continuations. The pretty-printed form of this same
#     grammar -- `root` wrapped across eight indented lines -- is rejected
#     outright with "failed to parse grammar". Keep each rule on one line
#     however long it gets.
#
#   * The conventional JSON string rule (`[^"\\] | "\\" ["\\/bfnrt]`) does not
#     parse: a backslash inside a character class is refused. The rule below
#     excludes the quote and the control characters that are illegal inside a
#     JSON string instead of admitting escape sequences. That is also stronger
#     for this use -- the verifier emits short observational sentences with no
#     need for embedded quotes, so removing the escape machinery removes a whole
#     class of malformed-JSON failure rather than parsing around it.
#
#   * Repetition bounds are required on anything a model could loop on. See the
#     note on `obslist` and `string` below.
GRAMMAR = (
    'root ::= "{" ws '
    '"\\"supported_observations\\":" ws obslist ws "," ws '
    '"\\"object_assessment\\":" ws object ws "," ws '
    '"\\"interaction_assessment\\":" ws interaction ws "," ws '
    '"\\"evidence_quality\\":" ws quality ws "," ws '
    '"\\"confidence\\":" ws confidence ws "," ws '
    '"\\"review_note\\":" ws note ws "}"\n'
    # Bounded, deliberately. An unbounded list lets the model loop until it
    # exhausts the token budget, and a response truncated mid-string is
    # unparseable no matter how correct the grammar is. Observed in testing: a
    # model repeated one observation until n_predict cut it off. Five
    # observations is more than any real event needs.
    'obslist ::= "[" ws (string (ws "," ws string){0,4})? ws "]"\n'
    'object ::= "\\"phone_like\\"" | "\\"paper_like\\"" | "\\"other\\"" '
    '| "\\"not_visible\\"" | "\\"uncertain\\""\n'
    'interaction ::= "\\"none\\"" | "\\"self_action\\"" '
    '| "\\"neighbouring_seat\\"" | "\\"uncertain\\""\n'
    'quality ::= "\\"sufficient\\"" | "\\"limited\\"" | "\\"insufficient\\""\n'
    'confidence ::= "0" "." [0-9] [0-9]? | "1" "." "0" | "0" | "1"\n'
    # Bounded for the same reason as obslist: an unbounded string lets the model
    # run to the token limit inside one field, and the result is truncated
    # mid-quote and unparseable.
    'string ::= "\\"" [^"\\n\\r\\t]{0,200} "\\""\n'
    # The review note gets its own, larger bound. At 200 characters the model's
    # note was being cut mid-sentence, which is worse than useless to a reviewer
    # -- a truncated explanation reads as a complete one. Observations are short
    # by nature; a note is a sentence or two.
    'note ::= "\\"" [^"\\n\\r\\t]{0,400} "\\""\n'
    # Whitespace is bounded too, and this one is easy to miss. `[ \t\n]*` is a
    # token sink: the model can emit unlimited blank lines between fields and
    # exhaust its budget without ever producing content. Observed directly --
    # samples truncated mid-key after padding the list with newlines.
    'ws ::= [ \\t\\n]{0,4}\n'
)

PROMPT = """You are reviewing a still contact sheet from examination-hall CCTV.

The frames, in order, show: before the event, its onset, its peak, its end, and
a magnified crop of the seat involved.

Describe ONLY what is visible. Follow these rules exactly:

1. Report posture and movement: head orientation, arm and hand position, whether
   a hand passes below the desk line, and any interaction with a neighbouring
   seat.
2. Do NOT claim an object is a phone unless its shape and screen are plainly
   visible. At this resolution an object is usually NOT identifiable; prefer
   "uncertain" or "not_visible".
3. Never state or imply that cheating, misconduct or rule-breaking occurred.
   Report observations only.
4. If the image quality cannot support a conclusion, say so with
   evidence_quality "insufficient".

Metadata for this event:
{metadata}
"""


class TruncatedVerification(RuntimeError):
    """Generation stopped at the token limit, leaving the response incomplete."""


@dataclass
class Verification:
    supported_observations: list[str]
    object_assessment: str
    interaction_assessment: str
    evidence_quality: str
    confidence: float
    review_note: str

    def validate(self) -> None:
        if self.object_assessment not in OBJECT_VALUES:
            raise ValueError(f"bad object_assessment: {self.object_assessment}")
        if self.interaction_assessment not in INTERACTION_VALUES:
            raise ValueError(f"bad interaction_assessment: {self.interaction_assessment}")
        if self.evidence_quality not in QUALITY_VALUES:
            raise ValueError(f"bad evidence_quality: {self.evidence_quality}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        if not isinstance(self.supported_observations, list):
            raise ValueError("supported_observations must be a list")

    @classmethod
    def parse(cls, raw: str) -> "Verification":
        data = json.loads(raw)
        obj = cls(
            supported_observations=data["supported_observations"],
            object_assessment=data["object_assessment"],
            interaction_assessment=data["interaction_assessment"],
            evidence_quality=data["evidence_quality"],
            confidence=float(data["confidence"]),
            review_note=data["review_note"],
        )
        obj.validate()
        return obj


# Language a verdict would use. The verifier is not permitted to produce it, and
# neither is any other part of the system (FR-19).
FORBIDDEN = (
    "cheat", "cheating", "cheated", "malpractice", "guilty", "misconduct",
    "caught", "violation", "culprit", "offender", "suspect",
)


def contains_verdict(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in FORBIDDEN)


def sanitize(v: Verification) -> tuple[Verification, list[str]]:
    """Strip any statement that reads as a verdict rather than an observation.

    Grammar constrains structure, not vocabulary. This is the last gate before a
    reviewer sees the text, and a removed statement is reported rather than
    silently dropped.
    """
    removed: list[str] = []
    kept = []
    for observation in v.supported_observations:
        (removed if contains_verdict(observation) else kept).append(observation)

    note = v.review_note
    if contains_verdict(note):
        removed.append(note)
        note = "Review note withheld: it contained a conclusion rather than an observation."

    return Verification(
        supported_observations=kept,
        object_assessment=v.object_assessment,
        interaction_assessment=v.interaction_assessment,
        evidence_quality=v.evidence_quality,
        confidence=v.confidence,
        review_note=note,
    ), removed


def build_contact_sheet(
    frames: list[np.ndarray], columns: int = 3, tile: int = 320
) -> np.ndarray:
    """Tile selected frames into one image for the verifier."""
    if not frames:
        raise ValueError("no frames supplied for the contact sheet")
    rows = (len(frames) + columns - 1) // columns
    sheet = np.zeros((rows * tile, columns * tile, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        h, w = frame.shape[:2]
        scale = min(tile / w, tile / h)
        rw, rh = max(int(w * scale), 1), max(int(h * scale), 1)
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
        r, c = divmod(index, columns)
        y, x = r * tile + (tile - rh) // 2, c * tile + (tile - rw) // 2
        sheet[y : y + rh, x : x + rw] = resized
        cv2.putText(sheet, str(index + 1), (c * tile + 8, r * tile + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 210, 255), 2, cv2.LINE_AA)
    return sheet


def select_frames(t_start: float, t_end: float, peak_t: float, count: int) -> list[float]:
    """Timestamps for the contact sheet: before, onset, peak, end, plus fill."""
    anchors = [max(t_start - 1.0, 0.0), t_start, peak_t, t_end]
    if count <= len(anchors):
        return anchors[:count]
    extra = np.linspace(t_start, t_end, count - len(anchors) + 2)[1:-1]
    return sorted(anchors + [float(x) for x in extra])


class GemmaVerifier:
    """Verifier backed by a local `llama-server` over HTTP.

    Deliberately not the `llama-cpp-python` bindings. Two reasons, and the
    second is the important one:

      * The bindings ship no prebuilt wheel for every platform and fall back to
        compiling llama.cpp, which needs a C++ toolchain the analysis machine
        may not have. The prebuilt `llama-server` binary needs none.

      * An out-of-process verifier cannot take the pipeline down with it. PRD
        12.3 requires that a VLM failure leave motion and detection outputs
        intact; a crash in an in-process native extension would violate that,
        while a dead HTTP endpoint is just an error response.
    """

    def __init__(self, cfg: VerifyConfig, base_url: str = "http://127.0.0.1:8080"):
        self.cfg = cfg
        self.base_url = base_url.rstrip("/")

    def health(self, timeout: float = 5.0) -> bool:
        """True when a server is up and has a model loaded."""
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=timeout) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def load(self) -> None:
        """Confirm the verifier endpoint is reachable."""
        if not self.health():
            raise ConnectionError(
                f"no llama-server at {self.base_url}. Start it with:\n"
                f"  llama-server -m {self.cfg.model_path} "
                f"--mmproj {self.cfg.mmproj_path} -c {self.cfg.n_ctx} "
                f"-ngl {self.cfg.n_gpu_layers} --port 8080"
            )

    def verify(
        self, sheet_path: str | Path, metadata: dict, timeout: float = 600.0
    ) -> Verification:
        import base64
        import urllib.request

        self.load()
        image = base64.b64encode(Path(sheet_path).read_bytes()).decode("ascii")
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                    {"type": "text",
                     "text": PROMPT.format(metadata=json.dumps(metadata, indent=2))},
                ],
            }],
            "grammar": GRAMMAR,
            "n_predict": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "cache_prompt": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())

        choice = body["choices"][0]
        # Grammar guarantees a valid *prefix* of the schema, not that generation
        # finished. A response cut off at the token limit is unparseable, so it
        # is reported as a failure rather than half-read.
        if choice.get("finish_reason") == "length":
            raise TruncatedVerification(
                f"verifier hit the {self.cfg.max_tokens}-token limit before "
                f"completing its response"
            )
        return Verification.parse(choice["message"]["content"])


def persist(
    conn: sqlite3.Connection, event_id: int, model: str, v: Verification
) -> int:
    payload = asdict(v)
    return db.insert(
        conn, "vlm_verification",
        event_id=event_id,
        model=model,
        object_assessment=v.object_assessment,
        interaction_assessment=v.interaction_assessment,
        evidence_quality=v.evidence_quality,
        confidence=v.confidence,
        observations=json.dumps(payload["supported_observations"]),
        review_note=v.review_note,
    )
