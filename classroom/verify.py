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
GRAMMAR = r'''
root        ::= "{" ws
                "\"supported_observations\":" ws obslist ws "," ws
                "\"object_assessment\":" ws object ws "," ws
                "\"interaction_assessment\":" ws interaction ws "," ws
                "\"evidence_quality\":" ws quality ws "," ws
                "\"confidence\":" ws confidence ws "," ws
                "\"review_note\":" ws string ws
                "}"
obslist     ::= "[" ws (string (ws "," ws string)*)? ws "]"
object      ::= "\"phone_like\"" | "\"paper_like\"" | "\"other\"" | "\"not_visible\"" | "\"uncertain\""
interaction ::= "\"none\"" | "\"self_action\"" | "\"neighbouring_seat\"" | "\"uncertain\""
quality     ::= "\"sufficient\"" | "\"limited\"" | "\"insufficient\""
confidence  ::= "0" "." [0-9] [0-9]? | "1" "." "0" | "0" | "1"
string      ::= "\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""
ws          ::= [ \t\n]*
'''

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
    """llama.cpp-backed verifier, loaded lazily."""

    def __init__(self, cfg: VerifyConfig):
        self.cfg = cfg
        self._llm = None

    def load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama

        path = Path(self.cfg.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"verifier weights not found: {path}. "
                f"Download a Gemma 4 E4B instruction-tuned GGUF and place it there."
            )
        self._llm = Llama(
            model_path=str(path),
            n_ctx=self.cfg.n_ctx,
            n_gpu_layers=self.cfg.n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

    def verify(self, sheet_path: str | Path, metadata: dict) -> Verification:
        from llama_cpp import LlamaGrammar

        self.load()
        grammar = LlamaGrammar.from_string(GRAMMAR)
        prompt = PROMPT.format(metadata=json.dumps(metadata, indent=2))
        result = self._llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"file://{Path(sheet_path).resolve()}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            grammar=grammar,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        return Verification.parse(result["choices"][0]["message"]["content"])


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
