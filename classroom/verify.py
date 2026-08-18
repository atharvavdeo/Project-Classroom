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
    # The lower bound is not decoration. A live run produced an observation list
    # whose first entry was a real sentence and whose remaining four entries
    # were the literal string `" ,"` -- the model padding the list to satisfy a
    # grammar that accepted any length including none. A floor of twelve
    # characters makes that padding unrepresentable rather than discouraged.
    #
    # The upper bound rose from 200 to 240 after the same run cut observations
    # mid-word: "...Both hands are visible on the desk surface, near the
    # keyboard. The 7". The real fix for that is in the prompt, which now asks
    # for one short observation per item instead of a paragraph; the extra
    # headroom is so a well-formed short observation is never the thing that
    # trips the bound.
    'string ::= "\\"" [^"\\n\\r\\t]{12,240} "\\""\n'
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

# The prompt is split into a standing system message and a per-event user
# message. The rules do not change between events, and keeping them out of the
# user turn stops per-event metadata from being read as a new instruction.
#
# Three things in here are answers to measured failure modes rather than
# generic prompt hygiene:
#
#   * **The scene is described before the task.** Every desk in this footage
#     carries a monitor, a keyboard and a mouse, and those three objects are
#     exactly what the stock detector misread as phones -- 40% precision, with
#     computer mice the single largest confuser (PRD 8.1.2). A model with no
#     prior for what is normally on the desk sees a small dark rectangle beside
#     a hand and has nothing to weigh "phone" against. Naming the furniture is
#     the cheapest available correction.
#
#   * **The metadata is fenced off as reference, not evidence.** The packet
#     carries a motion z-score, and a model told "peak deviation 8.2 sigma" will
#     happily narrate the dramatic event that number implies. The deviation is a
#     statement about pixels, not about behaviour, and the prompt says so.
#
#   * **The frame legend is generated, not asserted.** The previous version
#     described five tiles in prose while the configured sheet held six, so the
#     model was told the last tile was a magnified crop when it was another
#     timeline frame. The legend is now built from the same timestamps that
#     select the frames, so the two cannot disagree.
SYSTEM = """You are an evidence reviewer for recorded examination-hall CCTV. You describe what is visible in still frames. You do not decide whether any rule was broken -- a human reviewer does that, using your description.

The scene is a computer-based test centre. Every candidate sits at a desk that normally holds a monitor, a keyboard, a mouse and its cable, and many desks have a glass or acrylic partition. Small dark rectangular objects on these desks are usually that equipment. Seats carry printed number placards, and a date and time is burned into the image.

Rules, in order of precedence:

1. Never state or imply that cheating, misconduct, or a rule violation    occurred. Report posture, movement and position only.
2. Do not name an object as a phone or paper unless its shape and edges are    plainly resolvable. The footage is 720p from a ceiling mount; an object the    size of a hand occupies roughly thirty pixels across, which is not enough to    tell a phone from a mouse. When it is not resolvable, answer "not_visible"    or "uncertain". These are correct answers, not failures.
3. Report only what is in the image. The numbers supplied with the request    describe pixel motion, not behaviour; never turn them into an observation.
4. If the image cannot support any conclusion, set evidence_quality to    "insufficient" and say why in the note.

5. A red box marks the seat under review. Describe only the person inside    that box. Other people are visible in these frames and are not the subject;    if the box is empty, say so and set evidence_quality to "insufficient".

Each supported observation is ONE short statement -- a clause or a sentence, not a paragraph, and never a repeat of a previous one. Give between one and five. If you have only one thing to say, give one.

Useful things to report when visible: head orientation and whether it is turned down or toward a neighbour; where each hand is; whether a hand passes below the desk edge; whether the candidate leans out of their own seat area; whether anyone else is within reach."""

USER_TEMPLATE = """This contact sheet holds {count} numbered stills, read left to right and top to bottom, followed by a magnified crop of the seat under review. The red box marks that seat in every full-frame tile.

{legend}

Reference values for this event (pixel statistics, not observations):
{metadata}

Describe what is visible."""


def frame_legend(times: list[float], t_start: float, t_end: float, peak_t: float) -> str:
    """Label each tile with what it shows, derived from the actual timestamps."""
    lines = []
    for index, t in enumerate(times, start=1):
        if t < t_start:
            role = "before the flagged interval"
        elif abs(t - peak_t) < 1e-6:
            role = "peak of the flagged motion"
        elif abs(t - t_start) < 1e-6:
            role = "start of the flagged interval"
        elif abs(t - t_end) < 1e-6:
            role = "end of the flagged interval"
        else:
            role = "during the flagged interval"
        lines.append(f"  {index}. t={t:.2f}s -- {role}")
    return "\n".join(lines)


# Retained so anything importing the single-string prompt keeps working; the
# request path uses SYSTEM and USER_TEMPLATE.
PROMPT = SYSTEM + "\n\n" + USER_TEMPLATE


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
    frames: list[np.ndarray], columns: int = 3, tile: int = 320,
    highlight: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Tile selected frames into one image for the verifier.

    `highlight` is the polygon of the seat the event belongs to, in source-frame
    coordinates. It is drawn on every tile, scaled with the tile, because
    without it the verifier has no way to know which of several people in a wide
    CCTV frame is the subject.

    That is not a hypothetical. Measured on a full run of
    `03.CCTV Mobile Usage.mkv`: every event sent to the verifier belonged to the
    right-hand desk row, which was unoccupied, and the model described in
    confident detail the candidate seated on the far left of the frame --
    `evidence_quality: "sufficient"`, `confidence: 1.0`. The seat label was in
    the metadata as text. A vision model cannot resolve a text label to a region
    of an image, so the attribution was never conveyed at all.
    """
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

        # Drawn after placement, in sheet coordinates, so the box lands on the
        # pixels it describes whatever the tile scaling turned out to be.
        if highlight:
            pts = np.array([[int(px * scale) + x, int(py * scale) + y]
                            for px, py in highlight], dtype=np.int32)
            cv2.polylines(sheet, [pts], True, (60, 60, 255), 2, cv2.LINE_AA)

        cv2.putText(sheet, str(index + 1), (c * tile + 8, r * tile + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 210, 255), 2, cv2.LINE_AA)
    return sheet


def build_event_sheet(
    frames: list[np.ndarray], polygon: list[tuple[float, float]],
    columns: int = 3, tile: int = 320,
) -> np.ndarray:
    """Timeline tiles with the seat marked, plus a magnified crop of that seat.

    The magnified tile is what the original prompt already described and the
    implementation never produced. At the measured object scale a full 720p
    frame shrunk into a 320 px tile leaves a hand roughly four pixels across;
    the crop is the only tile on which anything at seat scale is legible.
    """
    from . import crops

    if not frames:
        raise ValueError("no frames supplied for the contact sheet")
    marked = build_contact_sheet(frames, columns, tile, highlight=polygon)

    # The crop comes from the frame nearest the middle of the sequence, which is
    # where the peak sits by construction in `select_frames`.
    middle = frames[len(frames) // 2]
    crop, _transform = crops.seat_crop(middle, polygon, tile, min_crop_px=32)
    strip = np.zeros((tile, marked.shape[1], 3), dtype=np.uint8)
    ch, cw = crop.shape[:2]
    scale = min(tile / cw, tile / ch)
    rw, rh = max(int(cw * scale), 1), max(int(ch * scale), 1)
    strip[(tile - rh) // 2:(tile - rh) // 2 + rh, 8:8 + rw] = cv2.resize(
        crop, (rw, rh), interpolation=cv2.INTER_AREA)
    cv2.putText(strip, "seat under review, magnified", (rw + 24, tile // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 210, 255), 2, cv2.LINE_AA)
    cv2.rectangle(strip, (6, (tile - rh) // 2 - 2), (8 + rw, (tile - rh) // 2 + rh),
                  (60, 60, 255), 2)
    return np.vstack([marked, strip])


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
        self, sheet_path: str | Path, metadata: dict, timeout: float = 600.0,
        legend: str = "", frame_count: int | None = None,
    ) -> Verification:
        import base64
        import urllib.request

        self.load()
        image = base64.b64encode(Path(sheet_path).read_bytes()).decode("ascii")
        user_text = USER_TEMPLATE.format(
            count=frame_count if frame_count is not None
            else self.cfg.contact_sheet_frames,
            legend=legend or "  (tile timestamps were not supplied)",
            metadata=json.dumps(metadata, indent=2),
        )
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
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
