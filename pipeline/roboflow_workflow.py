"""Client for the `general-segmentation-api-6` Roboflow workflow (SAM 3).

### What this workflow actually is

Fetched from the Roboflow API rather than assumed
(`GET api.roboflow.com/kawaljeets-workspace/workflows/general-segmentation-api-6`):

    inputs   classes  (WorkflowParameter, string)
             image    (InferenceImage)
    steps    roboflow_core/sam3@v3   model_id sam3/sam3_final
                                     class_names $inputs.classes
             roboflow_core/mask_visualization@v1
             roboflow_core/label_visualization@v1
    outputs  annotated_image  <- label_visualization.image
             predictions      <- sam_3.predictions

So it is **SAM 3, prompted with free text**. `classes` is a comma-separated
prompt, not a fixed label set. That is the whole reason it is worth wiring in:
every other detector here has a frozen vocabulary, and our two worst failures
are vocabulary failures.

### Why it earns its place beside D-FINE and the chit model

Measured on our own frames, not claimed:

**It fixes the 04_talking failure.** On the frame where `chit-paper-new/2`
labelled a keyboard `chit-paper` at 0.74, prompting SAM 3 with
`"mobile phone, paper chit, keyboard"` returned **six keyboards at 0.52-0.89 and
zero phones, zero chits**. Being able to name the confuser is what stock COCO
and the single-class chit model both lack.

**It fixes the class confusion.** On verification frame 3 -- a man plainly
holding a phone -- the chit model said `chit-paper 0.52`. SAM 3 says
**`mobile phone 0.91`**.

**It stays silent on the placards** (frame 4), like the chit model and unlike
D-FINE.

### What it is worse at

**Recall.** On frames 2 and 5, where the chit model found held paper at 0.56 and
0.72, SAM 3 returned nothing. It is the more precise instrument and the less
sensitive one, so it belongs as a *referee* over existing detections, not as a
replacement for them.

### Response shape, from a real call

Do not infer these names; they are the workflow's own output names and were read
off an actual response:

    {"outputs": [ {"annotated_image": {"type": "base64", "value": "..."},
                   "predictions":     {"image": {"width": W, "height": H},
                                       "predictions": [ ... ]}} ],
     "profiler_trace": ...}

Each prediction carries `x, y, width, height` (**centre-based**), `confidence`,
`class`, `class_id`, `detection_id`, and `rle_mask` (COCO RLE, several KB).

Two traps in that payload, both hit in testing:

* `class` arrives with **leading whitespace** -- `" keyboard"` -- because the
  prompt is split on commas without stripping. Always `.strip()`.
* `rle_mask` is large and we never use it. It is dropped on parse rather than
  carried through the pipeline, along with any `points`.
* `predictions.image` reports the dimensions of the image **as the workflow saw
  it**, which is not always the size we sent, so coordinates must be read
  against it rather than against our own frame.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# Workspace, workflow and its parameter/output names are all environment
# overridable, because they are not properties of SAM 3 -- they are properties
# of whichever Roboflow account currently has credits. `kawaljeets-workspace`
# ran out mid-session (HTTP 402, credit_cap_exceeded) and everything had to
# move; that must be a config change, never a code edit.
#
# The two workflows differ in contract, which is why the names are constants
# rather than literals:
#
#   kawaljeets/general-segmentation-api-6   classes    -> predictions
#   kitretsu/sam-3                          className  -> model_predictions
#
# `className` also takes a LIST, where `classes` took a comma-separated string.
WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "kitretsu-gayitunde")
WORKFLOW = os.environ.get("ROBOFLOW_SAM3_WORKFLOW", "sam-3")

# The prompt parameter's name, and whether it wants a list.
PROMPT_PARAM = os.environ.get("ROBOFLOW_SAM3_PROMPT_PARAM", "className")
PROMPT_IS_LIST = PROMPT_PARAM == "className"
ENDPOINT = "https://serverless.roboflow.com"

# The workflow's own output names, read from its definition. Used to validate a
# response rather than to guess at one.
OUTPUT_ANNOTATED = os.environ.get("ROBOFLOW_SAM3_OUT_ANNOTATED",
                                  "label_visualization")
OUTPUT_PREDICTIONS = os.environ.get("ROBOFLOW_SAM3_OUT_PREDICTIONS",
                                    "model_predictions")

ATTRIBUTION = (f"Roboflow workflow {WORKSPACE}/{WORKFLOW} (SAM 3, "
               f"sam3/sam3_final)")


class WorkflowError(RuntimeError):
    """The workflow could not be run, or answered with something unusable."""


class WorkflowAuthError(WorkflowError):
    """The API key is missing, wrong, or not entitled to this workflow."""


class WorkflowResponseError(WorkflowError):
    """A 200 whose body does not match the workflow's declared outputs."""


class WorkflowConfigError(WorkflowError):
    """The workspace or workflow named does not exist for this key.

    Kept apart from every other failure on purpose. This one is fatal for the
    whole pass -- retrying it, or letting it accumulate in a generic error
    counter, is how a five-second typo became a four-minute wait and then a
    negative finding against real people.
    """


@dataclass
class Config:
    workspace: str = WORKSPACE
    workflow: str = WORKFLOW
    endpoint: str = ENDPOINT

    # SAM 3 runs ~1.4-3.3 s per image on the serverless tier, so the timeout is
    # generous compared with the plain detector endpoint.
    timeout_s: float = 180.0

    # Retries with backoff. The chit pass lost 2,937 of 3,439 calls to transient
    # 429s under 3 attempts with a <=1.5 s backoff, so this is deliberately more
    # patient and the backoff is exponential.
    retries: int = 4
    backoff_s: float = 1.5

    # Concurrency is *not* configured here. The same rate limiting means callers
    # should stay modest; `tools/` uses 6 workers.


@dataclass
class Segment:
    """One SAM 3 instance. Deliberately does not carry the mask."""

    cls: str
    confidence: float
    # Corner-based, in the coordinate space of `Prediction.image_size`.
    box: tuple[float, float, float, float]
    detection_id: str = ""

    @property
    def area(self) -> float:
        return max(self.box[2] - self.box[0], 0.0) * max(
            self.box[3] - self.box[1], 0.0)


@dataclass
class Prediction:
    """Everything from one workflow run that we keep."""

    segments: list = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    annotated_png: bytes | None = None       # only when explicitly requested

    def of_class(self, name: str) -> list:
        target = name.strip().lower()
        return [s for s in self.segments if s.cls.lower() == target]


def api_key() -> str:
    """Read the key from the environment.

    Not a parameter with a default: a key committed to the repo is a key that
    has to be rotated. Same contract as `pipeline.chit_detector.api_key`.
    """
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise WorkflowAuthError(
            "ROBOFLOW_API_KEY is not set. Get one at "
            "app.roboflow.com/settings/api and export it; do not hardcode it "
            "in a tracked file.")
    return key


def _parse(payload: dict, keep_annotated: bool) -> Prediction:
    """Turn a workflow response into a `Prediction`, defensively.

    Every lookup here is against a name taken from the workflow definition, and
    every one is checked, because a 200 with an unexpected body is the failure
    mode that silently produces "no detections" instead of an error.
    """
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise WorkflowResponseError(
            f"response has no 'outputs' list; keys were {list(payload)}")
    entry = outputs[0]
    if not isinstance(entry, dict):
        raise WorkflowResponseError(
            f"outputs[0] is {type(entry).__name__}, expected a dict")

    block = entry.get(OUTPUT_PREDICTIONS)
    # Two response shapes, because the two workflows differ:
    #
    #   kawaljeets/general-segmentation-api-6
    #       "predictions": {"image": {...}, "predictions": [...]}
    #   kitretsu/sam-3
    #       "model_predictions": [...]
    #
    # The old one nested the list under a *fixed* inner key "predictions",
    # which happened to equal the outer key -- a coincidence that hid the
    # distinction until the outer name changed. Normalise both to one shape
    # here rather than letting the difference leak into the loop below.
    if isinstance(block, list):
        block = {"predictions": block, "image": entry.get("image") or {}}
    if not isinstance(block, dict):
        raise WorkflowResponseError(
            f"no '{OUTPUT_PREDICTIONS}' dict in outputs[0]; "
            f"keys were {list(entry)}")

    dims = block.get("image") or {}
    width = int(dims.get("width") or 0)
    height = int(dims.get("height") or 0)

    segments = []
    # Always the fixed inner key, never OUTPUT_PREDICTIONS: the outer name is
    # workflow-specific, the inner one is Roboflow's own response format.
    for raw in (block.get("predictions") or []):
        if not isinstance(raw, dict):
            continue
        try:
            cx = float(raw["x"])
            cy = float(raw["y"])
            bw = float(raw["width"])
            bh = float(raw["height"])
        except (KeyError, TypeError, ValueError):
            continue
        segments.append(Segment(
            # `.strip()` is load-bearing: the API returns " keyboard".
            cls=str(raw.get("class", "")).strip(),
            confidence=float(raw.get("confidence") or 0.0),
            box=(round(cx - bw / 2, 1), round(cy - bh / 2, 1),
                 round(cx + bw / 2, 1), round(cy + bh / 2, 1)),
            detection_id=str(raw.get("detection_id", "")),
        ))
        # `rle_mask` and `points` are deliberately not copied. They are several
        # KB each, nothing downstream reads them, and carrying them through a
        # per-frame pipeline is how a 3,000-crop run runs out of memory.

    annotated = None
    if keep_annotated:
        image_block = entry.get(OUTPUT_ANNOTATED)
        if isinstance(image_block, dict) and image_block.get("value"):
            annotated = base64.b64decode(image_block["value"])

    return Prediction(segments=segments, image_size=(width, height),
                      annotated_png=annotated)


class SegmentationWorkflow:
    """Runs `general-segmentation-api-6` on single images.

    `inference-sdk` is the documented client, but it declares
    `Requires-Python >=3.10,<3.13` and this project runs 3.13, so it will not
    install. The REST contract is one POST, and `pipeline.chit_detector` already
    established `requests` as this repo's pattern for Roboflow, so this follows
    it rather than introducing a second toolchain.
    """

    def __init__(self, cfg: Config | None = None, key: str | None = None):
        import requests

        self.cfg = cfg or Config()
        self.key = key or api_key()
        self.session = requests.Session()
        self.url = (f"{self.cfg.endpoint}/{self.cfg.workspace}"
                    f"/workflows/{self.cfg.workflow}")
        self.calls = 0
        self.failures = 0

    def segment(self, image: bytes | Path | str, classes: str, *,
                keep_annotated: bool = False) -> Prediction:
        """Segment one image for a comma-separated text prompt.

        `classes` is passed through to SAM 3 as `class_names`. It is a prompt,
        not an enum: "mobile phone, paper chit, keyboard" is valid and naming
        the *confuser* is often what makes the result usable.

        Raises `WorkflowAuthError` on 401/403 and `WorkflowError` when every
        attempt failed, rather than returning empty -- an empty result must mean
        "SAM 3 found nothing", never "the call did not happen".
        """
        if isinstance(image, (str, Path)):
            payload = Path(image).read_bytes()
        else:
            payload = image
        encoded = base64.b64encode(payload).decode()

        body = {"api_key": self.key,
                "inputs": {"image": {"type": "base64", "value": encoded},
                           PROMPT_PARAM: (
                               [c.strip() for c in classes.split(",") if c.strip()]
                               if PROMPT_IS_LIST else classes)}}

        last = ""
        for attempt in range(self.cfg.retries):
            try:
                response = self.session.post(self.url, json=body,
                                             timeout=self.cfg.timeout_s)
                self.calls += 1
                if response.status_code == 200:
                    return _parse(response.json(), keep_annotated)
                if response.status_code in (401, 403):
                    raise WorkflowAuthError(
                        f"{response.status_code} from {self.url}: "
                        f"{response.text[:200]}")
                # A 404 is a misconfiguration, not a transient failure, and it
                # must not be allowed to look like one. Measured: an entire
                # adjudication pass lost 1,682 of 1,774 calls this way, because
                # ROBOFLOW_WORKSPACE pointed at a workspace the key does not
                # belong to. Every call 404'd, the counter recorded 1,682
                # generic errors, the run reported "SAM 3 unavailable", and
                # fusion turned that into a negative verdict on three people
                # with a minute of sustained handling between them.
                #
                # Retrying it four times with exponential backoff also turned a
                # five-second configuration error into a four-minute one.
                if response.status_code == 404:
                    raise WorkflowConfigError(
                        f"404 from {self.url}. The workspace or workflow does "
                        f"not exist for this API key. Check ROBOFLOW_WORKSPACE "
                        f"and ROBOFLOW_SAM3_WORKFLOW; the key's own workspace "
                        f"is reported by GET https://api.roboflow.com/ . "
                        f"Body: {response.text[:200]}")
                last = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code < 500 and response.status_code != 429:
                    break
            except (WorkflowAuthError, WorkflowConfigError):
                raise
            except WorkflowResponseError:
                raise
            except Exception as error:                           # noqa: BLE001
                last = f"{type(error).__name__}: {error}"
            time.sleep(self.cfg.backoff_s * (2 ** attempt))

        self.failures += 1
        raise WorkflowError(
            f"{self.cfg.workflow} failed after {self.cfg.retries} attempts. "
            f"Last: {last}")

    def write_annotated(self, prediction: Prediction, path: Path) -> Path:
        """Write the workflow's annotated image to disk.

        It arrives as a ~400 KB base64 blob. It is written straight out and the
        bytes dropped: it must never be logged, and holding one per frame across
        a video is hundreds of MB of nothing useful.
        """
        if prediction.annotated_png is None:
            raise WorkflowError(
                "no annotated image on this prediction; pass "
                "keep_annotated=True to segment()")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(prediction.annotated_png)
        return path
