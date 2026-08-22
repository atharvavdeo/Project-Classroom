"""Paper-chit detection via a purpose-trained Roboflow model.

### Why this exists

D-FINE is stock COCO. It has no `chit` class, and the two classes it does
offer that could stand in -- `book` and `cell phone` -- both fail us:

* `book` fires on notebooks, snack packets and desk clutter,
* `cell phone` fires on the numbered seat placards on the wall, at confidence
  0.30-0.49 against 0.933 for the one phone verified by eye.

That second one is the flaw that made every object flag in run 1501
untrustworthy. Fine-tuning D-FINE would fix it but needs labelled data we do
not have.

`chit-paper-new/2` is an object detector trained on 598 images of paper chits
(1,294 after 3x augmentation), single class `chit-paper`, CC BY 4.0, published
by the Chit-Paper Project on Roboflow Universe. Tested by hand on nine crops
from our own recordings before being wired in here:

    frames 1, 2, 5      hit on the held paper            0.74, 0.56, 0.72
    frame  9            hit on a held sheet              0.68
    frame  3            hit, but the object is a PHONE   0.52
    frame  7            hit, object ambiguous            0.52
    frame  8            three overlapping boxes          0.22-0.28
    frame  4            SILENT on the placard wall       --
    frame  6            silent, though sheets on desk    --

Frame 4 is why this module exists: the placards that poison D-FINE produce
nothing here.

### What it does not tell you

The class is `chit-paper`, but frames 3 and 9 show what that really means in
practice: **a small light object being handled**. It fired on a phone, and it
fired on what is plainly a legitimate answer sheet. Paper on a desk in an exam
hall is expected; what makes a chit a chit is concealment and handling, which
is a pose question, not a detection question.

So a hit here is *corroboration that a person was handling something*, to be
weighed against wrist geometry and `hand_below_desk_line`. It is not a finding
on its own, and `attach()` deliberately writes it into `near_hand_objects`
alongside every other detector rather than into a channel of its own.

### Scale

The model was trained at 640x640 and the hand-verified crops were 318-916 px
wide. A person box in our footage has a median width of **96 px**, so crops are
upscaled before being sent -- see `Config.min_crop_px`. Sending a 30x73 px crop
to a detector trained on 640x640 imagery measures nothing.

### Network

This calls a hosted endpoint, which conflicts with the offline requirement in
PS 2. It is usable for development and for building the labelled set we need,
but a submission has to run the weights locally. `serverless.roboflow.com`
sustained ~17 req/s at 16 workers in testing, so a 6,109-crop video is about
six minutes.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass

ENDPOINT = "https://serverless.roboflow.com"
MODEL_ID = "chit-paper-new/2"

# Attribution required by the model's CC BY 4.0 licence.
ATTRIBUTION = ("chit-paper-new/2 by the Chit-Paper Project, Roboflow Universe, "
               "CC BY 4.0 -- https://universe.roboflow.com/chit-paper-project/"
               "chit-paper-new")

# What this detector's single class maps to in our taxonomy. It is *not*
# `secondary_paper_chit`: that label asserts the paper is illicit, which this
# model cannot establish (see the module docstring -- it fired on an answer
# sheet). `paper_like_object` says what was actually observed.
TAXONOMY = "paper_like_object"


@dataclass
class Config:
    """Client and crop settings."""

    model_id: str = MODEL_ID
    endpoint: str = ENDPOINT

    # Detector score floor. The hand-check put true hits at 0.52-0.74 and the
    # one clearly bad output (frame 8's three overlapping boxes) at 0.22-0.28,
    # so 0.40 separates them with room on both sides. Kept low here because
    # discarding at detection time is irreversible; the reportable floor is
    # `TimelineConfig.min_object_confidence`, applied at reprofile time.
    confidence: float = 0.40
    overlap: float = 0.50

    # Person boxes are padded before cropping so a chit held just outside the
    # box -- on the desk in front of the hands -- is still inside the crop.
    pad_fraction: float = 0.25

    # Crops are upscaled until the short side reaches this, because the model
    # was trained at 640x640 and our median person is 96 px wide. Capped by
    # `max_upscale` so a 24 px box is not blown up 13x into pure interpolation
    # noise, which invents texture the detector will happily label.
    min_crop_px: int = 320
    max_upscale: float = 4.0

    # Below this width a person is too small for the crop to carry a chit at
    # any useful scale. 4,934 of 6,109 boxes on the paper video clear 60 px.
    min_box_px: int = 60

    workers: int = 16
    timeout_s: float = 60.0
    retries: int = 3


def api_key() -> str:
    """Read the key from the environment.

    Deliberately not a parameter with a default: a key committed to the repo is
    a key that has to be rotated.
    """
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Export it before running; do not "
            "hardcode it in a tracked file.")
    return key


def crop_geometry(box, frame_w: int, frame_h: int, cfg: Config):
    """Padded, clamped integer crop rectangle for a person box.

    Returns None when the box is under `min_box_px`, so callers can skip the
    request entirely rather than pay for an answer that cannot be meaningful.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    if (x2 - x1) < cfg.min_box_px:
        return None
    px = (x2 - x1) * cfg.pad_fraction
    py = (y2 - y1) * cfg.pad_fraction
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(frame_w, int(x2 + px))
    cy2 = min(frame_h, int(y2 + py))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return cx1, cy1, cx2, cy2


def upscale_factor(w: int, h: int, cfg: Config) -> float:
    """How much to enlarge a crop so the detector sees it near training scale."""
    short = min(w, h)
    if short <= 0:
        return 1.0
    return max(1.0, min(cfg.max_upscale, cfg.min_crop_px / short))


class ChitClient:
    """Thin HTTP client for the hosted model.

    `inference-sdk` is not installable on Python 3.13 (it caps at 3.12), and
    the REST contract is a single POST, so there is nothing to gain from it.
    """

    def __init__(self, cfg: Config | None = None, key: str | None = None):
        import requests

        self.cfg = cfg or Config()
        self.key = key or api_key()
        self.session = requests.Session()
        self.url = f"{self.cfg.endpoint}/{self.cfg.model_id}"
        self.calls = 0
        self.failures = 0

    def infer(self, jpeg_bytes: bytes) -> list[dict]:
        """Detections for one encoded image, or [] if the call never succeeded.

        A failed call returns empty rather than raising, because one bad
        response should not abandon a six-minute pass over a video -- but
        `self.failures` counts them so the caller can report how much of the
        recording was actually measured instead of silently reporting zero
        chits for frames that were never scored.
        """
        payload = base64.b64encode(jpeg_bytes)
        for attempt in range(self.cfg.retries):
            try:
                response = self.session.post(
                    self.url,
                    params={"api_key": self.key,
                            "confidence": self.cfg.confidence,
                            "overlap": self.cfg.overlap},
                    data=payload,
                    headers={"Content-Type":
                             "application/x-www-form-urlencoded"},
                    timeout=self.cfg.timeout_s)
                self.calls += 1
                if response.status_code == 200:
                    return response.json().get("predictions", []) or []
                # 429 and 5xx are worth another try; 4xx generally is not.
                if response.status_code < 500 and response.status_code != 429:
                    self.failures += 1
                    return []
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(0.5 * (attempt + 1))
        self.failures += 1
        return []


def to_frame_box(prediction: dict, origin, scale: float):
    """Map one prediction from crop space back to frame pixel coordinates.

    Roboflow returns centre-x, centre-y, width, height in the *sent* image's
    pixels, so this undoes the upscale and then the crop offset. Getting this
    wrong draws boxes in plausible-looking but meaningless places, which is
    exactly the kind of error that survives review because the output still
    looks like a detection.
    """
    ox, oy = origin
    cx = float(prediction["x"]) / scale
    cy = float(prediction["y"]) / scale
    w = float(prediction["width"]) / scale
    h = float(prediction["height"]) / scale
    return [round(ox + cx - w / 2, 1), round(oy + cy - h / 2, 1),
            round(ox + cx + w / 2, 1), round(oy + cy + h / 2, 1)]
