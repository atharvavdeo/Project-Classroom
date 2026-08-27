"""SAM 3 run locally on the GPU, from the Hugging Face weights.

### STATUS: WRITTEN, NEVER SUCCESSFULLY RUN

This module has **not been executed against real weights**. The download of
`facebook/sam3` (3.44 GB) failed repeatedly on this machine -- `huggingface_hub`
restarted from byte zero on every stall because the CDN issues a fresh etag per
attempt -- and the work was moved back to the hosted Roboflow workflow, which
serves the same checkpoint (`sam3/sam3_final`).

It is kept because the offline requirement in PS 2 still has to be met and this
is the shape of the answer, but **nothing here is verified**: the processor call
signature was checked against `transformers` 5.15, and the memory arithmetic is
reasoning rather than measurement. Treat the first real run as a debugging
session, not a deployment.

The working path today is `pipeline/roboflow_workflow.py`.

### Why this exists

`pipeline/roboflow_workflow.py` calls SAM 3 over HTTP. That works and it is what
proved the model's value -- on 12_paper it corroborated 133 of 136 detections on
the one man who took the paper, and on 04_talking it suppressed 368 keyboards --
but it is a **network call**, and PS 2 requires the system to run offline. It is
also ~2.5 s per image against a serverless endpoint, which caps it at a referee
over a handful of survivors rather than a detector that can see a whole video.

This module runs the same model from `facebook/sam3` locally, so both of those
limits go away.

### What it is

`facebook/sam3` is `Sam3VideoModel` in transformers >= 5.x, with a text-promptable
head: `Sam3Processor` takes an image plus a text phrase and returns instance
masks and boxes for that phrase. Same open-vocabulary behaviour as the hosted
workflow, so the prompts in `tools/adjudicate_with_sam3.py` transfer directly.

### The memory problem, stated plainly

The checkpoint is **3.44 GB** and this machine's GPU is an **RTX 3050 Laptop with
4 GB**. At float32 the weights alone do not leave room for activations, so this
module loads in half precision and, if that still does not fit, says so and
stops rather than silently falling back to CPU. A silent CPU fallback is the
exact failure this project has been bitten by before: it turns a 20-minute run
into an overnight one and nothing in the output records why.

Half precision is not a quality compromise being hidden -- it is recorded in the
result so a reviewer can see the device and dtype every figure was produced on.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "facebook/sam3"

ATTRIBUTION = ("SAM 3 (facebook/sam3), Meta AI. Licence: see the model card -- "
               "it is gated and not Apache/MIT, so check before shipping.")


class Sam3Unavailable(RuntimeError):
    """The model could not be loaded on the requested device."""


@dataclass
class Config:
    model_id: str = MODEL_ID

    # "cuda" or "cpu". There is no "auto": choosing silently is how a run ends
    # up on CPU without anyone noticing.
    device: str = "cuda"

    # float16 on a 4 GB card. bfloat16 is numerically safer but Ampere laptop
    # parts handle fp16 fine here and it is the smaller of the two in practice.
    dtype: str = "float16"

    # Score floor for a returned instance.
    threshold: float = 0.40

    # Longest side the image is resized to before inference. The processor has
    # its own resize, but capping here bounds peak activation memory, which is
    # what actually decides whether this fits in 4 GB.
    max_side: int = 1024


@dataclass
class Instance:
    """One detected instance. Deliberately holds no mask."""

    cls: str
    confidence: float
    box: tuple[float, float, float, float]      # corner-based, image pixels


@dataclass
class Result:
    instances: list = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    # Recorded so no figure can be quoted without the device that produced it.
    device: str = ""
    dtype: str = ""

    def of_class(self, name: str) -> list:
        target = name.strip().lower()
        return [i for i in self.instances if i.cls.lower() == target]


class LocalSam3:
    """Text-promptable SAM 3, resident on the GPU.

    The model is loaded once and reused. Loading per call would dominate the
    runtime and thrash a 4 GB card.
    """

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.model = None
        self.processor = None
        self._loaded_device = ""
        self._loaded_dtype = ""

    def load(self) -> None:
        import torch
        from transformers import Sam3Processor, Sam3VideoModel

        if self.model is not None:
            return

        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            raise Sam3Unavailable(
                "device='cuda' but torch reports no CUDA device. Refusing to "
                "fall back to CPU silently -- pass device='cpu' if that is "
                "what you want, and expect it to be very slow.")

        dtype = getattr(torch, self.cfg.dtype)
        self.processor = Sam3Processor.from_pretrained(self.cfg.model_id)
        try:
            self.model = Sam3VideoModel.from_pretrained(
                self.cfg.model_id, dtype=dtype)
            self.model.to(self.cfg.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            self.model = None
            gc.collect()
            if self.cfg.device == "cuda":
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                raise Sam3Unavailable(
                    f"SAM 3 ({self.cfg.dtype}) did not fit: {error}. "
                    f"GPU has {total / 1e9:.1f} GB, {free / 1e9:.1f} GB free; "
                    f"the checkpoint is ~3.4 GB. Free VRAM or run on CPU "
                    f"explicitly.") from error
            raise
        self.model.eval()
        self._loaded_device = self.cfg.device
        self._loaded_dtype = self.cfg.dtype

    def vram(self) -> tuple[float, float]:
        """(allocated GB, total GB) — 0,0 on CPU."""
        import torch
        if self._loaded_device != "cuda" or not torch.cuda.is_available():
            return (0.0, 0.0)
        return (torch.cuda.memory_allocated() / 1e9,
                torch.cuda.get_device_properties(0).total_memory / 1e9)

    def detect(self, image, prompt: str) -> Result:
        """Detect instances of `prompt` in one image.

        `image` is a PIL image, a path, or an HWC uint8 RGB array. `prompt` is a
        single noun phrase -- "mobile phone", not a comma-separated list. The
        hosted workflow accepts a list because it splits it for us; here each
        phrase is one call, so callers loop.
        """
        import numpy as np
        import torch
        from PIL import Image

        self.load()

        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil = Image.fromarray(image)
        else:
            pil = image.convert("RGB")

        width, height = pil.size
        scale = 1.0
        if max(width, height) > self.cfg.max_side:
            scale = self.cfg.max_side / max(width, height)
            pil = pil.resize((int(width * scale), int(height * scale)),
                             Image.BILINEAR)

        inputs = self.processor(images=pil, text=prompt, return_tensors="pt")
        inputs = {k: (v.to(self._loaded_device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        processed = self.processor.post_process_instance_segmentation(
            outputs, threshold=self.cfg.threshold,
            target_sizes=[(pil.size[1], pil.size[0])])[0]

        instances = []
        boxes = processed.get("boxes")
        scores = processed.get("scores")
        if boxes is not None and scores is not None:
            for box, score in zip(boxes.tolist(), scores.tolist()):
                if score < self.cfg.threshold:
                    continue
                # Undo the max_side resize so boxes are in source pixels.
                instances.append(Instance(
                    cls=prompt.strip(),
                    confidence=round(float(score), 4),
                    box=(round(box[0] / scale, 1), round(box[1] / scale, 1),
                         round(box[2] / scale, 1), round(box[3] / scale, 1))))
        # Masks are deliberately dropped, for the same reason `rle_mask` is
        # dropped in `roboflow_workflow`: nothing downstream reads them and a
        # per-frame pipeline cannot afford to carry them.

        return Result(instances=instances, image_size=(width, height),
                      device=self._loaded_device, dtype=self._loaded_dtype)

    def unload(self) -> None:
        import torch
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
