"""SAM 3 as an HTTP referee for the Project Classroom pipeline.

Deployed to a Hugging Face Space so the pipeline can call open-vocabulary
segmentation without the machine running it needing a GPU. The local path
(`pipeline/sam3_local.py`) needs 3.44 GB of weights resident; this development
machine has a 4 GB card, which does not leave room for activations.

The response is shaped to match the Roboflow workflow this replaces --
centre-based boxes, a `predictions` list, an `image` block with width and
height -- so `pipeline/roboflow_workflow.py`'s parser works against either
backend unchanged.

Masks are never returned. Nothing downstream reads them, and a per-frame
pipeline cannot afford to carry them.
"""
from __future__ import annotations

import base64
import io
import os

import gradio as gr
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3VideoModel

MODEL_ID = "facebook/sam3"
MAX_SIDE = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"loading {MODEL_ID} on {DEVICE} ({DTYPE})", flush=True)
processor = Sam3Processor.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
model = Sam3VideoModel.from_pretrained(
    MODEL_ID, dtype=DTYPE, token=os.environ.get("HF_TOKEN"))
model.to(DEVICE).eval()
print("loaded", flush=True)


def _decode(image_b64: str) -> Image.Image:
    if "," in image_b64[:64] and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def predict(image_b64: str, prompt: str, threshold: float = 0.4) -> dict:
    """Detect every phrase in `prompt` (comma separated) in one image.

    Each phrase is a separate forward pass: SAM 3's text head takes one noun
    phrase at a time. The caller sends one comma-separated string because that
    is what the Roboflow workflow accepted, and splitting here keeps the two
    backends interchangeable.
    """
    if not image_b64:
        return {"error": "no image", "predictions": []}

    image = _decode(image_b64)
    width, height = image.size
    scale = 1.0
    if max(width, height) > MAX_SIDE:
        scale = MAX_SIDE / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)),
                             Image.BILINEAR)

    predictions = []
    for phrase in [p.strip() for p in (prompt or "").split(",") if p.strip()]:
        inputs = processor(images=image, text=phrase, return_tensors="pt")
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=float(threshold),
            target_sizes=[(image.size[1], image.size[0])])[0]

        boxes = result.get("boxes")
        scores = result.get("scores")
        if boxes is None or scores is None:
            continue
        for box, score in zip(boxes.tolist(), scores.tolist()):
            if score < threshold:
                continue
            # Undo the MAX_SIDE resize, then emit centre-based boxes so the
            # payload matches the Roboflow workflow's shape exactly.
            x1, y1, x2, y2 = (v / scale for v in box)
            predictions.append({
                "class": phrase,
                "confidence": round(float(score), 4),
                "x": round((x1 + x2) / 2, 1),
                "y": round((y1 + y2) / 2, 1),
                "width": round(x2 - x1, 1),
                "height": round(y2 - y1, 1),
            })

    return {"image": {"width": width, "height": height},
            "predictions": predictions,
            "device": DEVICE}


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(label="image (base64)", lines=2),
            gr.Textbox(label="classes (comma separated)",
                       value="mobile phone,paper chit,keyboard"),
            gr.Number(label="threshold", value=0.4)],
    outputs=gr.JSON(label="predictions"),
    title="SAM 3 exam-hall referee",
    description=("Open-vocabulary segmentation for the Drishti AI Project "
                 "Classroom pipeline. Send a base64 image and a comma-"
                 "separated prompt; get centre-based boxes back. Masks are "
                 "not returned."),
    api_name="predict",
)

if __name__ == "__main__":
    demo.queue(max_size=32).launch()
