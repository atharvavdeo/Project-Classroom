---
title: SAM3 Exam Referee
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: other
---

# SAM 3 exam-hall referee

> **NOT DEPLOYED.** Creating this Space returned `402 Payment Required`:
> *"Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on
> free cpu-basic requires a PRO subscription."* The files are complete and
> deploy in one command once PRO is active. Until then the pipeline uses the
> hosted Roboflow workflow, which serves the same checkpoint.
>
> `facebook/sam3` is also **not served by any HF Inference provider** — its
> `inferenceProviderMapping` is empty and the router returns *"Model not
> supported by provider hf-inference"* — so the serverless API is not an
> alternative either.

Text-promptable segmentation (`facebook/sam3`) exposed as an API for the
Drishti AI Project Classroom pipeline.

It exists because the pipeline's paper detector has one class and cannot say
"that is a keyboard". Measured on our own footage: on the frame where the chit
detector returned `chit-paper 0.74` on a keyboard, prompting SAM 3 with
`"mobile phone, paper chit, keyboard"` returns six keyboards and zero chits.

## API

`POST /gradio_api/call/predict` with `[image_base64, prompt, threshold]`.

Returns `{"image": {"width": W, "height": H}, "predictions": [...]}` where each
prediction is `{"class", "confidence", "x", "y", "width", "height"}` —
**centre-based** boxes, matching the Roboflow workflow this replaces, so the
pipeline's parser is unchanged.

Masks are deliberately not returned. Nothing downstream reads them and they are
several KB each.
