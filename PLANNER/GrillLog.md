---
status: complete
mode: A
---
# Grill Log

## Raw idea
Before training, move Project Classroom toward an inference-first, robust video-processing architecture. Evaluate and choose pose, multi-person tracking, small-object detection, hand-object refinement, RF-DETR versus D-FINE, and SAHI for offline 720p examination-hall footage with roughly 30–60 people, using an RTX 3090. Training on SCB or local data is deferred; first produce an implementation plan for pure inference and customizations.

## Resolved
### Q1: What camera topology and deployment hardware define Product Zero?
**A:** Product Zero processes one fixed camera per examination hall, offline, on a friend's NVIDIA RTX 3090 laptop. Multiple timestamp-synchronised overlapping camera views are future work. SelfPose3D is deferred and will only be tested later against the same evaluation frames and metrics after the single-camera pipeline is established. AlphaPose is the selected full-body pose challenger. macOS is used only for independent preprocessing and smaller experiments, not the complete model stack.

### Q2: What evidence and evaluation outputs are mandatory?
**A:** Every preprocessing step must timestamp and save reviewable images to one common output folder. Every model detection must be saved and evaluated. The known frames containing phones and chits are evaluation ground truth: the system must run on the entire video, then compare outputs against those identified frames; it must not infer only from those frames. The PRD and implementation instructions must be detailed enough that a coding agent or implementer makes no unstated architectural decisions.

### Q3: Does the shared Product Zero scope accurately define the plan?
**A:** Yes. Do not include MacBook implementation. Build the entire systematic NVIDIA RTX 3090 pipeline, including all pre-training mitigations, model comparison, whole-video evaluation against identified phone/chit frames, and Gemma evaluation of flagged frames.

## Open
