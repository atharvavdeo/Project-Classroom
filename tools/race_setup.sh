#!/usr/bin/env bash
# Bootstrap a rented GPU pod and start draining the job queue.
#
#   bash tools/race_setup.sh
#
# Written for a Race Engineering pod on the "Jupyter PyTorch" template, which
# already has CUDA and torch. It installs only what this pipeline adds on top,
# verifies the GPU is real before spending money on it, and then runs one job.
#
# The pod is billed by the hour and its disk is temporary, so this script is
# deliberately ordered by cost: the cheap checks that can abort the whole thing
# run first, and nothing large is downloaded until the GPU has been confirmed.
#
# Required environment (export these before running):
#   DRISHTI_CONSOLE       the console URL
#   DRISHTI_AGENT_TOKEN   authorises claiming work
#   ROBOFLOW_API_KEY      stages 14b and 14d
set -euo pipefail

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mSTOP: %s\033[0m\n' "$*" >&2; exit 1; }

for var in DRISHTI_CONSOLE DRISHTI_AGENT_TOKEN ROBOFLOW_API_KEY; do
  [ -n "${!var:-}" ] || die "$var is not set. Nothing will work without it."
done

say "GPU check — this decides whether it is worth continuing"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || die "nvidia-smi failed. This pod has no usable GPU; do not pay for it."

python - <<'PY' || die "torch cannot see the GPU. Stop here: on CPU this run takes hours, not minutes."
import sys
try:
    import torch
except ImportError:
    sys.exit("torch is not installed on this pod")
print(f"torch {torch.__version__}")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
print(f"device {torch.cuda.get_device_name(0)}")
print(f"vram   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PY

say "Console reachable?"
curl -fsS "${DRISHTI_CONSOLE}/api/health" || die "cannot reach ${DRISHTI_CONSOLE}"
echo

say "Installing pipeline dependencies"
# onnxruntime-gpu, not onnxruntime: the CPU wheel loads happily and then runs
# D-FINE on the CPU while reporting success, which is the most expensive silent
# failure available on a machine rented by the hour.
pip install --quiet --upgrade pip
pip install --quiet \
  "onnxruntime-gpu" \
  "rtmlib" \
  "supervision" \
  "av" \
  "opencv-python-headless" \
  "requests" \
  "pyyaml" \
  "numpy"

python - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("onnxruntime providers:", ", ".join(providers))
if "CUDAExecutionProvider" not in providers:
    print("  WARNING: no CUDA provider. Object detection will run on the CPU.")
PY

say "Fetching the D-FINE weights"
# The person/object detector is the one model with no automatic download.
mkdir -p models/dfine/onnx
if [ ! -f models/dfine/onnx/model.onnx ]; then
  echo "models/dfine/onnx/model.onnx is missing."
  echo "Copy it up from your own machine before running a job, e.g.:"
  echo "  scp models/dfine/onnx/model.onnx <pod>:$(pwd)/models/dfine/onnx/"
  die "no detector weights"
fi

say "Pose backend"
# AlphaPose needs a 333 MB checkpoint that is not in git. rtmlib downloads its
# own ONNX weights on first use, so a fresh pod uses it automatically -- see
# --pose-backend in tools/run_person_timeline.py. It is a *different model* and
# the run records which one produced the result.
if [ -f models/alphapose/fast_421_res152_256x192.pth ]; then
  echo "AlphaPose checkpoint present — using the measured baseline."
else
  echo "No AlphaPose checkpoint — the run will use rtmlib and say so."
fi

say "Taking one job"
# --once first, always. A wiring mistake then costs one recording rather than
# the whole queue, and you can watch the stage names go past before committing
# GPU hours to it.
python tools/gpu_agent.py --once

say "Done. To drain the rest of the queue:"
echo "  python tools/gpu_agent.py"
