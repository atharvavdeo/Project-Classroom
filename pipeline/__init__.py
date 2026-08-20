"""Product Zero pipeline (PRD.md, RTX 3090 handoff).

New infrastructure lives here: the immutable run folder, the run manifest,
configuration validation, preflight, and the five-state gate protocol.

Existing measured code in `classroom/` is *adapted*, not rewritten. Modules move
across one at a time, each behind a test, so that no working and measured
component is replaced by an unmeasured one.
"""
# Register the CUDA runtime on the DLL search path before anything imports
# onnxruntime. Without this it finds no CUDA libraries, warns once, and falls
# back to the CPU provider silently -- the run still completes, just several
# times slower, which makes every timing in the artifacts misleading. See
# `cuda_paths.py` for why torch's bundled libraries are the right source.
from . import cuda_paths as _cuda_paths  # noqa: E402

_cuda_paths.ensure()
