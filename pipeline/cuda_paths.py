"""Make onnxruntime's CUDA provider loadable, without a system CUDA install.

`onnxruntime-gpu` links against the CUDA runtime and cuDNN by DLL name and
searches the process DLL path for them. On a machine with no CUDA Toolkit
installed it finds nothing, prints a warning, and **silently falls back to the
CPU provider** — the session still builds, the run still completes, and the
only symptom is that it is slow. A silent fallback is the worst failure mode
here, because the timings then mean something other than they appear to.

The libraries are already on disk: a CUDA build of torch ships them inside
`torch/lib`. Measured on this machine — `cublas64_12.dll`, `cublasLt64_12.dll`,
`cudart64_12.dll`, `cudnn64_9.dll` and the cuDNN sub-libraries — which is
exactly the CUDA 12 + cuDNN 9 pairing `onnxruntime-gpu` 1.22 asks for. So
rather than requiring a separate multi-gigabyte toolkit install, this registers
torch's directory on the DLL search path before onnxruntime is imported.

Import order matters: `os.add_dll_directory` only affects libraries loaded
*after* it runs, so `pipeline/__init__.py` calls this at package import, ahead
of any module that touches onnxruntime.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_STATE: dict[str, object] = {"done": False, "added": [], "reason": ""}


def torch_cuda_lib_dir() -> Path | None:
    """Where a CUDA build of torch keeps its CUDA runtime DLLs, if present."""
    try:
        import torch
    except Exception:                                        # noqa: BLE001
        return None
    lib = Path(torch.__file__).resolve().parent / "lib"
    if not lib.is_dir():
        return None
    # Only useful if it actually holds a CUDA runtime; a CPU build has the
    # directory but not the libraries.
    if not any(lib.glob("cudart64_*.dll")) and not any(lib.glob("libcudart*")):
        return None
    return lib


def ensure() -> dict:
    """Register CUDA library directories. Idempotent; safe with no GPU."""
    if _STATE["done"]:
        return dict(_STATE)
    _STATE["done"] = True

    if sys.platform != "win32":
        # Linux resolves these through the loader's own search path and the
        # nvidia-* wheels' RPATHs; there is nothing to add.
        _STATE["reason"] = "not Windows; the dynamic loader resolves CUDA itself"
        return dict(_STATE)

    added: list[str] = []
    lib = torch_cuda_lib_dir()
    if lib is not None:
        try:
            os.add_dll_directory(str(lib))
            added.append(str(lib))
        except (OSError, AttributeError) as exc:             # noqa: PERF203
            _STATE["reason"] = f"{type(exc).__name__}: {exc}"

    # The nvidia-* wheels, when installed, lay their DLLs out one directory per
    # component. Harmless to add when absent.
    for package in ("cublas", "cudnn", "cuda_runtime", "cufft", "curand"):
        candidate = (Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
                     / package / "bin")
        if candidate.is_dir():
            try:
                os.add_dll_directory(str(candidate))
                added.append(str(candidate))
            except OSError:
                pass

    _STATE["added"] = added
    if not added and not _STATE["reason"]:
        _STATE["reason"] = (
            "no CUDA runtime found. Install a CUDA build of torch "
            "(pip install torch --index-url https://download.pytorch.org/whl/cu128) "
            "or the NVIDIA runtime wheels; onnxruntime will use the CPU "
            "provider until then.")
    return dict(_STATE)


def report() -> dict:
    """What was registered, and whether a CUDA session can actually be built.

    Deliberately builds nothing: callers that want proof create a session and
    read `get_providers()` back, because the provider *list* being available is
    not the same as a session *using* it.
    """
    ensure()
    out = dict(_STATE)
    try:
        import onnxruntime as ort
        out["ort_version"] = ort.__version__
        out["ort_available_providers"] = ort.get_available_providers()
    except Exception as exc:                                 # noqa: BLE001
        out["ort_version"] = None
        out["ort_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["torch_cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["vram_mib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2 ** 20)
    except Exception:                                        # noqa: BLE001
        out["torch_version"] = None
    return out
