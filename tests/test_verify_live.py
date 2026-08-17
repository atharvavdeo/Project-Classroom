"""Gemma verifier executed against real exam-centre footage.

The final unproven path: a contact sheet built from real frames, sent to the
production model through the production code path, constrained by the production
grammar, and parsed by the production schema.

This tests plumbing and safety, not accuracy. The model runs on CPU here and
sees a genuinely ambiguous scene; what matters is that the response is
schema-valid, stays inside the enumerated vocabulary, carries no verdict
language, and that the abstention-friendly prompt actually produces uncertainty
rather than confident object claims.

    tools/llamacpp/llama-server.exe -m models/gemma/gemma-4-E4B_q4_0-it.gguf \
        --mmproj models/gemma/gemma-4-E4B-it-mmproj.gguf -c 4096 --port 8098
    python tests/test_verify_live.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from classroom import crops, verify  # noqa: E402
from classroom.config import VerifyConfig  # noqa: E402

BASE = "http://127.0.0.1:8098"
FRAMES_DIR = Path(
    r"C:\Users\HARDEE~1\AppData\Local\Temp\claude\C--DrishtiAI"
    r"\d9cbb711-8a41-4ee8-9666-c80f8eaac1b5\scratchpad\frames"
)
SEQUENCE = ["paper_t0020.jpg", "paper_t0045.jpg", "paper_t0070.jpg"]
SEAT = [(30, 300), (330, 300), (330, 620), (30, 620)]


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def main() -> int:
    cfg = VerifyConfig()
    verifier = verify.GemmaVerifier(cfg, base_url=BASE)
    if not verifier.health():
        print(f"SKIP: no llama-server at {BASE}")
        return 0

    paths = [FRAMES_DIR / n for n in SEQUENCE]
    if any(not p.is_file() for p in paths):
        print("SKIP: extracted frames not available")
        return 0

    ok = True
    print("building a contact sheet from real frames")
    frames = [cv2.imread(str(p)) for p in paths]
    # Event shape: before, onset, peak, end, plus the magnified seat crop.
    crop, transform = crops.seat_crop(frames[1], SEAT, 640, context_px=0)
    sheet_frames = [frames[0], frames[1], frames[2], crop]
    sheet = verify.build_contact_sheet(sheet_frames, columns=2, tile=384)
    sheet_path = Path("docs/verifier_contact_sheet.jpg")
    sheet_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  {sheet.shape[1]}x{sheet.shape[0]}, "
          f"{sheet_path.stat().st_size / 1e3:.0f} KB -> {sheet_path}")
    ok &= check("seat crop magnified into the sheet", transform.supersampling > 1.0,
                f"{transform.supersampling:.2f}x")

    metadata = {
        "seat": "L-front",
        "event_duration_s": 4.7,
        "peak_deviation_sd": 7.0,
        "below_desk_fraction": 0.0,
        "detector_object_evidence": None,
        "source_resolution": "1280x720",
        "note": "Object evidence unavailable; detector not fine-tuned for this domain.",
    }

    print("\nrunning the production verifier path (CPU, expect minutes)")
    started = time.perf_counter()
    try:
        result = verifier.verify(sheet_path, metadata, timeout=1800)
    except verify.TruncatedVerification as exc:
        print(f"  truncated: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    print(f"  completed in {elapsed:.0f}s")

    print("\nresponse")
    for obs in result.supported_observations:
        print(f"    observation: {obs}")
    print(f"    object:      {result.object_assessment}")
    print(f"    interaction: {result.interaction_assessment}")
    print(f"    quality:     {result.evidence_quality}")
    print(f"    confidence:  {result.confidence}")
    print(f"    note:        {result.review_note}")

    print("\nschema and vocabulary")
    ok &= check("parsed and validated", isinstance(result, verify.Verification))
    ok &= check("object in vocabulary",
                result.object_assessment in verify.OBJECT_VALUES)
    ok &= check("interaction in vocabulary",
                result.interaction_assessment in verify.INTERACTION_VALUES)
    ok &= check("quality in vocabulary",
                result.evidence_quality in verify.QUALITY_VALUES)
    ok &= check("confidence is a probability", 0.0 <= result.confidence <= 1.0)
    ok &= check("observation list bounded by the grammar",
                len(result.supported_observations) <= 5,
                str(len(result.supported_observations)))

    print("\nFR-19: no verdict reaches the reviewer")
    ok &= check("review note carries no verdict",
                not verify.contains_verdict(result.review_note))
    ok &= check("no observation carries a verdict",
                not any(verify.contains_verdict(o)
                        for o in result.supported_observations))
    clean, removed = verify.sanitize(result)
    ok &= check("sanitisation leaves a usable record",
                isinstance(clean, verify.Verification))
    if removed:
        print(f"    (sanitiser removed {len(removed)} statement(s))")

    print("\nabstention: the prompt discourages confident object claims")
    ok &= check("object claim is not overconfident",
                result.object_assessment != "phone_like" or result.confidence < 0.9,
                f"{result.object_assessment} @ {result.confidence}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
