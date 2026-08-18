"""Phase 7 validation gate: verifier contract, verdict suppression, contact sheet.

The model weights are not needed to test the parts that matter for safety. What
is checked here is that a malformed or unsafe response cannot reach a reviewer:
the schema is enforced, out-of-vocabulary values are rejected, and any statement
phrased as a verdict rather than an observation is stripped and reported.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from classroom import db, verify  # noqa: E402
from classroom.config import VerifyConfig  # noqa: E402
from classroom.verify import Verification  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


GOOD = {
    "supported_observations": ["candidate's right hand moves below the desk line"],
    "object_assessment": "uncertain",
    "interaction_assessment": "self_action",
    "evidence_quality": "limited",
    "confidence": 0.4,
    "review_note": "Hand lowered below desk for several seconds; object not resolvable.",
}


def main() -> int:
    ok = True

    print("schema: well-formed response parses")
    v = Verification.parse(json.dumps(GOOD))
    ok &= check("parsed", v.object_assessment == "uncertain")
    ok &= check("confidence preserved", abs(v.confidence - 0.4) < 1e-9)
    ok &= check("observations preserved", len(v.supported_observations) == 1)

    print("\nschema: out-of-vocabulary values rejected")
    for field, bad in [
        ("object_assessment", "definitely_a_phone"),
        ("interaction_assessment", "copying"),
        ("evidence_quality", "conclusive"),
    ]:
        payload = dict(GOOD, **{field: bad})
        try:
            Verification.parse(json.dumps(payload))
            ok &= check(f"rejects {field}={bad}", False)
        except ValueError:
            ok &= check(f"rejects {field}={bad}", True)

    print("\nschema: confidence must be a probability")
    for bad in (1.5, -0.2):
        try:
            Verification.parse(json.dumps(dict(GOOD, confidence=bad)))
            ok &= check(f"rejects confidence={bad}", False)
        except ValueError:
            ok &= check(f"rejects confidence={bad}", True)

    print("\nschema: missing field is an error, not a default")
    incomplete = {k: v for k, v in GOOD.items() if k != "evidence_quality"}
    try:
        Verification.parse(json.dumps(incomplete))
        ok &= check("rejects missing field", False)
    except KeyError:
        ok &= check("rejects missing field", True)

    print("\nFR-19: verdict language never reaches a reviewer")
    for phrase in ["the student was cheating", "confirmed malpractice",
                   "this candidate is guilty", "clear misconduct observed",
                   "caught copying from a neighbour"]:
        ok &= check(f"flags {phrase!r}", verify.contains_verdict(phrase))
    for phrase in ["right hand moves below the desk line",
                   "head turned toward the neighbouring seat for 6 seconds",
                   "object not visible at this resolution"]:
        ok &= check(f"allows {phrase!r}", not verify.contains_verdict(phrase))

    print("\nsanitize: verdicts stripped, observations kept")
    tainted = Verification.parse(json.dumps(dict(
        GOOD,
        supported_observations=[
            "right hand moves below the desk line",
            "the candidate is clearly cheating",
            "head turns toward the neighbouring desk",
        ],
        review_note="Confirmed malpractice by seat 12.",
    )))
    clean, removed = verify.sanitize(tainted)
    ok &= check("verdict observation removed", len(clean.supported_observations) == 2)
    ok &= check("legitimate observations kept",
                "right hand moves below the desk line" in clean.supported_observations)
    ok &= check("verdict note replaced", not verify.contains_verdict(clean.review_note))
    ok &= check("removals reported, not silent", len(removed) == 2, f"{len(removed)}")
    ok &= check("assessments untouched", clean.object_assessment == "uncertain")

    unchanged, none_removed = verify.sanitize(v)
    ok &= check("clean response passes through untouched", not none_removed)

    print("\ngrammar covers every enumerated value")
    for value in verify.OBJECT_VALUES:
        ok &= check(f"grammar has object {value}", f'"\\"{value}\\""' in verify.GRAMMAR
                    or f'"{value}"' in verify.GRAMMAR)
    for value in verify.INTERACTION_VALUES:
        ok &= check(f"grammar has interaction {value}", value in verify.GRAMMAR)
    for value in verify.QUALITY_VALUES:
        ok &= check(f"grammar has quality {value}", value in verify.GRAMMAR)
    ok &= check("grammar pins every schema key",
                all(k in verify.GRAMMAR for k in GOOD))

    print("\nprompt steers away from object identification")
    ok &= check("forbids verdict language", "Never state or imply" in verify.SYSTEM)
    ok &= check("offers abstention as a correct answer",
                "not_visible" in verify.SYSTEM and "uncertain" in verify.SYSTEM
                and "not failures" in verify.SYSTEM)
    # The measured confusers on this corpus are CBT desk equipment, not the
    # erasers and rulers a generic classroom prompt would guard against
    # (PRD 8.1.2). A model with no prior for what is normally on the desk has
    # nothing to weigh "phone" against.
    for item in ("monitor", "keyboard", "mouse"):
        ok &= check(f"names {item} as normal desk equipment", item in verify.SYSTEM)
    ok &= check("fences the metadata off as non-observational",
                "not behaviour" in verify.SYSTEM
                or "not observations" in verify.USER_TEMPLATE)
    ok &= check("the user turn carries the tile legend",
                "{legend}" in verify.USER_TEMPLATE and "{count}" in verify.USER_TEMPLATE)

    print("\nthe tile legend is derived, not asserted")
    # The previous prompt described five tiles in prose while the configured
    # sheet held six, so the model was told the last tile was a magnified crop
    # when it was another timeline frame.
    for count in (4, 5, 6, 8):
        times = verify.select_frames(10.0, 16.0, 13.0, count)
        legend = verify.frame_legend(times, 10.0, 16.0, 13.0)
        ok &= check(f"legend has one line per tile at count={count}",
                    len(legend.splitlines()) == len(times),
                    f"{len(legend.splitlines())} vs {len(times)}")
    times = verify.select_frames(10.0, 16.0, 13.0, 6)
    legend = verify.frame_legend(times, 10.0, 16.0, 13.0)
    ok &= check("the peak tile is labelled as the peak",
                legend.count("peak of the flagged motion") == 1, legend)
    ok &= check("the pre-roll tile is labelled as before",
                legend.count("before the flagged interval") == 1)

    print("\ncontact sheet")
    frames = [np.full((720, 1280, 3), i * 40, dtype=np.uint8) for i in range(5)]
    sheet = verify.build_contact_sheet(frames, columns=3, tile=320)
    ok &= check("tiled to a grid", sheet.shape == (640, 960, 3), str(sheet.shape))
    ok &= check("content written", int(sheet.max()) > 0)
    single = verify.build_contact_sheet(frames[:1], columns=3, tile=320)
    ok &= check("single frame still tiles", single.shape == (320, 960, 3), str(single.shape))
    try:
        verify.build_contact_sheet([])
        ok &= check("empty sheet rejected", False)
    except ValueError:
        ok &= check("empty sheet rejected", True)

    print("\nframe selection covers the event shape")
    picks = verify.select_frames(t_start=10.0, t_end=16.0, peak_t=13.0, count=6)
    ok &= check("requested count returned", len(picks) == 6, str(len(picks)))
    ok &= check("ordered", picks == sorted(picks))
    ok &= check("includes pre-roll before onset", picks[0] < 10.0, f"{picks[0]:.2f}")
    ok &= check("spans to the event end", max(picks) >= 16.0 - 1e-9)
    ok &= check("peak included", any(abs(p - 13.0) < 1e-6 for p in picks))
    ok &= check("clamps pre-roll at zero",
                verify.select_frames(0.2, 3.0, 1.0, 4)[0] >= 0.0)

    print("\npersistence")
    with tempfile.TemporaryDirectory() as tmp:
        conn = db.connect(Path(tmp) / "v.db")
        session_id = db.insert(conn, "session", name="fixture")
        camera_id = db.insert(conn, "camera", session_id=session_id, label="cam")
        video_id = db.insert(
            conn, "source_video", camera_id=camera_id, path="x", sha256="y" * 64,
            size_bytes=1, container="mp4", codec="h264", pix_fmt="yuv420p",
            width=640, height=480, avg_fps=25.0, duration_s=10.0, frame_count=250,
            is_vfr=0, has_mv=1,
        )
        cal_id = db.insert(conn, "calibration", camera_id=camera_id, version=1,
                           frame_w=640, frame_h=480)
        seat_id = db.insert(conn, "seat", calibration_id=cal_id, seat_label="12",
                            polygon="[]")
        event_id = db.insert(conn, "event", source_video_id=video_id, seat_id=seat_id,
                             t_start=1.0, t_end=3.0, peak_t=2.0, peak_deviation=5.0)
        vid = verify.persist(conn, event_id, "gemma-4-E4B", clean)
        ok &= check("verification stored", vid > 0)
        row = db.one(conn, "SELECT * FROM vlm_verification WHERE id = ?", (vid,))
        ok &= check("observations round-trip",
                    len(json.loads(row["observations"])) == 2)
        ok &= check("no verdict text persisted",
                    not verify.contains_verdict(row["review_note"]))
        conn.close()

    print("\nan absent verifier endpoint fails loudly, not silently")
    gv = verify.GemmaVerifier(VerifyConfig(), base_url="http://127.0.0.1:59999")
    ok &= check("health check reports down", gv.health(timeout=1.0) is False)
    try:
        gv.load()
        ok &= check("unreachable server raises", False)
    except ConnectionError as exc:
        ok &= check("unreachable server raises", True, type(exc).__name__)
        ok &= check("error names the start command", "llama-server -m" in str(exc))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
