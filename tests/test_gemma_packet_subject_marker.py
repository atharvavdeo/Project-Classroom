"""Phase 6 gate: the whole-frame-only packet must fail (PRD 11, PRD 18).

PRD 18's acceptance criterion is written as a test that must fail:

> Gemma evidence packets contain marked+magnified subject; whole-frame-only
> packet test fails.

This is that test. The failure it guards against is measured, not hypothetical:
in an earlier phase of this project the verifier described the wrong person at
confidence 1.0, having been handed frames containing a whole room. Nothing in
the output looked wrong. A fluent description of the wrong candidate is worse
than no description, because it is actionable and false.

The defence has to be structural. A prompt saying "describe only the marked
subject" is worthless when nothing is marked, so `review_packet` refuses to call
the model at all on a packet that fails composition — the check runs before the
runner is invoked, and the test proves the runner was never reached.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evidence import (  # noqa: E402
    FULL_FRAME, OBJECT_CROP, SUBJECT_CROP, EvidencePacket, ObjectEvidence,
    PacketConfig, Tile, build, frame_times, summarise,
)
from pipeline.gemma_review import build_prompt, legend, review_packet  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


EVENT = {
    "event_id": "EV-1", "seat_id": "S-61",
    "start_pts_ms": 10000.0, "end_pts_ms": 14000.0, "peak_pts_ms": 12000.0,
    "source_video": "v.mkv", "source_video_sha256": "abc123",
    "calibration_version": "cam12/e1/v1", "camera_epoch": "e1",
}
SEAT_BOX = (60.0, 280.0, 380.0, 700.0)


def good_packet(cfg=None) -> EvidencePacket:
    return build(EVENT, SEAT_BOX,
                 [ObjectEvidence("phone", 0.62,
                                 "temporally_supported_object_evidence",
                                 "dfine_native", 3, 200.0, 500.0, 232.0, 522.0,
                                 704.0)],
                 1280, 720, cfg or PacketConfig())


def main() -> int:
    ok = True
    cfg = PacketConfig()

    print("a properly composed packet validates")
    packet = good_packet()
    problems = packet.validate(cfg)
    ok &= check("no composition problems", not problems, str(problems))
    ok &= check("it has 4-8 full frames",
                cfg.min_frames <= len(packet.frame_tiles) <= cfg.max_frames,
                str(len(packet.frame_tiles)))
    ok &= check("it has a magnified subject crop", bool(packet.subject_crops))
    ok &= check("every frame marks the subject",
                all(t.subject_marked for t in packet.frame_tiles))
    ok &= check("every frame labels it",
                all(t.subject_label == "S-61" for t in packet.frame_tiles))
    ok &= check("every tile carries a PTS",
                all(t.pts_stamped for t in packet.tiles))
    ok &= check("the peak is on the sheet",
                any(abs(t.pts_ms - 12000.0) < 1.0 for t in packet.frame_tiles),
                "an evenly spaced sheet routinely straddles the one moment the "
                "event was flagged for")

    print("\nTHE GUARD: a whole-frame-only packet is rejected")
    whole_frame_only = EvidencePacket(
        event_id="EV-BAD", seat_state="S-61", source_video="v.mkv",
        source_video_sha256="abc123", start_pts_ms=10000.0, end_pts_ms=14000.0,
        peak_pts_ms=12000.0, calibration_version="cam12/e1/v1", camera_epoch="e1",
        tiles=[Tile(FULL_FRAME, 10000.0 + i * 1000, 0.0, 0.0, 1280.0, 720.0,
                    subject_marked=False, rendered_w=480, rendered_h=270)
               for i in range(6)])
    problems = whole_frame_only.validate(cfg)
    ok &= check("the packet is invalid", not whole_frame_only.valid)
    ok &= check("because no subject is marked",
                any("do not mark the subject" in p for p in problems),
                next((p for p in problems if "mark" in p), "")[:70])
    ok &= check("and because there is no magnified crop",
                any("magnified subject crop" in p for p in problems))

    calls = {"n": 0}

    def runner(prompt):
        calls["n"] += 1
        return "{}"

    review = review_packet(whole_frame_only, runner)
    ok &= check("the model was never called", calls["n"] == 0,
                "a prompt saying 'describe only the marked subject' is "
                "worthless when nothing is marked")
    ok &= check("the review is blocked, not empty", review.state == "blocked",
                review.state)
    ok &= check("the reason cites PRD 11", "PRD 11" in review.reason,
                review.reason[:60])
    ok &= check("it is not usable evidence", not review.usable)
    ok &= check("and it forces human review", review.needs_human_review,
                "a blocked packet is a reason for a human to look, not to skip")

    print("\na marked frame with no label is still rejected")
    unlabelled = good_packet()
    for tile in unlabelled.frame_tiles:
        tile.subject_label = ""
    ok &= check("invalid", not unlabelled.valid)
    ok &= check("the reason names the missing label",
                any("without labelling" in p for p in unlabelled.validate(cfg)))

    print("\ntoo few frames is rejected; a movement cannot be seen to develop")
    thin = good_packet()
    thin.tiles = [t for t in thin.tiles if t.role != FULL_FRAME][:1] + \
        thin.frame_tiles[:2]
    ok &= check("invalid", not thin.valid)
    ok &= check("the reason cites the minimum",
                any("minimum of 4" in p for p in thin.validate(cfg)),
                next((p for p in thin.validate(cfg) if "minimum" in p), "")[:60])

    print("\nout-of-order tiles are rejected")
    shuffled = good_packet()
    frames = shuffled.frame_tiles
    frames[0].pts_ms, frames[-1].pts_ms = frames[-1].pts_ms, frames[0].pts_ms
    ok &= check("invalid", not shuffled.valid)
    ok &= check("the reason names the ordering",
                any("chronological" in p for p in shuffled.validate(cfg)))

    print("\nmagnification is bounded, so a tile cannot imply absent detail")
    over = good_packet()
    over.subject_crops[0].rendered_w = 4000
    over.subject_crops[0].rendered_h = 4000
    ok &= check("invalid above the bound", not over.valid,
                f"{over.subject_crops[0].magnification:.1f}x")
    ok &= check("the reason says it reads as detail",
                any("reads as detail" in p for p in over.validate(cfg)))
    ok &= check("a within-bound crop is fine",
                good_packet().subject_crops[0].magnification
                <= cfg.max_subject_magnification,
                f"{good_packet().subject_crops[0].magnification:.2f}x")

    print("\nreference leakage into a packet is refused (PRD 15)")
    leaky = good_packet()
    object.__setattr__(leaky, "reference_id", "REF-1")
    ok &= check("invalid", not leaky.valid)
    ok &= check("the reason cites stage 12",
                any("stage 12" in p for p in leaky.validate(cfg)))

    print("\nthe legend is generated from the tiles, never asserted")
    text = legend(packet)
    ok &= check("it counts the tiles that exist",
                f"{len(packet.tiles)} numbered tiles" in text, text[:60])
    ok &= check("it states the frame timestamps",
                "12.00s" in text, "a legend and a sheet must not disagree")
    ok &= check("it describes the magnified crop",
                "magnified" in text and "source pixels" in text)

    fewer = good_packet(PacketConfig(max_frames=5))
    ok &= check("the legend follows a different tile count",
                f"{len(fewer.tiles)} numbered tiles" in legend(fewer),
                f"{len(fewer.tiles)} tiles")

    print("\nthe prompt carries the subject, the bounds and the grammar")
    prompt = build_prompt(packet)
    ok &= check("the seat is named", "S-61" in prompt["user"])
    ok &= check("the system message forbids describing others",
                "not under review" in prompt["system"])
    ok &= check("it fences the motion metadata off as reference",
                "reference only" in prompt["user"].lower())
    ok &= check("the detector's own observations are supplied",
                "phone at 0.62" in prompt["user"], prompt["user"][-120:])
    ok &= check("a grammar is attached", prompt["grammar"].startswith("root ::="))
    ok &= check("the prompt version travels with it",
                bool(prompt["prompt_version"]), prompt["prompt_version"])

    print("\nframe_times always includes the peak and stays in order")
    times = frame_times(10000.0, 14000.0, 12345.0, cfg)
    ok &= check("chronological", times == sorted(times))
    ok &= check("within the packet bound", len(times) <= cfg.max_frames,
                str(len(times)))
    ok &= check("the peak is present", 12345.0 in times, str(times))
    early = frame_times(0.0, 2000.0, 500.0, cfg)
    ok &= check("an event at the start of the video does not go negative",
                all(t >= 0.0 for t in early), str(early[:3]))

    print("\nthe summary counts invalid packets rather than hiding them")
    report = summarise([packet, whole_frame_only])
    ok &= check("both are counted", report["packets"] == 2)
    ok &= check("one is invalid", report["invalid"] == 1)
    ok &= check("with reasons", bool(report["invalid_reasons"]))
    ok &= check("an empty set is not 'nothing happened'",
                "never 'nothing" not in summarise([])["note"]
                and "valid outcome" in summarise([])["note"])

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
