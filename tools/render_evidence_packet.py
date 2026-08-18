"""Render evidence packets and run the bounded Gemma review (PRD 11, 12).

    python tools/render_evidence_packet.py --run artifacts/runs/<run_id> \
        --calibration calib/cam12_e1_v1.json "C:/DrishtiAI/03.CCTV Mobile Usage.mkv"

    # with a llama-server already serving Gemma:
    python tools/render_evidence_packet.py ... --gemma-url http://127.0.0.1:8080/v1/chat/completions

Composes one packet per event, validates it against PRD 11 **before** rendering
or reviewing, writes the contact sheet, and calls Gemma only on packets that
pass. A packet without a marked and magnified subject is never sent: an earlier
phase of this project measured a verifier describing the wrong person at
confidence 1.0 from whole-frame tiles, and nothing in that output looked wrong.

Writes `10_evidence/events/<event_id>/` and `11_gemma/<event_id>/`, then ranks
everything into `10_evidence/ranked_events.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, evidence as ev, gemma_review as gr, ranking  # noqa: E402
from pipeline.calibration import SEAT_CONTEXT, CalibrationProfile  # noqa: E402
from pipeline.run_manifest import (COMPLETE, RESOURCE_UNAVAILABLE,  # noqa: E402
                                   RunManifest, RunStatus, SKIPPED_UNAVAILABLE)
from tools.run_pose_ab import SequentialFrameReader, read_jsonl  # noqa: E402


def seat_boxes(profile: CalibrationProfile) -> dict:
    out = {}
    for seat in profile.seats:
        polygon = seat.zones.get(SEAT_CONTEXT)
        if not polygon:
            continue
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        out[seat.seat_id] = (min(xs), min(ys), max(xs), max(ys))
    return out


def objects_for(event_id: str, groups: list[dict], config_id: str
                ) -> list[ev.ObjectEvidence]:
    """Object evidence for one event, from one detector configuration.

    One configuration, named on every tile. Mixing configurations into a packet
    would let a reviewer attribute one detector's finding to another, and PRD
    17.9 forbids configurations overwriting each other's evidence.
    """
    out = []
    for group in groups:
        if group["event_id"] != event_id or group["config_id"] != config_id:
            continue
        if group["cls"] == "insufficient_visual_evidence":
            continue
        box = group.get("box") or [0, 0, 0, 0]
        out.append(ev.ObjectEvidence(
            cls=group["cls"], confidence=group.get("max_confidence", 0.0),
            status=group["status"], config_id=group["config_id"],
            support_frames=group.get("support_frames", 1),
            x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
    out.sort(key=lambda o: -o.confidence)
    return out[:4]


def render_sheet(reader, packet: ev.EvidencePacket, out_path: Path,
                 tile_px: int = 480) -> bool:
    """Draw the contact sheet: marked full frames, then magnified crops."""
    try:
        import cv2
        import numpy as np
    except Exception:                                    # noqa: BLE001
        return False

    # Read every tile's frame in ascending PTS order, then compose in the
    # packet's own order.
    #
    # The reader is a single forward decode pass. The subject crop and the
    # object crops sit at the event's *peak*, which is earlier than the last
    # frame tile, so reading tiles in packet order asked the reader to go
    # backwards and it correctly refused -- silently dropping the magnified
    # subject crop from every sheet. The packet still validated, because
    # composition is checked against the described tiles; only the rendered
    # image was missing the one thing PRD 11 requires most.
    frames_by_tile: dict[int, object] = {}
    for index in sorted(range(len(packet.tiles)),
                        key=lambda i: packet.tiles[i].pts_ms):
        frames_by_tile[index] = reader(packet.tiles[index])

    images = []
    for index, tile in enumerate(packet.tiles):
        frame = frames_by_tile.get(index)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        x1, y1 = int(max(tile.x1, 0)), int(max(tile.y1, 0))
        x2, y2 = int(min(tile.x2, width)), int(min(tile.y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        view = frame[y1:y2, x1:x2].copy()

        if tile.role == ev.FULL_FRAME and tile.subject_marked:
            # The red box and label, taken from the subject crop's source
            # rectangle. Without them the tile shows a room, not a subject.
            box = packet.subject_crops[0] if packet.subject_crops else None
            if box is not None:
                cv2.rectangle(view, (int(box.x1), int(box.y1)),
                              (int(box.x2), int(box.y2)), (0, 0, 255), 3)
                cv2.putText(view, tile.subject_label,
                            (int(box.x1), max(int(box.y1) - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        scale = tile_px / max(view.shape[1], 1)
        view = cv2.resize(view, (tile_px, max(int(view.shape[0] * scale), 1)),
                          interpolation=cv2.INTER_CUBIC if scale > 1
                          else cv2.INTER_AREA)
        # PTS on every tile (PRD 11), so no image can be cited without it.
        strip = np.zeros((26, view.shape[1], 3), dtype=view.dtype)
        cv2.putText(strip, f"{len(images) + 1}. {tile.caption}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        images.append(np.vstack([strip, view]))

    if not images:
        return False

    cols = 3
    rows = []
    height = max(i.shape[0] for i in images)
    for start in range(0, len(images), cols):
        chunk = images[start:start + cols]
        padded = []
        for image in chunk:
            canvas = np.zeros((height, tile_px, 3), dtype=image.dtype)
            canvas[:image.shape[0], :image.shape[1]] = image[:height, :tile_px]
            padded.append(canvas)
        while len(padded) < cols:
            padded.append(np.zeros((height, tile_px, 3), dtype=images[0].dtype))
        rows.append(np.hstack(padded))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--object-config", default="dfine_native")
    ap.add_argument("--gemma-url", default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    if paths.sealed:
        print(f"{args.run} is sealed; reprocessing needs a new run.",
              file=sys.stderr)
        return 2
    status = RunStatus(paths, RunManifest.read(paths).run_id)
    profile = CalibrationProfile.read(args.calibration)
    boxes = seat_boxes(profile)

    events = read_jsonl(paths.dir / "06_segmentation/events.jsonl")
    candidates = {c["event_id"]: c for c in
                  read_jsonl(paths.dir / "05_motion_roi/candidate_manifest.jsonl")}
    groups = read_jsonl(paths.dir / "10_evidence/temporal_confirmation.jsonl")
    if not events:
        print("no events; run tools/run_motion_scan.py first.", file=sys.stderr)
        return 2

    # --------------------------------------------------- compose packets --
    status.begin("10_evidence")
    cfg = ev.PacketConfig()
    packets = []
    for event in events:
        candidate = candidates.get(event["event_id"], {})
        merged = {**event, **{k: v for k, v in candidate.items()
                              if k not in event or not event.get(k)}}
        packet = ev.build(
            merged, boxes.get(merged.get("seat_id", "")),
            objects_for(event["event_id"], groups, args.object_config),
            profile.frame_w, profile.frame_h, cfg,
            health_state=candidate.get("health_state", []),
            config_hashes={"calibration": profile.version_id,
                           "object_config": args.object_config})
        packets.append(packet)

    packet_summary = ev.summarise(packets)
    paths.write_json("10_evidence/packet_summary.json", packet_summary)
    for packet in packets:
        directory = paths.dir / f"10_evidence/events/{packet.event_id}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "evidence.json").write_text(
            json.dumps(packet.to_row(), indent=2), encoding="utf-8")

    print(f"{len(packets)} packets composed: {packet_summary['valid']} valid, "
          f"{packet_summary['invalid']} invalid")
    for reason, count in packet_summary["invalid_reasons"].items():
        print(f"    {count:3}  {reason}")

    # ------------------------------------------------------ contact sheets --
    rendered = 0
    if not args.no_render:
        for packet in packets:
            if not packet.valid:
                continue
            # One reader per packet. Events overlap in time, so the next
            # packet's pre-roll routinely precedes the previous packet's last
            # tile; a shared forward-only reader refuses those and the sheet
            # silently comes out short. The reader stops decoding once it has
            # the last frame the packet needs, so this is not a full pass each
            # time.
            reader = SequentialFrameReader(args.video)
            out = paths.dir / f"11_gemma/{packet.event_id}/contact_sheet.png"
            if render_sheet(reader, packet, out):
                rendered += 1
            else:
                print(f"  no frames decoded for {packet.event_id}; the sheet is "
                      f"absent rather than partial")
            reader.close()
        print(f"{rendered} of {sum(1 for p in packets if p.valid)} "
              f"contact sheets rendered")

    # ------------------------------------------------------ Gemma review --
    status.begin("11_gemma")
    valid_packets = [p for p in packets if p.valid]
    if args.gemma_url:
        runner = gr.llamacpp_runner(args.gemma_url)
        reviews = gr.review(valid_packets, runner)
    else:
        reviews = gr.unavailable(
            valid_packets, RESOURCE_UNAVAILABLE,
            "no --gemma-url was supplied, so no VLM review was performed. Every "
            "packet is recorded as needing human review rather than being "
            "treated as reviewed and clean.")
    # Blocked packets are recorded too, so a reviewer can see which events the
    # VLM was never allowed to see and why.
    reviews += [gr.Review(event_id=p.event_id, state="blocked",
                          reason="; ".join(p.validate(cfg)))
                for p in packets if not p.valid]

    for review in reviews:
        directory = paths.dir / f"11_gemma/{review.event_id}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "raw_response.txt").write_text(review.raw_response,
                                                    encoding="utf-8")
        (directory / "validation.json").write_text(
            json.dumps(review.to_row(), indent=2), encoding="utf-8")
        if review.parsed is not None:
            (directory / "parsed.json").write_text(
                json.dumps(review.parsed, indent=2), encoding="utf-8")
        if review.usable or review.state == "available":
            (directory / "prompt.json").write_text(
                json.dumps(gr.build_prompt(
                    next(p for p in packets if p.event_id == review.event_id)),
                    indent=2), encoding="utf-8")

    review_summary = gr.summarise(reviews)
    paths.write_json("11_gemma/summary.json", review_summary)
    print(f"\nGemma: {review_summary['attempted']} attempted "
          f"({review_summary['schema_valid']} schema-valid, "
          f"{review_summary['schema_invalid']} invalid, "
          f"{review_summary['repaired']} repaired), "
          f"{review_summary['not_attempted']} not attempted")
    if review_summary["not_attempted_reason"]:
        print(f"  {review_summary['not_attempted_reason']}")
    print(f"  {review_summary['needs_human_review']} of "
          f"{review_summary['reviews']} need human review")

    # ------------------------------------------------------------ ranking --
    reviews_by_event = {r.event_id: r for r in reviews}
    manifest = RunManifest.read(paths)
    ranked = ranking.rank(events, reviews_by_event=reviews_by_event)
    for row in ranked:
        paths.append_jsonl("10_evidence/ranked_events.jsonl", row.to_row())

    duration = max((e["end_pts_ms"] for e in events), default=0.0)
    status_path = paths.dir / "RUN_STATUS.json"
    analysed = duration
    if status_path.is_file():
        recorded = json.loads(status_path.read_text(encoding="utf-8"))
        # `stages` is a list of stage records, not a mapping. The motion stage
        # is the one that measured the whole decodable duration, so its
        # total_ms is the analysed span; the events' own extent is only the
        # part that was flagged.
        stage = next((s for s in recorded.get("stages", [])
                      if s.get("stage") == "05_motion_roi"), {})
        analysed = stage.get("total_ms") or duration

    report = ranking.summarise(
        ranked, analysed_ms=analysed,
        configurations={m["config_id"]: m["state"]
                        for m in manifest.to_dict().get("models", [])})
    paths.write_json("10_evidence/ranked_events.json", report)
    print(f"\n{report['headline']}")

    status.finish("10_evidence", COMPLETE, events=len(packets),
                  valid_packets=packet_summary["valid"])
    status.finish("11_gemma",
                  COMPLETE if any(r.usable for r in reviews) else SKIPPED_UNAVAILABLE,
                  reason=None if any(r.usable for r in reviews) else
                  "no VLM review was performed; every event needs human review",
                  reviews=len(reviews))
    status.write()
    print(f"\nwritten to {paths.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
