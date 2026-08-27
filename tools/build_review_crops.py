"""Cut the frames a reviewer actually looks at, one per candidate.

The console reviews one person at a time and asks a human yes or no. That is
only answerable against a picture, so this writes, per track that carries a
proposal:

  - `key.jpg`    the single most significant moment, cropped around the person
  - `frame.jpg`  the same instant full-frame, because the real question is
                 *whose object was that*, and only the wide shot answers it
  - `strip/*`    sampled moments across the track for the contact sheet

**Significance is SAM 3's verdict, not the detector's score.** A supported
frame outranks any unsupported one however confident D-FINE was, because the
referee looked at that crop specifically and named the object. Ranking by raw
confidence put a 0.90 shirt above a 0.55 verified phone.

**On a supported frame only the supported object is drawn.** The other
proposals in that instant are equipment context and drawing them buries the one
box the reviewer is being asked about. On an unsupported frame everything is
drawn, dimmed, because there the question is still open.

Usage:
    python tools/build_review_crops.py --run artifacts/runs/1512/12_paper \\
        --video "<source>.mkv" --out console/public/crops/1512_12_paper
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

# BGR, mirroring the console palette so a crop and its timeline mark read as
# the same thing.
SUPPORTED_BGR = (92, 130, 229)   # --lane-object, the one box that matters
CONTEXT_BGR = (150, 150, 150)    # equipment and unverified proposals
PERSON_BGR = (230, 207, 138)     # --accent-2

# SAM 3 verdicts that mean "the referee backed this object claim".
SUPPORTING = ("corroborated", "phone_supported", "reclassified_as_phone")


def verdict_of(obj: dict) -> str:
    sam3 = obj.get("sam3") or {}
    return str(sam3.get("verdict") or "")


def is_supported(obj: dict) -> bool:
    return verdict_of(obj) in SUPPORTING


def _pad_box(box, width, height, pad=0.55):
    """Widen a person box so the crop shows the desk, not just the torso."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    return (max(int(x1 - w * pad), 0), max(int(y1 - h * pad * 0.6), 0),
            min(int(x2 + w * pad), width), min(int(y2 + h * pad * 0.6), height))


def _draw(frame, person_box, objects, offset=(0, 0), person=True):
    ox, oy = offset
    if person and person_box:
        x1, y1, x2, y2 = [int(v) for v in person_box[:4]]
        cv2.rectangle(frame, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy),
                      PERSON_BGR, 2)
    for obj in objects:
        box = obj.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        supported = is_supported(obj)
        cv2.rectangle(frame, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy),
                      SUPPORTED_BGR if supported else CONTEXT_BGR,
                      3 if supported else 1)
    return frame


def _sample_rows(rows: list, limit: int) -> list:
    """Evenly spaced picks, always including both ends."""
    if len(rows) <= limit:
        return rows
    step = (len(rows) - 1) / (limit - 1)
    return [rows[round(i * step)] for i in range(limit)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", default="samples_sam3_adjudicated.jsonl")
    ap.add_argument("--strip", type=int, default=12)
    args = ap.parse_args()

    stage = args.run / "14_person_timeline"
    path = stage / args.samples
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        return 1
    sam3_dir = stage / "sam3_adjudication"

    by_track: dict[int, list] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("near_hand_objects"):
            by_track[int(row.get("track_id", -1))].append(row)

    if not by_track:
        print("no track carries a proposal; nothing to cut")
        return 0

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.out.mkdir(parents=True, exist_ok=True)
    strip_dir = args.out / "strip"
    strip_dir.mkdir(exist_ok=True)
    sam3_out = args.out / "sam3"
    manifest: dict[str, dict] = {}

    def read_at(pts_ms: float):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(round(pts_ms / 1000 * fps)), 0))
        ok, frame = cap.read()
        return frame if ok else None

    for track_id, rows in sorted(by_track.items()):
        rows.sort(key=lambda r: r["pts_ms"])

        best = None
        best_rank = (-1, -1.0)
        supported_total = 0
        for row in rows:
            for obj in row.get("near_hand_objects") or []:
                supported = is_supported(obj)
                supported_total += int(supported)
                rank = (1 if supported else 0, float(obj.get("confidence") or 0))
                if rank > best_rank:
                    best, best_rank = (row, obj), rank

        if best is None:
            continue
        key_row, key_obj = best
        key_supported = is_supported(key_obj)
        key_pts = key_row["pts_ms"]

        # On a supported frame, draw only the box the referee backed.
        draw_objects = (
            [o for o in key_row["near_hand_objects"] if is_supported(o)]
            if key_supported else key_row["near_hand_objects"]
        )

        entry: dict = {
            "track_id": track_id,
            "key_pts_ms": key_pts,
            "key_class": key_obj.get("cls"),
            "key_confidence": round(float(key_obj.get("confidence") or 0), 4),
            "key_verdict": verdict_of(key_obj) or None,
            "key_supported": key_supported,
            "wrist_distance_norm": key_obj.get("wrist_distance_norm"),
            "nearest_wrist": key_obj.get("nearest_wrist"),
            "supported_frames": supported_total,
            "proposals": sum(len(r["near_hand_objects"]) for r in rows),
            "sightings": len(rows),
            "strip": [],
        }

        # SAM 3's own annotated crop for that instant, when the referee was
        # actually called on it. This is the segmentation the reviewer asked to
        # see, and it exists only for adjudicated frames.
        sam3_name = f"{int(key_pts)}ms_trk{track_id}.jpg"
        if (sam3_dir / sam3_name).exists():
            # Copy it next to the other crops. Naming the file in the manifest
            # without putting it where the manifest says it is produced a
            # console tab that pointed at a 404 -- the reviewer's complaint was
            # that SAM 3's own output was "not visible", and this was why.
            sam3_out.mkdir(exist_ok=True)
            shutil.copyfile(sam3_dir / sam3_name, sam3_out / sam3_name)
            entry["sam3_crop"] = sam3_name

        frame = read_at(key_pts)
        if frame is not None:
            wide = frame.copy()
            _draw(wide, key_row.get("box"), draw_objects)
            cv2.imwrite(str(args.out / f"trk{track_id}_frame.jpg"), wide,
                        [cv2.IMWRITE_JPEG_QUALITY, 82])

            x1, y1, x2, y2 = _pad_box(
                key_row.get("box") or [0, 0, width, height], width, height)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size:
                # The clean crop is written first, before anything is drawn on
                # it. A vision model shown the annotated version describes our
                # own box back to us -- measured: Gemma reported "a small,
                # orange-bordered rectangular object" on track 26, which is the
                # rectangle this tool drew. Asking a second opinion about a
                # picture we have already marked up is not a second opinion.
                cv2.imwrite(str(args.out / f"trk{track_id}_raw.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 86])
                entry["raw"] = f"trk{track_id}_raw.jpg"

                _draw(crop, key_row.get("box"), draw_objects,
                      offset=(x1, y1), person=False)
                cv2.imwrite(str(args.out / f"trk{track_id}_key.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 86])
                entry["key"] = f"trk{track_id}_key.jpg"
                entry["frame"] = f"trk{track_id}_frame.jpg"

        for row in _sample_rows(rows, args.strip):
            f = read_at(row["pts_ms"])
            if f is None:
                continue
            x1, y1, x2, y2 = _pad_box(row.get("box") or [0, 0, width, height],
                                      width, height, pad=0.35)
            crop = f[y1:y2, x1:x2].copy()
            if not crop.size:
                continue
            objects = row["near_hand_objects"]
            _draw(crop, None, objects, offset=(x1, y1), person=False)
            name = f"trk{track_id}_{int(row['pts_ms'])}.jpg"
            cv2.imwrite(str(strip_dir / name), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 74])
            entry["strip"].append({
                "pts_ms": row["pts_ms"],
                "file": f"strip/{name}",
                "supported": any(is_supported(o) for o in objects),
                "n": len(objects),
            })

        manifest[str(track_id)] = entry

    cap.release()
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8")

    supported = sum(1 for e in manifest.values() if e["key_supported"])
    strips = sum(len(e["strip"]) for e in manifest.values())
    print(f"{len(manifest)} track(s) -> {args.out}")
    print(f"  {supported} with a SAM 3 supported key frame")
    print(f"  {sum(1 for e in manifest.values() if 'sam3_crop' in e)} "
          f"with SAM 3's own annotated crop")
    print(f"  {strips} contact-sheet frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
