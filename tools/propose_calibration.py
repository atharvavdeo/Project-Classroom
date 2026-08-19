"""Propose a DRAFT calibration from real person detections (PRD 7).

    python tools/propose_calibration.py --video "C:/DrishtiAI/01....mkv" \
        --detector models/dfine/onnx/model.onnx --out calib/cam01_e1_v1.json

PRD 7 permits exactly this and bounds it in the same sentence:

> Automatic polygon proposal may assist, but never approves calibration.

So every profile this writes is `approval_state: draft` with no reviewer, and
`pipeline.calibration.attribute_box` refuses to name a seat for an unapproved
profile. Motion, ROI, segmentation, pose and object detection all still run --
that is PRD 3's "permit health diagnostic only; block seat-attributed
evidence". The seat zones here are a *measurement of where people actually sat*,
not an invention: they come from running the real person detector over frames
sampled across the whole recording and clustering the footprints that resulted.

What this cannot do, and does not pretend to do: decide which physical seat a
cluster corresponds to, mark ambiguous boundaries, exclude monitors and
reflections, or confirm the geometry under changed lighting. Those need a human,
and until one has done them the profile stays draft.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pipeline.artifacts import sha256_file  # noqa: E402
from pipeline.calibration import (  # noqa: E402
    DESK_WORK_SURFACE, DRAFT, LAP_BAG, SEAT_CONTEXT, STATIC_BACKGROUND,
    UPPER_BODY, CalibrationProfile, MaskRegion, Seat,
)


def sample_frames(video: Path, count: int):
    """Decode `count` frames spread across the whole recording, in one pass."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration_s = float(container.duration / 1e6) if container.duration else 0.0
        time_base = float(stream.time_base)
        wanted = [duration_s * (i + 0.5) / count for i in range(count)] \
            if duration_s else []
        out, index = [], 0
        for frame in container.decode(stream):
            if frame.pts is None or index >= len(wanted):
                continue
            pts_s = float(frame.pts * time_base)
            if pts_s >= wanted[index]:
                out.append((pts_s * 1000.0, frame.to_ndarray(format="bgr24")))
                index += 1
        return out


def detect_people(frames, detector_path: Path, confidence: float):
    """Run the real detector over the sampled frames. No synthetic boxes."""
    from classroom.crops import extract
    from classroom.detect import ONNXDetector

    detector = ONNXDetector(str(detector_path), input_size=640, class_map={})
    boxes = []
    for pts_ms, frame in frames:
        height, width = frame.shape[:2]
        canvas, transform = extract(frame, (0, 0, width, height), 640)
        for det in detector(canvas, transform, confidence, pts_ms):
            if det.cls == "person":
                boxes.append((pts_ms, *det.box, det.confidence))
    return boxes


def cluster(boxes, min_support: int, merge_px: float):
    """Group person boxes by where their feet land.

    Clustered on the *footprint* -- the bottom-centre of the box -- not the
    centre, for the same reason attribution scores the footprint: a leaning
    person's centre moves a long way while their seated position does not.

    A cluster is kept only if it was seen in `min_support` distinct frames. One
    person walking past produces a trail of single-frame clusters; somebody
    seated produces a dense one in the same place all recording.
    """
    points = []
    for pts_ms, x1, y1, x2, y2, score in boxes:
        points.append((pts_ms, (x1 + x2) / 2.0, y2, x1, y1, x2, y2))

    clusters: list[dict] = []
    for pts_ms, fx, fy, x1, y1, x2, y2 in points:
        placed = False
        for group in clusters:
            if abs(group["fx"] - fx) <= merge_px and abs(group["fy"] - fy) <= merge_px:
                n = group["n"]
                group["fx"] = (group["fx"] * n + fx) / (n + 1)
                group["fy"] = (group["fy"] * n + fy) / (n + 1)
                group["n"] = n + 1
                group["frames"].add(round(pts_ms, 1))
                group["boxes"].append((x1, y1, x2, y2))
                placed = True
                break
        if not placed:
            clusters.append({"fx": fx, "fy": fy, "n": 1,
                             "frames": {round(pts_ms, 1)},
                             "boxes": [(x1, y1, x2, y2)]})

    kept = [c for c in clusters if len(c["frames"]) >= min_support]
    kept.sort(key=lambda c: (-len(c["frames"]), c["fy"]))
    return kept, clusters


def zones_from(group, frame_w: int, frame_h: int):
    """Seat/desk/upper-body/lap polygons from a cluster's observed extent.

    Uses robust percentiles of the boxes actually seen, not the min/max: a
    single bad detection would otherwise stretch the zone across the room.
    """
    arr = np.array(group["boxes"], dtype=float)
    x1 = float(np.percentile(arr[:, 0], 10))
    y1 = float(np.percentile(arr[:, 1], 10))
    x2 = float(np.percentile(arr[:, 2], 90))
    y2 = float(np.percentile(arr[:, 3], 90))

    x1, y1 = max(x1, 0.0), max(y1, 0.0)
    x2, y2 = min(x2, float(frame_w)), min(y2, float(frame_h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, None

    def rect(a, b, c, d):
        return [(round(a, 1), round(b, 1)), (round(c, 1), round(b, 1)),
                (round(c, 1), round(d, 1)), (round(a, 1), round(d, 1))]

    height = y2 - y1
    # Desk line at the lower third of the observed person extent: below it a
    # seated person's hands are behind the desk edge from this camera.
    desk_y = y1 + height * 0.66
    return {
        SEAT_CONTEXT: rect(x1, y1, x2, y2),
        UPPER_BODY: rect(x1, y1, x2, desk_y),
        DESK_WORK_SURFACE: rect(x1, desk_y, x2, y2),
        LAP_BAG: rect(x1, y1 + height * 0.82, x2, y2),
    }, round(desk_y, 1)


def _bounds(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _rebuild(seat, x1, y1, x2, y2):
    """Rewrite a seat's four zones from a new outer rectangle."""
    # Cast to plain floats. The clip midpoints come from detector box
    # coordinates, which numpy hands back as float32, and `json.dumps` refuses
    # those -- the profile then fails to write with the separation silently
    # never applied.
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    height = y2 - y1
    desk_y = y1 + height * 0.66

    def rect(a, b, c, d):
        return [(round(float(a), 1), round(float(b), 1)),
                (round(float(c), 1), round(float(b), 1)),
                (round(float(c), 1), round(float(d), 1)),
                (round(float(a), 1), round(float(d), 1))]

    seat.zones[SEAT_CONTEXT] = rect(x1, y1, x2, y2)
    seat.zones[UPPER_BODY] = rect(x1, y1, x2, desk_y)
    seat.zones[DESK_WORK_SURFACE] = rect(x1, desk_y, x2, y2)
    seat.zones[LAP_BAG] = rect(x1, y1 + height * 0.82, x2, y2)
    seat.desk_line_y = round(desk_y, 1)


def _separate(seats, groups, gap: float = 2.0, min_side: float = 24.0):
    """Clip each seat rectangle to its own axis-aligned Voronoi cell.

    Every seat is clipped against every *other* seat's midpoint half-plane, and
    each seat is computed from the original centres rather than from already
    clipped neighbours. That matters: an earlier pairwise version fixed (A, B)
    and then (A, C), and the second fix re-expanded A back over B. Pairwise
    repair does not converge; independent clipping partitions in one pass.

    A seat clipped below `min_side` on either axis is dropped -- its footprint
    was inside a neighbour's cell, which means the cluster was not a separate
    seat in the first place.
    """
    centres = [(group["fx"], group["fy"]) for group in groups]
    originals = [_bounds(seat.zones[SEAT_CONTEXT]) for seat in seats]

    kept, clipped_pairs = [], 0
    for i, seat in enumerate(seats):
        x1, y1, x2, y2 = originals[i]
        acx, acy = centres[i]
        for j, (bcx, bcy) in enumerate(centres):
            if i == j:
                continue
            bx1, by1, bx2, by2 = originals[j]
            # Only clip where the original rectangles actually overlapped.
            if x2 <= bx1 or bx2 <= x1 or y2 <= by1 or by2 <= y1:
                continue
            clipped_pairs += 1
            if abs(acx - bcx) >= abs(acy - bcy):
                mid = (acx + bcx) / 2.0
                if acx <= bcx:
                    x2 = min(x2, mid - gap)
                else:
                    x1 = max(x1, mid + gap)
            else:
                mid = (acy + bcy) / 2.0
                if acy <= bcy:
                    y2 = min(y2, mid - gap)
                else:
                    y1 = max(y1, mid + gap)

        if x2 - x1 < min_side or y2 - y1 < min_side:
            continue
        _rebuild(seat, x1, y1, x2, y2)
        kept.append(seat)

    # A detection near a split line genuinely could belong to either seat, so
    # every pair that had to be split is recorded as an ambiguous boundary.
    for i, a in enumerate(kept):
        ax1, ay1, ax2, ay2 = _bounds(a.zones[SEAT_CONTEXT])
        for b in kept[i + 1:]:
            bx1, by1, bx2, by2 = _bounds(b.zones[SEAT_CONTEXT])
            near = (min(ax2, bx2) - max(ax1, bx1) > -4 * gap
                    and min(ay2, by2) - max(ay1, by1) > -4 * gap)
            if near:
                if b.seat_id not in a.ambiguous_with:
                    a.ambiguous_with.append(b.seat_id)
                if a.seat_id not in b.ambiguous_with:
                    b.ambiguous_with.append(a.seat_id)
    return kept, clipped_pairs // 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera-id", default=None)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--confidence", type=float, default=0.45)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--merge-px", type=float, default=70.0)
    ap.add_argument("--max-seats", type=int, default=12)
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 2
    camera_id = args.camera_id or args.video.name[:2]

    print(f"sampling {args.frames} frames across {args.video.name}")
    frames = sample_frames(args.video, args.frames)
    if not frames:
        print("no frames decoded", file=sys.stderr)
        return 2
    frame_h, frame_w = frames[0][1].shape[:2]
    print(f"  {len(frames)} frames at {frame_w}x{frame_h}")

    boxes = detect_people(frames, args.detector, args.confidence)
    print(f"  {len(boxes)} person detections at confidence >= {args.confidence}")
    if not boxes:
        print("no people detected; a calibration cannot be proposed from this "
              "footage and seat-attributed evidence stays blocked (PRD 3)",
              file=sys.stderr)
        return 2

    kept, allc = cluster(boxes, args.min_support, args.merge_px)
    print(f"  {len(allc)} footprint clusters, {len(kept)} with >= "
          f"{args.min_support} frames of support")
    kept = kept[:args.max_seats]

    # Seat and its source cluster are carried together. Keeping two parallel
    # lists and indexing both silently misaligned them whenever a cluster
    # produced no usable zone, which clipped every seat against the wrong
    # neighbour's centre and left the zones overlapping.
    seats, seat_groups = [], []
    for index, group in enumerate(kept, start=1):
        zones, desk_y = zones_from(group, frame_w, frame_h)
        if zones is None:
            continue
        seats.append(Seat(seat_id=f"{camera_id}-P{index:02d}", zones=zones,
                          desk_line_y=desk_y,
                          notes=f"proposed from {len(group['frames'])} frames "
                                f"of person detections; unreviewed"))
        seat_groups.append(group)
    if not seats:
        print("no cluster produced a usable zone", file=sys.stderr)
        return 2

    # Proposed zones overlap wherever people sit close together, and overlapping
    # zones cannot be used: motion in a shared region would be credited to both
    # seats, which was measured in an earlier phase as phantom events on the
    # neighbour whenever the real candidate moved.
    #
    # Contested pixels are therefore split at the midpoint between the two
    # footprint centres -- the Voronoi boundary -- rather than given to one
    # seat. The pair is still recorded as an ambiguous boundary, because a
    # detection landing near that line genuinely could belong to either.
    before = len(seats)
    seats, separated = _separate(seats, seat_groups)
    if not seats:
        print("every proposed zone collapsed under separation", file=sys.stderr)
        return 2
    for i, a in enumerate(seats):
        for b in seats[i + 1:]:
            a.adjacent.append(b.seat_id)
            b.adjacent.append(a.seat_id)
    if separated:
        print(f"  clipped {separated} overlapping zone pair(s) to their Voronoi "
              f"midpoints; {before - len(seats)} zone(s) dropped as too small")

    # Static background support: the frame minus every proposed seat. Alignment
    # needs somewhere that is not a person; this is the honest remainder.
    background = MaskRegion(
        STATIC_BACKGROUND,
        [(0.0, 0.0), (float(frame_w), 0.0),
         (float(frame_w), float(frame_h)), (0.0, float(frame_h))],
        note="whole frame minus seats; seats are subtracted at use time by "
             "calibration.static_support()")

    profile = CalibrationProfile(
        camera_id=camera_id, epoch_id="e1", version=1,
        frame_w=frame_w, frame_h=frame_h,
        source_sha256=sha256_file(args.video),
        reference_pts_ms=frames[0][0],
        operator="tools/propose_calibration.py (automatic proposal)",
        reviewer="",
        approval_state=DRAFT,
        seats=seats, masks=[background],
        proposal_source="person_detection via D-FINE",
        confidence_notes=(
            f"DRAFT. Proposed automatically from {len(boxes)} person detections "
            f"over {len(frames)} frames sampled across the whole recording. "
            f"PRD 7 allows automatic proposal but never automatic approval, so "
            f"this profile is unapproved and seat attribution is blocked "
            f"downstream. A reviewer must confirm the zones, mark monitor and "
            f"reflection exclusions, and resolve the ambiguous boundaries "
            f"before any seat-attributed evidence may be produced."))

    issues = profile.validate()
    blocking = [i for i in issues if i.severity == "blocking"]
    print(f"\nproposed {len(seats)} seat zone(s) for camera {camera_id}")
    for issue in issues:
        print(f"  {issue.severity:8} {issue.code}: {issue.detail[:70]}")
    if blocking:
        print(f"\n{len(blocking)} blocking issue(s); writing anyway so the "
              f"reviewer can see them, but this profile cannot be used.",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    profile.write(args.out)
    print(f"\nwritten {args.out}  ({profile.version_id}, {profile.approval_state})")
    print("Seat attribution stays BLOCKED until a reviewer approves it (PRD 3).")
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
