"""Phase 3: whole-video motion, baseline, ROI, segmentation and gates.

    python tools/run_motion_scan.py --run artifacts/runs/<run_id> \
        --calibration calib/cam12_e1_v1.json "C:/DrishtiAI/03.CCTV Mobile Usage.mkv"

    python tools/run_motion_scan.py --preview --calibration ... VIDEO

Writes `05_motion_roi/` and `06_segmentation/` into the run folder: per-window
metrics, the ROI manifest, the candidate manifest, every state transition with
its reason, the gate log, and the events themselves.

This stage needs no neural model and no reference manifest. Per PRD 3 and the
owner's direction, a missing reference manifest or unapproved taxonomy blocks
detector selection and evaluation — not this.

`candidate_manifest.jsonl` is the frozen hand-off PRD 4 requires: video hash,
PTS, calibration version, event ID, health state and source crop geometry. Every
later stage consumes it and nothing later may re-derive candidates, so that an
input change can never be credited to a model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, baseline as bl, gates, motion, roi, segmentation as seg  # noqa: E402
from pipeline.calibration import CalibrationProfile, SEAT_CONTEXT  # noqa: E402
from pipeline.health import HealthTimeline  # noqa: E402
from pipeline.run_manifest import (BLOCKED, COMPLETE, RunManifest,  # noqa: E402
                                   RunStatus, SKIPPED_UNAVAILABLE)


def bbox(polygon) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--source-sha256", default=None)
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 2

    profile = CalibrationProfile.read(args.calibration)
    blocking = profile.blocking_issues
    if blocking:
        print(f"calibration {profile.version_id} has blocking issues:",
              file=sys.stderr)
        for issue in blocking:
            print(f"  {issue}", file=sys.stderr)
        return 2

    paths = status = None
    if not args.preview:
        if args.run is None:
            print("--run is required unless --preview is given", file=sys.stderr)
            return 2
        paths = artifacts.open_run(args.run.parent, args.run.name)
        if paths.sealed:
            print(f"{args.run} is sealed; reprocessing needs a new run.",
                  file=sys.stderr)
            return 2
        status = RunStatus(paths, RunManifest.read(paths).run_id)
        status.begin("05_motion_roi")

    print(f"calibration  {profile.version_id}  ({profile.approval_state})")
    if not profile.approved:
        print("  NOTE: this profile is not approved, so seat attribution is "
              "blocked downstream (PRD 3). Motion and ROI still run.")

    print(f"\nscanning {args.video.name}")
    scan = motion.scan(args.video, profile, motion.MotionConfig(),
                       progress_every_s=60.0)
    print(f"  motion vectors {'present' if scan.motion_vectors_available else 'ABSENT'}"
          f" ({scan.mv_per_frame:.0f}/frame)")
    print(f"  {len(scan.windows)} zone-windows over "
          f"{scan.duration_ms / 1000.0:.1f} s in {scan.seconds:.1f} s "
          f"({scan.realtime_factor:.0f}x realtime)")

    bcfg = bl.BaselineConfig()
    baselines = bl.learn(scan.windows, bcfg)
    print("\nbaselines")
    for (seat_id, zone), base in sorted(baselines.items()):
        flag = "  DEGENERATE" if base.degenerate else ""
        print(f"  {seat_id:10} {zone:20} median {base.median:.5f}  "
              f"spread {base.spread:.5f}  n={base.sample_windows}"
              f"{'  typing regime' if base.typing_median else ''}{flag}")
        if base.degenerate:
            print(f"      {base.degenerate_reason}")

    scored = bl.score(scan.windows, baselines, bcfg)
    motion.suppression_gates(scored, scan.gate_log)
    scan.observations += motion.classify_noise(
        scan.windows, bcfg, scan.duration_ms, scan.gate_log)

    # Health conditions and epoch changes become walls no event may bridge.
    timeline = HealthTimeline(scan.observations)
    walls = [(o.start_pts_ms, o.end_pts_ms, o.condition)
             for o in scan.observations if o.blocks_bridging]
    events, transitions = seg.segment(
        scored, seg.SegmentConfig(), boundaries=seg.BoundaryPolicy(walls))

    report = seg.summarise(events, scan.duration_ms)
    print(f"\nsegmentation")
    print(f"  {report['normal_priority']} normal-priority events, "
          f"{report['low_priority_retained']} retained at low priority")
    print(f"  {report['candidate_fraction'] * 100:.2f}% of the recording flagged")
    print(f"  close reasons: "
          f"{ {k: v for k, v in report['by_close_reason'].items() if v} }")

    print(f"\ngates: { {k: v for k, v in scan.gate_log.counts().items() if v} }")
    print(f"  nothing discarded; {scan.gate_log.summary()['retained_records']} "
          f"records retained and searchable")
    conditions: dict[str, int] = {}
    for observation in scan.observations:
        conditions[observation.condition] = conditions.get(observation.condition, 0) + 1
    print(f"health observations: {conditions or 'none'}")

    if paths is None:
        return 0

    # ------------------------------------------------------------ artifacts --
    seat_boxes = {s.seat_id: bbox(s.zones[SEAT_CONTEXT])
                  for s in profile.seats if SEAT_CONTEXT in s.zones}

    for item in scored:
        window = item.window
        paths.append_jsonl("05_motion_roi/window_metrics.jsonl", {
            "seat_id": window.seat_id, "zone": window.zone,
            "start_pts_ms": round(window.start_ms, 3),
            "end_pts_ms": round(window.end_ms, 3),
            "residual": round(window.residual, 6),
            "area_ratio": round(window.area_ratio, 5),
            "moving_blocks": window.moving_blocks,
            "coherence": round(window.coherence, 4),
            "below_desk": round(window.below_desk, 4),
            "deviation": round(item.deviation, 4),
            "regime": item.regime,
            "suppressed_by": item.suppressed_by,
        })

    for (seat_id, zone), base in sorted(baselines.items()):
        paths.append_jsonl("05_motion_roi/baselines.jsonl", base.to_row())

    for record in scan.gate_log.to_rows():
        paths.append_jsonl("05_motion_roi/gate_log.jsonl", record)

    for observation in scan.observations:
        paths.append_jsonl("02_health/observations.jsonl",
                           {"source": str(args.video), **observation.to_row()})

    for transition in transitions:
        paths.append_jsonl("06_segmentation/event_state_transitions.jsonl",
                           transition.to_row())
    for event in events:
        paths.append_jsonl("06_segmentation/events.jsonl", event.to_row())

    # The frozen hand-off (PRD 4). Later stages read only this.
    source_sha = args.source_sha256 or ""
    if not source_sha:
        manifest = RunManifest.read(paths)
        match = next((s for s in manifest.sources
                      if Path(s.path).name == args.video.name), None)
        source_sha = match.sha256 if match else ""

    for event in events:
        box = seat_boxes.get(event.seat_id)
        paths.append_jsonl("05_motion_roi/candidate_manifest.jsonl", {
            "event_id": event.event_id,
            "source_video": str(args.video),
            "source_video_sha256": source_sha,
            "calibration_version": profile.version_id,
            "camera_epoch": profile.epoch_id,
            "seat_id": event.seat_id,
            "zone": event.zone,
            "start_pts_ms": round(event.start_pts_ms, 3),
            "end_pts_ms": round(event.end_pts_ms, 3),
            "peak_pts_ms": round(event.peak_pts_ms, 3),
            "priority": event.priority,
            "health_state": [o.condition for o in
                             timeline.over(event.start_pts_ms, event.end_pts_ms)],
            "gate_state": scan.gate_log.worst(event.start_pts_ms,
                                              event.end_pts_ms, event.seat_id),
            "source_crop": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
            if box else None,
        })

    # Stages this pass actually performed, recorded rather than left `pending`.
    # PRD 18 requires the run to evidence its own coverage, and a stage that ran
    # but reports `pending` is indistinguishable from one that never ran.
    manifest_record = RunManifest.read(paths)
    paths.write_json("04_calibration/calibration_snapshot.json", {
        "version_id": profile.version_id,
        "approval_state": profile.approval_state,
        "operator": profile.operator,
        "reviewer": profile.reviewer,
        "seats": [s.seat_id for s in profile.seats],
        "frame_w": profile.frame_w, "frame_h": profile.frame_h,
    })
    status.finish("source", COMPLETE,
                  sources=len(manifest_record.sources))
    status.finish("01_ingest", COMPLETE,
                  covered_ms=scan.duration_ms, total_ms=scan.duration_ms,
                  reason="one complete forward decode pass; motion vectors "
                         f"{'present' if scan.motion_vectors_available else 'absent'}")
    status.finish("02_health", COMPLETE,
                  covered_ms=scan.duration_ms, total_ms=scan.duration_ms,
                  observations=len(scan.observations))
    status.finish("04_calibration", COMPLETE,
                  reason=f"{profile.version_id} ({profile.approval_state})",
                  seats=len(profile.seats))
    # Honest about what did not run: no camera-motion condition was observed on
    # this pass, so no global transform was estimated. Identity is the named
    # baseline (PRD 6.3.2), not a silent default.
    moved = [o for o in scan.observations
             if o.condition in ("camera_motion", "camera_repositioned")]
    status.finish("03_alignment",
                  COMPLETE if moved else SKIPPED_UNAVAILABLE,
                  reason=None if moved else
                  "no camera_motion or camera_repositioned condition was "
                  "observed, so no global transform was estimated. Identity is "
                  "the named baseline, not a silent default (PRD 6.3.2)",
                  observations=len(moved))

    status.finish("05_motion_roi", COMPLETE,
                  covered_ms=scan.duration_ms, total_ms=scan.duration_ms,
                  windows=len(scan.windows),
                  gate_records=len(scan.gate_log.records))
    status.finish("06_segmentation", COMPLETE,
                  covered_ms=scan.duration_ms, total_ms=scan.duration_ms,
                  events=len(events),
                  normal_priority=report["normal_priority"],
                  low_priority=report["low_priority_retained"])
    # PRD 3: without an approved taxonomy and a reference manifest, detector
    # selection and evaluation are blocked. Recorded here so the run says so
    # rather than a later stage failing obscurely.
    status.finish("09_object", BLOCKED,
                  reason="no approved taxonomy; PRD 3 blocks detector selection")
    status.finish("12_evaluation", BLOCKED,
                  reason="no reference manifest; object evaluation is incomplete "
                         "and no quality claim may be made")

    print(f"\nwritten to {paths.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
