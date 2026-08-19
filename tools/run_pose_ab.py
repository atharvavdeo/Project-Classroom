"""Phase 4: candidate-only tracking and the matched pose A/B (PRD 9).

    python tools/run_pose_ab.py --run artifacts/runs/<run_id> \
        --calibration calib/cam12_e1_v1.json "C:/DrishtiAI/03.CCTV Mobile Usage.mkv"

Reads the frozen `05_motion_roi/candidate_manifest.jsonl` and nothing else. It
does not re-derive candidates, cannot widen them, and never reads the reference
manifest -- PRD 15 makes that readable only at stage 12, and
`tests/test_reference_isolation.py` fails the run if a row here carries one.

Order, which is mandatory rather than stylistic (PRD 4):

  1. plan the person-detection sample rows and write them before decoding
  2. detect people on exactly those rows
  3. track under each named tracker configuration, on identical boxes
  4. associate seats, retaining every refusal and every candidate score
  5. freeze `08_pose/pose_manifest.jsonl` and hash it
  6. run every pose configuration against that identical manifest
  7. write a comparison that names no winner

Writes `07_tracking/<config>/` and `08_pose/<config>/` per configuration, so no
configuration can overwrite another's artifacts (PRD 17.9).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, person_detection as pd, pose, seat_association as sa, tracking  # noqa: E402
from pipeline.calibration import CalibrationProfile  # noqa: E402
from pipeline.run_manifest import (BLOCKED, COMPLETE, LAUNCH_FAILED,  # noqa: E402
                                   LICENCE_BLOCKED, MISSING_WEIGHTS,
                                   RESOURCE_UNAVAILABLE, RunManifest, RunStatus,
                                   SKIPPED_UNAVAILABLE)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


class SequentialFrameReader:
    """One forward decode pass, serving rows in ascending PTS order.

    Not a seek-per-row reader. Phase 1 measured PyAV's seek on this corpus
    returning to the head of the file on two of six files; a reader built on
    seek would silently serve frame 0 for every row. It is also 0.5-1.0 s per
    seek, which over a few hundred rows dominates the run.

    Rows arrive already sorted by PTS, so a single pass with a held frame serves
    all of them. A row before the first decoded frame or after the last gets
    None, and None means a hole in the evidence -- preserved, per PRD 8, rather
    than filled with the nearest frame that happens to be in hand.
    """

    def __init__(self, video: Path):
        self.video = video
        self._container = None
        self._stream = None
        self._frames = None
        self._current = None            # (pts_ms, ndarray)
        self._last_pts = -1.0
        self.rewinds = 0
        self.misses = 0

    def _open(self) -> bool:
        if self._frames is not None:
            return True
        try:
            import av
        except Exception:                                # noqa: BLE001
            return False
        if not self.video.is_file():
            return False
        self._container = av.open(str(self.video))
        self._stream = self._container.streams.video[0]
        self._frames = self._container.decode(self._stream)
        return True

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = self._stream = self._frames = None

    def __call__(self, row):
        if not self._open():
            return None
        target = row.pts_ms
        if target < self._last_pts:
            # Out-of-order request. Recorded rather than silently served the
            # wrong frame; the caller's ordering guarantee has broken.
            self.rewinds += 1
            self.misses += 1
            return None

        time_base = float(self._stream.time_base)
        while True:
            if self._current is not None and self._current[0] >= target - 1.0:
                self._last_pts = target
                return self._current[1]
            try:
                frame = next(self._frames)
            except (StopIteration, EOFError):
                self.misses += 1
                return None
            if frame.pts is None:
                continue
            self._current = (float(frame.pts * time_base) * 1000.0, None)
            if self._current[0] >= target - 1.0:
                self._current = (self._current[0],
                                 frame.to_ndarray(format="bgr24"))


def frame_reader_for(video: Path, enabled: bool):
    """A reader, or None when no model will consume it."""
    return SequentialFrameReader(video) if enabled else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--sample-hz", type=float, default=2.0)
    ap.add_argument("--person-model", type=Path, default=None)
    ap.add_argument("--pose", action="store_true",
                    help="run the pose configurations. rtmlib fetches its "
                         "own weights on first use, so this needs network "
                         "access once.")
    ap.add_argument("--pose-device", default="cpu")
    ap.add_argument("--limit-events", type=int, default=0,
                    help="process only the first N candidate events")
    ap.add_argument("--max-events", type=int, default=0,
                    help="cap the number of candidate events, sampled evenly "
                         "across the recording rather than truncated to its "
                         "first minutes. The cap is recorded in the run.")
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    if paths.sealed:
        print(f"{args.run} is sealed; reprocessing needs a new run.",
              file=sys.stderr)
        return 2
    status = RunStatus(paths, RunManifest.read(paths).run_id)

    profile = CalibrationProfile.read(args.calibration)
    candidates = read_jsonl(paths.dir / "05_motion_roi/candidate_manifest.jsonl")
    if not candidates:
        print("no candidate manifest; run tools/run_motion_scan.py first.",
              file=sys.stderr)
        status.finish("07_tracking", BLOCKED,
                      reason="no candidate manifest to consume")
        status.write()
        return 2
    total_candidates = len(candidates)
    cap_note = None
    if args.limit_events:
        candidates = candidates[:args.limit_events]
        cap_note = (f"first {len(candidates)} of {total_candidates} events; "
                    f"--limit-events truncates rather than samples")
    elif args.max_events and total_candidates > args.max_events:
        # Evenly spaced, not the first N. A truncated head would measure only
        # the opening minutes of a long recording and report it as the whole.
        step = total_candidates / args.max_events
        picked = [candidates[int(i * step)] for i in range(args.max_events)]
        candidates = picked
        cap_note = (f"{len(candidates)} of {total_candidates} events, sampled "
                    f"evenly across the full recording (every ~{step:.1f}th). "
                    f"Motion covered the complete duration; only the model "
                    f"stages are capped, and this cap is why.")
    if cap_note:
        print(f"EVENT CAP: {cap_note}")

    # --------------------------------------------------- 1. the sample plan --
    status.begin("07_tracking")
    cfg = pd.SampleConfig(sample_hz=args.sample_hz)
    rows, notes = pd.plan(candidates, cfg)
    for row in rows:
        paths.append_jsonl("07_tracking/sample_plan.jsonl", row.to_row())
    for note in notes:
        paths.append_jsonl("07_tracking/sample_plan_notes.jsonl", note)
    print(f"{len(candidates)} candidate events -> {len(rows)} sample rows "
          f"at {args.sample_hz} Hz")

    # ------------------------------------------------ 2. person detection --
    reader = frame_reader_for(args.video, bool(args.person_model))
    if args.person_model and args.person_model.is_file():
        runner = pd.onnx_runner(args.person_model, frame_reader=reader)
        detection = pd.detect(rows, runner, "person_detection_primary", cfg)
    else:
        detection = pd.unavailable(
            "person_detection_primary", RESOURCE_UNAVAILABLE,
            "no --person-model was supplied, so no person boxes exist. Every "
            "later stage is blocked rather than run on fabricated boxes.")
    paths.write_json("07_tracking/person_detection.json", detection.metrics())
    print(f"person detection: {detection.state} "
          f"({detection.returned} boxes over {detection.rows_seen} rows)")

    for box in detection.boxes:
        paths.append_jsonl("07_tracking/person_boxes.jsonl", box.to_row())

    # ------------------------------------------------------- 3. tracking --
    rows_by_id = {r.row_id: r for r in rows}
    tracker_results = {}
    association_by_config = {}
    for config_id in tracking.CONFIGS:
        result = tracking.track(list(detection.boxes), config_id)
        tracker_results[config_id] = result
        directory = f"07_tracking/{config_id}"
        (paths.dir / directory).mkdir(parents=True, exist_ok=True)
        for candidate in result.tracks:
            paths.append_jsonl(f"{directory}/tracks.jsonl", candidate.to_row())
        for box in result.boxes:
            paths.append_jsonl(f"{directory}/observations.jsonl", box.to_row())

        # ------------------------------------------- 4. seat association --
        associations = sa.associate(result, profile)
        association_by_config[config_id] = associations
        for association in associations:
            paths.append_jsonl(f"{directory}/seat_association.jsonl",
                               association.to_row())
        seat_of = {f"{a.track_id}|{a.pts_ms:.3f}": a.seat_id
                   for a in associations if a.seat_id}
        paths.write_json(f"{directory}/metrics.json", {
            **result.metrics(),
            "continuity": tracking.continuity(result, seat_of),
            "association": sa.summarise(associations),
        })
        print(f"  {config_id:22} {result.state:12} "
              f"{len(result.tracks):4} tracks  "
              f"{sa.summarise(associations).get('resolved', 0):5} seats resolved")

    if cap_note:
        paths.write_json("07_tracking/event_cap.json", {
            "total_candidate_events": total_candidates,
            "events_processed": len(candidates),
            "note": cap_note})
    status.finish("07_tracking", COMPLETE if detection.available else BLOCKED,
                  reason=None if detection.available else detection.reason,
                  sample_rows=len(rows), person_boxes=len(detection.boxes))

    # --------------------------------------------- 5. freeze the manifest --
    status.begin("08_pose")
    primary = association_by_config[tracking.PRIMARY]
    manifest = pose.freeze(primary, rows_by_id)
    digest = pose.manifest_digest(manifest)
    for row in manifest:
        paths.append_jsonl("08_pose/pose_manifest.jsonl", row.to_row())
    paths.write_json("08_pose/pose_manifest_digest.json", {
        "digest": digest, "rows": len(manifest),
        "note": "every pose configuration is given these rows unchanged. A "
                "configuration reporting a different digest was not compared "
                "(PRD 4).",
    })
    print(f"\npose manifest frozen: {len(manifest)} rows, digest {digest[:16]}")

    # ------------------------------------------------------ 6. the pose A/B --
    results = []
    for config_id in pose.POSE_CONFIGS:
        if config_id == pose.ALPHAPOSE:
            ckpt = ROOT / "models/alphapose/fast_421_res152_256x192.pth"
            mcfg = ROOT / "models/alphapose/256x192_res152_lr1e-3_1x-duc.yaml"
            if not args.pose:
                results.append(pose.unavailable(
                    config_id, RESOURCE_UNAVAILABLE,
                    "--pose was not requested", manifest))
                continue
            if not (ckpt.is_file() and mcfg.is_file()):
                results.append(pose.unavailable(
                    config_id, MISSING_WEIGHTS,
                    f"checkpoint or model config absent under models/alphapose/",
                    manifest))
                continue
            reader = frame_reader_for(args.video, True)
            try:
                runner = pose.alphapose_runner(config_id, str(mcfg), str(ckpt),
                                               frame_reader=reader,
                                               device=args.pose_device)
            except Exception as exc:                    # noqa: BLE001
                results.append(pose.unavailable(
                    config_id, LAUNCH_FAILED,
                    f"{type(exc).__name__}: {exc}", manifest))
                continue
            results.append(pose.run(manifest, runner, config_id, profile))
            continue
        if not args.pose:
            results.append(pose.unavailable(
                config_id, RESOURCE_UNAVAILABLE,
                "--pose was not requested, so this configuration did not run. "
                "No substitute was used and no metric is reported for it.",
                manifest))
            continue
        # A fresh reader per configuration. The reader is a single forward
        # decode pass, so a shared one is exhausted by whichever configuration
        # runs first and every later configuration sees an empty video. That
        # reads in the report as a model that found nothing, which is the exact
        # misattribution PRD 4 exists to prevent -- and it did happen here
        # before this line: RTMO scored 1 of 74 rows purely from running second.
        pose_reader = frame_reader_for(args.video, args.pose)
        try:
            runner = pose.rtmlib_runner(config_id, frame_reader=pose_reader,
                                        device=args.pose_device)
        except Exception as exc:                        # noqa: BLE001
            results.append(pose.unavailable(
                config_id, LAUNCH_FAILED,
                f"{type(exc).__name__}: {exc}", manifest))
            continue
        results.append(pose.run(manifest, runner, config_id, profile))

    for result in results:
        directory = f"08_pose/{result.config_id}"
        (paths.dir / directory).mkdir(parents=True, exist_ok=True)
        paths.write_json(f"{directory}/metrics.json", result.metrics())
        for observation in result.observations:
            paths.append_jsonl(f"{directory}/observations.jsonl",
                               observation.to_row())
        print(f"  {result.config_id:26} {result.state:20} "
              f"{len(result.observations):5} observations")

    # ------------------------------------------------- 7. the comparison --
    report = pose.compare(results, manifest)
    paths.write_json("08_pose/comparison.json", report)
    print(f"\nmanifest equivalence: {report['manifest_equivalence']}")
    print(f"available: {report['available'] or 'none'}")
    for entry in report["unavailable"]:
        print(f"  unavailable  {entry['config_id']:26} {entry['state']}")
    print(f"selection: {report['selection']}")

    any_pose = any(r.available for r in results)
    status.finish("08_pose",
                  COMPLETE if any_pose else SKIPPED_UNAVAILABLE,
                  reason=None if any_pose else
                  "no pose configuration was runnable; the manifest is frozen "
                  "and every configuration is recorded as unavailable",
                  manifest_rows=len(manifest), manifest_digest=digest)
    status.write()
    print(f"\nwritten to {paths.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
