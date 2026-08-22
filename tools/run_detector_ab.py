"""Phase 5: the crop manifest, the detector/SAHI A/B, and temporal confirmation.

    python tools/run_detector_ab.py --run artifacts/runs/<run_id> \
        --calibration calib/cam12_e1_v1.json \
        --dfine models/dfine/onnx/model.onnx "C:/DrishtiAI/03.CCTV Mobile Usage.mkv"

Consumes `08_pose/pose_manifest.jsonl` and the pose observations. Freezes
`09_object/object_crop_manifest.jsonl`, hashes it, and hands that identical list
to every detector configuration (PRD 4.4).

Writes `09_object/<config>/` per configuration -- raw detections, merged
detections, and for the SAHI arms every slice that was run. No configuration
writes into another's directory, so PRD 17.9 holds structurally rather than by
convention.

Selection is not made here and cannot be: PRD 3 blocks it until the taxonomy is
approved, and PRD 15 places it at stage 12 against the reference manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import (artifacts, crops as cr, object_detection as od,  # noqa: E402
                      sahi_adapter as sahi, temporal_confirmation as tc)
from pipeline.calibration import CalibrationProfile  # noqa: E402
from pipeline.pose import PoseRow  # noqa: E402
from pipeline.run_manifest import (BLOCKED, COMPLETE, LAUNCH_FAILED,  # noqa: E402
                                   RESOURCE_UNAVAILABLE, RunManifest, RunStatus)
from tools.run_pose_ab import SequentialFrameReader, read_jsonl  # noqa: E402


class Observation:
    """A pose observation rehydrated from its JSONL row.

    Only the fields the crop policy consults. Reading the persisted row rather
    than recomputing is deliberate: the crop must be built from what pose
    actually recorded, so that a crop and its overlay can never disagree.
    """

    def __init__(self, row: dict):
        self.config_id = row.get("config_id", "")
        self.torso_scale_px = row.get("torso_scale_px")
        self.joints = [_Joint(j) for j in row.get("joints", [])]
        self._by_name = {j.name: j for j in self.joints}

    def state(self, index: int) -> str:
        from pipeline.pose import JOINT_NAMES
        joint = self._by_name.get(JOINT_NAMES.get(index, ""))
        return joint.observability if joint else "not_detected"


class _Joint:
    def __init__(self, row: dict):
        self.name = row["name"]
        self.x = row["x"]
        self.y = row["y"]
        self.confidence = row["confidence"]
        self.observability = row["observability"]


def pose_rows_from(paths) -> list[PoseRow]:
    out = []
    for row in read_jsonl(paths.dir / "08_pose/pose_manifest.jsonl"):
        row["box"] = tuple(row["box"])
        out.append(PoseRow(**row))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--dfine", type=Path, default=None,
                    help="path to the D-FINE ONNX checkpoint")
    ap.add_argument("--rfdetr", type=Path, default=None)
    ap.add_argument("--rfdetr-variant", default="nano",
                    help="nano|small|medium|base|large. Weights are fetched by "
                         "rfdetr itself on first use.")
    ap.add_argument("--no-rfdetr", action="store_true",
                    help="skip RF-DETR; it is recorded unavailable, not omitted")
    ap.add_argument("--pose-source", default="rtmpose_baseline",
                    help="which pose configuration positions the hand crops")
    ap.add_argument("--confidence", type=float, default=0.25)
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    if paths.sealed:
        print(f"{args.run} is sealed; reprocessing needs a new run.",
              file=sys.stderr)
        return 2
    status = RunStatus(paths, RunManifest.read(paths).run_id)
    profile = CalibrationProfile.read(args.calibration)

    pose_rows = pose_rows_from(paths)
    if not pose_rows:
        print("no pose manifest; run tools/run_pose_ab.py first.", file=sys.stderr)
        status.finish("09_object", BLOCKED, reason="no pose manifest to consume")
        status.write()
        return 2

    observations = {}
    observation_path = paths.dir / f"08_pose/{args.pose_source}/observations.jsonl"
    for row in read_jsonl(observation_path):
        observations[row["pose_row_id"]] = Observation(row)
    print(f"{len(pose_rows)} pose rows, {len(observations)} observations from "
          f"{args.pose_source}")

    # ------------------------------------------------- freeze the crops --
    status.begin("09_object")
    crop_cfg = cr.CropConfig()
    crops = cr.freeze(pose_rows, observations, profile, crop_cfg)
    crop_digest = cr.digest(crops)
    for crop in crops:
        paths.append_jsonl("09_object/object_crop_manifest.jsonl", crop.to_row())
    crop_summary = cr.summarise(crops, crop_cfg)
    paths.write_json("09_object/crop_manifest_summary.json", crop_summary)

    print(f"\ncrop manifest frozen: {len(crops)} crops, digest {crop_digest[:16]}")
    for name, count in crop_summary["by_crop_config"].items():
        print(f"  {name:34} {count:5}")
    if crop_summary["fallback_reasons"]:
        print(f"  fallback reasons: {crop_summary['fallback_reasons']}")
    print(f"  {crop_summary['insufficient_native_resolution']} crops under the "
          f"native-resolution floor "
          f"({crop_summary['insufficient_fraction'] * 100:.1f}%)")
    print(f"  median magnification {crop_summary['median_magnification']}x, "
          f"max {crop_summary['max_magnification']}x")

    # ------------------------------------------------------ the detector A/B --
    detect_cfg = od.DetectConfig(confidence=args.confidence)
    sahi_cfg = sahi.SahiConfig()
    results, slice_logs = [], {}

    for config_id in od.OBJECT_CONFIGS:
        is_sahi = config_id.endswith("_sahi")
        weights = args.rfdetr if config_id.startswith("rf_detr") else args.dfine

        if config_id.startswith("rf_detr") and args.no_rfdetr:
            results.append(od.unavailable(
                config_id, RESOURCE_UNAVAILABLE,
                "skipped by --no-rfdetr; it stays in the comparison as "
                "unavailable and is never substituted by D-FINE (PRD 5).",
                crop_digest))
            continue

        if config_id.startswith("rf_detr"):
            reader = SequentialFrameReader(args.video)
            try:
                base = od.rfdetr_runner(args.rfdetr_variant, crop_cfg,
                                        confidence=args.confidence,
                                        frame_reader=lambda c, _r=reader: _r(c))
            except Exception as exc:                    # noqa: BLE001
                results.append(od.unavailable(
                    config_id, LAUNCH_FAILED,
                    f"{type(exc).__name__}: {exc}", crop_digest))
                continue
            if is_sahi:
                log = sahi.SliceLog()
                slice_logs[config_id] = log
                runner = sahi.sahi_runner(base, sahi_cfg, on_slice=log.record)
            else:
                runner = base
            result = od.detect(crops, runner, config_id, detect_cfg, crop_digest)
            if is_sahi:
                result.slices_run = len(slice_logs[config_id].rows)
            results.append(result)
            continue
        if not (weights and weights.is_file()):
            results.append(od.unavailable(
                config_id, RESOURCE_UNAVAILABLE,
                f"no checkpoint supplied for {config_id}", crop_digest))
            continue

        # A fresh reader per configuration. A shared forward-only reader is
        # exhausted by whichever configuration runs first, and every later one
        # then measures an empty video -- which reads as a model difference.
        reader = SequentialFrameReader(args.video)

        def frame_for(crop, _reader=reader):
            return _reader(crop)

        try:
            base = od.dfine_runner(weights, crop_cfg,
                                   confidence=args.confidence,
                                   frame_reader=frame_for)
        except Exception as exc:                        # noqa: BLE001
            results.append(od.unavailable(
                config_id, LAUNCH_FAILED, f"{type(exc).__name__}: {exc}",
                crop_digest))
            continue

        if is_sahi:
            log = sahi.SliceLog()
            slice_logs[config_id] = log
            runner = sahi.sahi_runner(base, sahi_cfg, on_slice=log.record)
        else:
            runner = base

        result = od.detect(crops, runner, config_id, detect_cfg, crop_digest)
        if is_sahi:
            result.slices_run = len(slice_logs[config_id].rows)
        results.append(result)

    # ------------------------------------------------------------ persist --
    for result in results:
        directory = f"09_object/{result.config_id}"
        (paths.dir / directory).mkdir(parents=True, exist_ok=True)
        paths.write_json(f"{directory}/metrics.json", result.metrics())
        for detection in result.raw:
            paths.append_jsonl(f"{directory}/raw_detections.jsonl",
                               detection.to_row())
        for detection in result.merged:
            paths.append_jsonl(f"{directory}/merged_detections.jsonl",
                               detection.to_row())
        if result.config_id in slice_logs:
            log = slice_logs[result.config_id]
            for row in log.rows:
                paths.append_jsonl(f"{directory}/slices.jsonl", row)
            paths.write_json(f"{directory}/slice_summary.json", log.summary())
        print(f"  {result.config_id:16} {result.state:22} "
              f"{len(result.merged):5} merged  "
              f"{result.crops_abstained:4} abstained")

    comparison = od.compare(results, crops, crop_digest, taxonomy_approved=False)
    paths.write_json("09_object/comparison.json", comparison)

    # ------------------------------------------- temporal confirmation --
    status.begin("10_evidence")
    # The recurrence grid is a fraction of frame width, so tell it the
    # width. Without this every camera is judged on 1280-pixel cells.
    import av as _av
    with _av.open(str(args.video)) as _c:
        _frame_w = float(_c.streams.video[0].codec_context.width)
    confirm_cfg = tc.ConfirmConfig(frame_width_px=_frame_w)
    print(f"  recurrence grid: {confirm_cfg.recurrence_grid_fraction * _frame_w:.1f} px "
          f"({_frame_w:.0f} px frame)")

    # Wrist positions per frame, for the handheld-plausibility test. Read from
    # the same pose observations the crops were built from, so the two stages
    # cannot disagree about where the hands were.
    wrists: dict = {}
    for row in read_jsonl(observation_path):
        scale = row.get("torso_scale_px")
        if not scale:
            continue
        frame = wrists.setdefault(round(float(row["pts_ms"]), 1), [])
        for joint in row.get("joints", []):
            if joint.get("name") in ("left_wrist", "right_wrist") \
                    and joint.get("observability") == "visible":
                frame.append((float(joint["x"]), float(joint["y"]),
                              float(scale)))
    print(f"  wrist positions available on {len(wrists)} frame(s) from "
          f"{args.pose_source}")

    all_groups = []
    for result in results:
        if not result.available:
            continue
        groups = tc.confirm(result.merged, confirm_cfg)
        # Two post-passes, in this order. Recurrence is a property of the whole
        # set of groups and is invisible from inside any one of them, so it can
        # only run once grouping is finished. Handheld plausibility then demotes
        # what survives but is nowhere near a hand.
        groups = tc.flag_recurrent(groups, confirm_cfg)
        groups = tc.flag_handheld(groups, wrists, confirm_cfg)
        recurrent = sum(1 for g in groups
                        if g.status == tc.RECURRENT_SCENE_OBJECT)
        if recurrent:
            print(f"  {result.config_id}: {recurrent} group(s) restated as "
                  f"recurrent_scene_object")
        all_groups += groups
        for group in groups:
            paths.append_jsonl("10_evidence/temporal_confirmation.jsonl",
                               group.to_row())
    confirmation = tc.summarise(all_groups)
    paths.write_json("10_evidence/temporal_confirmation_summary.json",
                     confirmation)

    print(f"\ncrop equivalence: {comparison['crop_equivalence']}")
    print(f"available: {comparison['available'] or 'none'}")
    print(f"temporal confirmation: {confirmation.get('by_status', {})}")
    print(f"  {confirmation.get('retained_singletons', 0)} singletons retained, "
          f"{confirmation.get('deleted', 0)} deleted")
    print(f"selection: {comparison['selection']}")

    any_detector = any(r.available for r in results)
    status.finish("09_object", COMPLETE if any_detector else BLOCKED,
                  reason=None if any_detector else
                  "no detector configuration was runnable",
                  crops=len(crops), crop_digest=crop_digest)
    status.finish("10_evidence", COMPLETE if any_detector else BLOCKED,
                  reason=None if any_detector else "no detections to confirm",
                  groups=len(all_groups))
    status.write()
    print(f"\nwritten to {paths.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
