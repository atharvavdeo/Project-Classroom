"""PRD 6.3.7: run every named preprocessing transform enabled and disabled.

    python tools/run_preprocessing_ab.py --run artifacts/runs/<run_id> VIDEO

Writes `01_ingest/preprocessing/` -- per transform, the raw input, the
transformed output, the difference image and the measured statistics, plus a
comparison that recommends nothing until references exist.

Nothing this writes is consumed by any later stage. That is deliberate: PRD 6
says preprocessing "must never replace source pixels in final evaluation", so
this tool measures transforms and stops there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, preprocessing as pp  # noqa: E402
from pipeline.run_manifest import COMPLETE, RunManifest, RunStatus  # noqa: E402


def sample_frames(video: Path, count: int):
    """Frames spread across the whole recording, decoded in one forward pass."""
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--save-images", type=int, default=3,
                    help="how many frames per transform to persist as images")
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    if paths.sealed:
        print(f"{args.run} is sealed.", file=sys.stderr)
        return 2
    status = RunStatus(paths, RunManifest.read(paths).run_id)

    frames = sample_frames(args.video, args.frames)
    if not frames:
        print("no frames decoded", file=sys.stderr)
        return 2
    print(f"{len(frames)} frames sampled across {args.video.name}")

    try:
        import cv2
    except Exception:                                    # noqa: BLE001
        cv2 = None

    out_dir = paths.stage("01_ingest") / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = pp.PreprocessConfig()
    comparisons = []

    for config_id in pp.PREPROCESS_CONFIGS:
        saved = {"n": 0}

        def persist(pts_ms, raw, transformed, delta, _c=config_id, _s=saved):
            if cv2 is None or _s["n"] >= args.save_images:
                return
            prefix = artifacts.pts_prefix(pts_ms)
            base = out_dir / _c
            base.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(base / f"{prefix}_raw.jpg"), raw)
            cv2.imwrite(str(base / f"{prefix}_transformed.jpg"), transformed)
            cv2.imwrite(str(base / f"{prefix}_difference.jpg"), delta)
            _s["n"] += 1

        comparison = pp.compare(frames, config_id, cfg, on_frame=persist)
        comparison.artifacts = [f"01_ingest/preprocessing/{config_id}"]
        comparisons.append(comparison)
        print(f"  {config_id:26} mean|diff| {comparison.mean_abs_difference:7.3f}  "
              f"sharpness {comparison.delta.get('sharpness', 0):+9.2f}  "
              f"blocking {comparison.delta.get('blocking_ratio', 0):+7.4f}  "
              f"{comparison.seconds:5.1f}s")

    report = pp.summarise(comparisons)
    paths.write_json("01_ingest/preprocessing/comparison.json", report)
    for comparison in comparisons:
        paths.append_jsonl("01_ingest/preprocessing/measurements.jsonl",
                           comparison.to_row())

    print(f"\n{report['recommendation']}")
    status.finish("01_ingest", COMPLETE,
                  reason="decode pass plus preprocessing comparison",
                  preprocessing_configs=len(comparisons))
    status.write()
    print(f"\nwritten {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
