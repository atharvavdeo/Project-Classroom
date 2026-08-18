"""Phase 1: ingest one or more videos into a run folder (PRD 6.1, 6.2, 13).

    python tools/validate_video.py --run artifacts/runs/<run_id> "C:/DrishtiAI/*.mkv"
    python tools/validate_video.py --preview "C:/DrishtiAI/03.CCTV Mobile Usage.mkv"

Decodes each source completely, writes the inventory, the per-frame timestamp
index, the per-frame statistics, the health observations and contact sheets into
the PRD 13 layout, and records stage coverage in `RUN_STATUS.json`.

`--preview` runs the same measurements and prints them without writing anything,
for checking a file before committing a run to it.

Coverage accounting is the point of this stage. PRD 18 requires the complete
decodable duration processed or every missing range explicitly failed, and the
status row for `01_ingest` carries exactly that: decoded duration against total
duration, with gaps enumerated rather than absorbed.
"""
from __future__ import annotations

import argparse
import glob as globlib
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts, health, ingest  # noqa: E402
from pipeline.health import HealthConfig, HealthTimeline  # noqa: E402
from pipeline.run_manifest import (  # noqa: E402
    COMPLETE, FAILED, RunManifest, RunStatus,
)

# Contact sheets: a regular grid across the recording, plus one per incident.
REGULAR_SHEET_TILES = 16
INCIDENT_SHEET_TILES = 6


def expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = sorted(globlib.glob(pattern))
        out.extend(Path(m) for m in matches) if matches else out.append(Path(pattern))
    return out


def describe(result: ingest.IngestResult, timeline: HealthTimeline) -> str:
    inventory, index = result.inventory, result.index
    declared, measured = result.declared_vs_measured_fps
    trust = "matches" if result.fps_declaration_trustworthy else "DOES NOT MATCH"
    lines = [
        f"  {inventory.container} / {inventory.codec} / {inventory.pix_fmt}  "
        f"{inventory.width}x{inventory.height}",
        f"  declared {declared:.3f} fps, measured {measured:.3f} fps  ({trust})",
        f"  {index.count} frames, {index.duration_ms / 1000.0:.2f} s, "
        f"{index.keyframe_count} keyframes, "
        f"{'VFR' if index.is_vfr else 'CFR'} (cv {index.interval_cv:.4f})",
        f"  decodable {index.decodable_ms / 1000.0:.2f} s of "
        f"{index.duration_ms / 1000.0:.2f} s"
        + (f", {len(index.gaps)} gap(s) totalling {index.gap_ms / 1000.0:.2f} s"
           if index.gaps else ", no gaps"),
    ]
    if index.repeated_pts_count or index.backwards_pts_count:
        lines.append(f"  repeated PTS {index.repeated_pts_count}, "
                     f"backwards PTS {index.backwards_pts_count}")
    if index.decode_errors:
        lines.append(f"  decode errors: {len(index.decode_errors)}")
        for error in index.decode_errors[:3]:
            lines.append(f"    {error}")
    counts = {k: v for k, v in timeline.counts().items() if v}
    lines.append(f"  health: {counts or 'nothing observed'}")
    return "\n".join(lines)


def process(path: Path, paths: artifacts.RunPaths | None, cfg: HealthConfig,
            verbose: bool = True) -> tuple[ingest.IngestResult, HealthTimeline]:
    """Decode one source and, when a run folder is given, write its artifacts."""
    result = ingest.decode(
        path,
        progress_every_s=60.0 if verbose else 0.0,
        on_progress=(lambda t, i: print(f"    decoded {t:7.1f} s ({i} frames)"))
        if verbose else None,
    )
    observations = health.observe(
        result, cfg,
        baseline_sharpness=ingest.median_sharpness(result.stats),
        # Everything measured directly from this source is validated on source.
        # Conditions raised by fixtures elsewhere carry fixture_only.
        validated_on=health.VALIDATED_ON_SOURCE,
    )
    timeline = HealthTimeline(observations)

    if paths is None:
        return result, timeline

    stem = Path(path).stem[:40]
    paths.append_jsonl("source/video_inventory.jsonl", result.inventory.to_dict())
    for row in result.index.to_rows():
        paths.append_jsonl(f"source/timestamp_index_{stem}.jsonl", row)
    for stat in result.stats:
        paths.append_jsonl(f"01_ingest/decoded_frames_index_{stem}.jsonl",
                           stat.to_row())
    for observation in observations:
        paths.append_jsonl("02_health/observations.jsonl",
                           {"source": str(path), **observation.to_row()})
    for error in result.index.decode_errors:
        paths.append_jsonl("01_ingest/anomalies/decode_errors.jsonl",
                           {"source": str(path), "error": error})

    # Regular contact sheet across the recording.
    times = ingest.regular_sheet_times(result.index, REGULAR_SHEET_TILES)
    if times:
        sheet = ingest.contact_sheet(path, times)
        out = paths.stage("source") / "contact_sheets" / f"{stem}_regular.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # One contact sheet per integrity incident, so a reviewer can see what the
    # detector saw rather than taking a label on trust.
    for observation in observations:
        span = observation.end_pts_ms - observation.start_pts_ms
        step = max(span / max(INCIDENT_SHEET_TILES - 1, 1), 1.0)
        incident_times = [observation.start_pts_ms + i * step
                          for i in range(INCIDENT_SHEET_TILES)]
        incident_times = [t for t in incident_times
                          if result.index.inside_gap(t) is None]
        if not incident_times:
            continue
        try:
            sheet = ingest.contact_sheet(path, incident_times, columns=3)
        except ValueError:
            continue
        folder = paths.stage("02_health") / artifacts.pts_name(
            observation.start_pts_ms, observation.condition, "")
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{stem}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        (folder / "evidence.json").write_text(
            __import__("json").dumps(observation.to_row(), indent=2),
            encoding="utf-8")

    return result, timeline


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--run", type=Path, default=None,
                    help="an existing run folder from tools/preflight_rtx3090.py")
    ap.add_argument("--preview", action="store_true",
                    help="measure and print without writing anything")
    args = ap.parse_args()

    sources = expand(args.sources)
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        print(f"source not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    paths = status = None
    if not args.preview:
        if args.run is None:
            print("--run is required unless --preview is given. Create a run "
                  "folder with tools/preflight_rtx3090.py first.", file=sys.stderr)
            return 2
        paths = artifacts.open_run(args.run.parent, args.run.name)
        if paths.sealed:
            print(f"{args.run} is sealed; reprocessing must create a new run.",
                  file=sys.stderr)
            return 2
        manifest = RunManifest.read(paths)
        status = RunStatus(paths, manifest.run_id)
        status.begin("01_ingest")

    cfg = HealthConfig()
    total_ms = decoded_ms = 0.0
    failures = 0

    for path in sources:
        print(f"\n{path.name}")
        try:
            result, timeline = process(path, paths, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            if paths is not None:
                paths.append_jsonl("01_ingest/anomalies/failures.jsonl",
                                   {"source": str(path),
                                    "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(describe(result, timeline))
        total_ms += result.index.duration_ms
        decoded_ms += result.index.decodable_ms

    if status is not None:
        status.finish(
            "01_ingest",
            FAILED if failures and not decoded_ms else COMPLETE,
            reason=f"{failures} source(s) failed to decode" if failures else None,
            covered_ms=decoded_ms, total_ms=total_ms,
            sources=len(sources), failed=failures,
        )
        status.finish(
            "02_health", COMPLETE,
            covered_ms=decoded_ms, total_ms=total_ms,
        )
        print(f"\nwritten to {paths.dir}")

    share = decoded_ms / total_ms * 100 if total_ms else 0.0
    print(f"\n{decoded_ms / 1000.0:.1f} s decodable of {total_ms / 1000.0:.1f} s "
          f"({share:.2f}%) across {len(sources)} source(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
