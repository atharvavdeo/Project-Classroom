"""Command-line entry point.

    python -m classroom.cli ingest --session "CBT centre" D:/DrishtiAI/*.mkv
    python -m classroom.cli probe   D:/DrishtiAI/some.mkv
    python -m classroom.cli videos
    python -m classroom.cli coverage
    python -m classroom.cli evaluate
"""
from __future__ import annotations

import argparse
import glob as globlib
import sys
from pathlib import Path

from . import annotate, db, evaluate, ingest
from .config import Config


def _expand(patterns: list[str]) -> list[Path]:
    """Expand shell-style patterns, including absolute ones, preserving order."""
    out: list[Path] = []
    for pattern in patterns:
        matches = sorted(globlib.glob(pattern))
        out.extend(Path(m) for m in matches) if matches else out.append(Path(pattern))
    return out


def _session_id(conn, name: str) -> int:
    row = db.one(conn, "SELECT id FROM session WHERE name = ?", (name,))
    if row is not None:
        return int(row["id"])
    return db.insert(conn, "session", name=name)


def cmd_probe(args) -> int:
    failed = 0
    for path in _expand(args.paths):
        try:
            print(ingest.probe(path).summary())
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{path.name}\n  FAILED: {exc}", file=sys.stderr)
            failed += 1
        print()
    return 1 if failed else 0


def cmd_ingest(args) -> int:
    cfg = Config.load(args.config)
    conn = db.connect(cfg.db_path)
    session_id = _session_id(conn, args.session)
    failed = 0

    for path in _expand(args.paths):
        try:
            profile = ingest.probe(path)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {path.name}: {exc}", file=sys.stderr)
            failed += 1
            continue

        if not profile.has_mv:
            print(
                f"FAILED {path.name}: no codec motion vectors; "
                f"the tier-1 scan cannot run on this file",
                file=sys.stderr,
            )
            failed += 1
            continue

        label = args.camera or path.stem
        video_id = ingest.register(conn, profile, session_id, label, args.room)
        print(f"[{video_id}] {profile.summary()}\n")

    conn.close()
    return 1 if failed else 0


def cmd_videos(args) -> int:
    cfg = Config.load(args.config)
    conn = db.connect(cfg.db_path)
    rows = db.all_rows(
        conn,
        """
        SELECT v.id, c.label, c.room, v.codec, v.width, v.height,
               v.avg_fps, v.duration_s, v.frame_count, v.is_vfr, v.mv_per_frame
        FROM source_video v JOIN camera c ON c.id = v.camera_id
        ORDER BY v.id
        """,
    )
    if not rows:
        print("no videos ingested")
        return 0
    print(f"{'id':>3}  {'camera':22}  {'codec':6}  {'res':10}  {'fps':>6}  "
          f"{'dur_s':>8}  {'frames':>7}  {'vfr':>3}  {'mv/f':>6}")
    for r in rows:
        print(
            f"{r['id']:>3}  {r['label'][:22]:22}  {r['codec']:6}  "
            f"{r['width']}x{r['height']:<5}  {r['avg_fps']:>6.2f}  "
            f"{r['duration_s']:>8.2f}  {r['frame_count']:>7}  "
            f"{'yes' if r['is_vfr'] else 'no':>3}  {r['mv_per_frame']:>6.0f}"
        )
    conn.close()
    return 0


def cmd_calib_import(args) -> int:
    from .calibration import Calibration
    from . import calibration as cal_mod

    cfg = Config.load(args.config)
    conn = db.connect(cfg.db_path)
    cal = Calibration.read_json(args.path)
    cal.camera_id = args.camera_id
    cal_id = cal_mod.save(conn, cal)
    print(f"calibration {cal_id} (v{cal.version}) for camera {args.camera_id}: "
          f"{len(cal.seats)} seats, {len(cal.masks)} masks")
    conn.close()
    return 0


def cmd_run(args) -> int:
    from . import pipeline

    cfg = Config.load(args.config)
    conn = db.connect(cfg.db_path)
    result = pipeline.run(conn, args.video_id, args.calibration_id, cfg)
    print(
        f"\njob {result.job_id} complete in {result.seconds:.1f}s\n"
        f"  {result.windows} windows, {result.motion_rows} zone rows\n"
        f"  {result.events} events covering "
        f"{result.candidate_fraction * 100:.2f}% of the recording"
    )
    conn.close()
    return 0


def cmd_events(args) -> int:
    cfg = Config.load(args.config)
    conn = db.connect(cfg.db_path)
    rows = db.all_rows(
        conn,
        """
        SELECT e.id, s.seat_label, e.t_start, e.t_end, e.peak_t, e.peak_deviation
        FROM event e JOIN seat s ON s.id = e.seat_id
        WHERE (? IS NULL OR e.source_video_id = ?)
        ORDER BY e.peak_deviation DESC LIMIT ?
        """,
        (args.video_id, args.video_id, args.limit),
    )
    if not rows:
        print("no events")
        return 0
    print(f"{'id':>4}  {'seat':8}  {'start':>8}  {'end':>8}  {'dur':>6}  {'peak z':>7}")
    for r in rows:
        print(f"{r['id']:>4}  {r['seat_label'][:8]:8}  {r['t_start']:>8.2f}  "
              f"{r['t_end']:>8.2f}  {r['t_end'] - r['t_start']:>6.2f}  "
              f"{r['peak_deviation']:>7.2f}")
    conn.close()
    return 0


def cmd_evaluate(args) -> int:
    cfg = Config.load(args.config)
    conn = db.connect(cfg.db_path)
    print(evaluate.report(conn, args.video_id))
    conn.close()
    return 0


def cmd_coverage(args) -> int:
    """How much footage has been reviewed, per video.

    Separate from `evaluate` because it answers the question that comes first:
    is there enough ground truth yet for any of the other numbers to mean
    anything.
    """
    cfg = Config.load(args.config)
    conn = db.connect(cfg.db_path)
    rows = db.all_rows(
        conn, "SELECT id, path, duration_s FROM source_video ORDER BY id")
    if not rows:
        print("no video ingested")
        conn.close()
        return 0
    print(f"{'id':>4}  {'reviewed':>9}  {'of':>9}  {'share':>6}  "
          f"{'actions':>7}  file")
    total_reviewed = total_duration = 0.0
    for r in rows:
        spans = annotate.load_spans(conn, r["id"])
        reviewed = annotate.coverage_s(spans)
        actions = len(annotate.load_annotations(conn, r["id"]))
        share = reviewed / r["duration_s"] if r["duration_s"] else 0.0
        total_reviewed += reviewed
        total_duration += r["duration_s"]
        print(f"{r['id']:>4}  {reviewed:>8.1f}s  {r['duration_s']:>8.1f}s  "
              f"{share * 100:>5.1f}%  {actions:>7}  {Path(r['path']).name[:44]}")
    share = total_reviewed / total_duration if total_duration else 0.0
    print(f"\n{total_reviewed / 60:.1f} min reviewed of "
          f"{total_duration / 60:.1f} min ({share * 100:.1f}%)")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="classroom")
    parser.add_argument("--config", default=None, help="path to a JSON config overlay")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="measure files without storing anything")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("ingest", help="validate and register source video")
    p.add_argument("paths", nargs="+")
    p.add_argument("--session", required=True)
    p.add_argument("--camera", default=None, help="defaults to the filename stem")
    p.add_argument("--room", default=None, help="used for leave-one-room-out splits")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("videos", help="list ingested video")
    p.set_defaults(func=cmd_videos)

    p = sub.add_parser("calib-import", help="load a calibration JSON into the store")
    p.add_argument("path")
    p.add_argument("--camera-id", type=int, required=True)
    p.set_defaults(func=cmd_calib_import)

    p = sub.add_parser("run", help="motion scan and segmentation for one video")
    p.add_argument("--video-id", type=int, required=True)
    p.add_argument("--calibration-id", type=int, required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("events", help="list ranked events")
    p.add_argument("--video-id", type=int, default=None)
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("coverage", help="how much footage has been annotated")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("evaluate", help="measured accuracy against ground truth")
    p.add_argument("--video-id", type=int, default=None)
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
