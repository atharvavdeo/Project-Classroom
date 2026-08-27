"""Condense a finished run into one JSON the review console can load.

Why this exists: a run's own artifacts are the wrong shape for a browser.
`evidence_records.jsonl` is 38 MB and `samples.jsonl` is 12 MB for a 88-second
recording; shipping either to a client would be slow and would also hand the
browser 37,041 records to filter, which is work the pipeline already did.

So this is a *projection*, not a summary. Every number it emits is copied from
an artifact, never recomputed, because a console that disagrees with the run
folder is worse than no console. The one thing it does compute is episode
spans for the timeline, and those come from the evidence records' own reason
codes rather than from a fresh threshold.

Deliberately NOT included:
  - raw detections that never reached a person (they are in the run folder)
  - joints, masks, or any per-frame geometry beyond what a lane needs
  - anything derived. No rates, no accuracy, no per-person score.

Usage:
    python tools/build_console_bundle.py --run artifacts/runs/1512/12_paper \\
        --out console/public/data/1512_12_paper.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# The five presentation lanes. Each maps reason-code domains and profile flags
# onto one timeline row. The order here is the render order, top to bottom.
LANES = ("object", "orientation", "hands", "presence", "quality")

FLAG_LANE = {
    "target_object_near_hands": "object",
    "persistent_object_beside_seat": "object",
    "sustained_attention_away": "orientation",
    "frequent_orientation_change": "orientation",
    "prolonged_downward_orientation": "orientation",
    "no_orientation_baseline": "orientation",
    "hand_below_desk_line": "hands",
    "absent_from_frame": "presence",
    "low_coverage": "quality",
}

# Orientation labels are their own thing: they carry a dashboard_action the
# client switches on, and they may never be rendered as a flag.
ORIENTATION_LANE = "orientation"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _episodes(stage: Path) -> dict:
    """Object episode spans per track, taken from the chit evidence artifact.

    Falls back to an empty mapping rather than inventing spans: a missing
    artifact must read as "not computed", never as "this person had none".
    """
    doc = _read_json(stage / "chit_evidence.json", None)
    if doc is None:
        return {}
    rows = doc if isinstance(doc, list) else doc.get("tracks") or doc.get("evidence") or []
    out: dict[str, list] = defaultdict(list)
    for row in rows:
        track = str(row.get("track_id"))
        for span in row.get("episodes") or []:
            if isinstance(span, dict):
                start, end = span.get("from_ms"), span.get("to_ms")
            elif isinstance(span, (list, tuple)) and len(span) >= 2:
                start, end = span[0], span[1]
            else:
                continue
            if start is None or end is None:
                continue
            out[track].append({"from_ms": float(start), "to_ms": float(end)})
    return dict(out)


def _stages(run: Path) -> list:
    doc = _read_json(run / "RUN_STATUS.json", {})
    return [{"stage": s.get("stage"), "state": s.get("state"),
             "reason": s.get("reason"),
             "covered_ms": s.get("covered_ms"), "total_ms": s.get("total_ms")}
            for s in doc.get("stages", [])]


def _manifest(run: Path) -> dict:
    doc = _read_json(run / "RUN_MANIFEST.json", {})
    return {
        "run_id": doc.get("run_id"),
        "code_revision": doc.get("code_revision"),
        "code_dirty": doc.get("code_dirty"),
        "calibration_version": doc.get("calibration_version"),
        "seed": doc.get("seed"),
        "created_utc": doc.get("created_utc"),
        "sources": doc.get("sources"),
        # Model versions matter on every card, so they travel with the bundle.
        "models": doc.get("models"),
    }


def build(run: Path) -> dict:
    stage = run / "14_person_timeline"
    outcomes = _read_json(stage / "track_outcomes.json", [])
    profiles = _read_json(stage / "profiles.json", [])
    summary = _read_json(stage / "summary.json", {})
    contract = _read_json(stage / "contract_summary.json", {})
    sam3 = _read_json(stage / "sam3_adjudication_summary.json", {})
    chit = _read_json(stage / "chit_detection_summary.json", {})
    episodes = _episodes(stage)

    by_track = {str(p.get("track_id")): p for p in profiles}
    duration_ms = float(summary.get("duration_ms") or 0.0)

    tracks = []
    for row in outcomes:
        tid = str(row.get("track_id"))
        profile = by_track.get(tid, {})

        # Flags, bucketed into lanes. An unmapped flag lands in "quality"
        # rather than being dropped -- a flag the console does not recognise
        # is still a flag someone should see.
        flags = []
        for flag in profile.get("flags") or []:
            name = flag.get("flag", "")
            flags.append({
                "flag": name,
                "lane": FLAG_LANE.get(name, "quality"),
                "severity": flag.get("severity", "info"),
                "evidence": flag.get("evidence", ""),
                "why": flag.get("why", ""),
            })

        tracks.append({
            "track_id": row.get("track_id"),
            "state": row.get("state"),
            "priority": row.get("priority"),
            "seat": row.get("seat"),
            "seat_state": profile.get("seat_state"),
            "modalities": row.get("modalities") or [],
            "reason_codes": row.get("reason_codes") or [],
            "conditions": row.get("conditions") or [],
            "funnel": row.get("funnel"),

            "samples": profile.get("samples"),
            "resolvable_samples": profile.get("resolvable_samples"),
            "first_seen_ms": profile.get("first_seen_ms"),
            "last_seen_ms": profile.get("last_seen_ms"),
            "coverage_of_recording": profile.get("coverage_of_recording"),
            "median_head_scale_px": profile.get("median_head_scale_px"),

            "baseline_yaw": profile.get("baseline_yaw"),
            "baseline_pitch": profile.get("baseline_pitch"),
            "baseline_yaw_relative_to_torso":
                profile.get("baseline_yaw_relative_to_torso"),

            "flags": flags,
            "orientation_labels": profile.get("orientation_labels") or [],
            "orientation_events": profile.get("orientation_events") or [],
            "absences": profile.get("absences") or [],
            "episodes": episodes.get(tid, []),
            "near_hand_object_counts": profile.get("near_hand_object_counts") or {},
        })

    # Queue order is the pipeline's, not the console's. Sorting here keeps the
    # client from having to know that priority is descending.
    tracks.sort(key=lambda t: -(t.get("priority") or 0.0))

    states = defaultdict(int)
    for t in tracks:
        states[t["state"]] += 1

    return {
        "schema": 1,
        "run": _manifest(run),
        "video": summary.get("video") or (run.name),
        "duration_ms": duration_ms,
        "lanes": list(LANES),
        "counts": {
            "tracks": len(tracks),
            "states": dict(states),
            # Observability denominators. The console is built so these cannot
            # be omitted from any band it renders.
            "samples": summary.get("sampled_rows"),
            "people_with_a_yaw_baseline": summary.get("people_with_a_yaw_baseline"),
        },
        "coverage_of_recording": summary.get("coverage_of_recording") or {},
        "coverage_caveat": summary.get("coverage_caveat", ""),
        "note": summary.get("note", ""),
        "stages": _stages(run),
        "sam3": sam3,
        "chit": chit,
        "contract": contract,
        "tracks": tracks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--indent", type=int, default=0,
                    help="0 writes the compact form the console loads; use 2 "
                         "when a human needs to read the bundle")
    args = ap.parse_args()

    if not (args.run / "14_person_timeline").exists():
        print(f"ERROR: {args.run} has no 14_person_timeline stage")
        return 1

    bundle = build(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(bundle, indent=args.indent or None,
                   separators=(",", ":") if not args.indent else None),
        encoding="utf-8")

    size_kb = args.out.stat().st_size / 1024
    states = bundle["counts"]["states"]
    print(f"{args.run.parent.name}/{args.run.name}: "
          f"{bundle['counts']['tracks']} tracks -> {args.out} ({size_kb:.0f} KB)")
    for state, n in sorted(states.items(), key=lambda kv: -kv[1]):
        print(f"  {state:22s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
