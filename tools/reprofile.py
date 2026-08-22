"""Rebuild per-person profiles and flags from stored samples.

    python tools/reprofile.py --run artifacts/runs/1501/01_phone

Profiling and flagging are pure functions over `14_person_timeline/samples.jsonl`.
When a threshold or a rule changes there is no need to re-run detection, tracking
or pose -- the expensive part, several minutes per video on GPU, produced rows
that have not changed.

Rewrites `profiles.json` and `summary.json` in place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import person_timeline as pt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--duration-ms", type=float, default=0.0)
    ap.add_argument("--interval-ms", type=float, default=0.0)
    args = ap.parse_args()

    base = args.run / "14_person_timeline"
    sample_path = base / "samples.jsonl"
    if not sample_path.is_file():
        print(f"no samples.jsonl in {base}", file=sys.stderr)
        return 2

    fields = pt.PersonSample.__dataclass_fields__
    samples = []
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(pt.PersonSample(
            **{k: v for k, v in row.items() if k in fields}))

    duration_ms = args.duration_ms
    interval_ms = args.interval_ms
    notes_path = base / "sample_plan_notes.json"
    if notes_path.is_file():
        notes = json.loads(notes_path.read_text(encoding="utf-8"))
        if notes:
            duration_ms = duration_ms or float(notes[0].get("duration_ms", 0.0))
            hz = float(notes[0].get("sample_hz", 0.0) or 0.0)
            if hz and not interval_ms:
                interval_ms = 1000.0 / hz
    if not duration_ms:
        duration_ms = max(s.pts_ms for s in samples) if samples else 0.0
    if not interval_ms:
        interval_ms = 250.0

    cfg = pt.TimelineConfig()
    timelines = pt.build_timelines(samples)
    profiles = [pt.profile(t, duration_ms, interval_ms, cfg)
                for t in timelines.values()]
    for person in profiles:
        person.flags = pt.flag(person, cfg)
    profiles.sort(key=lambda p: -p.samples)

    report = pt.summarise(profiles, duration_ms, len(samples), cfg)
    report["flags"] = {}
    for person in profiles:
        for item in person.flags:
            report["flags"][item["flag"]] = report["flags"].get(item["flag"], 0) + 1

    (base / "profiles.json").write_text(
        json.dumps([p.to_row() for p in profiles], indent=2), encoding="utf-8")
    (base / "summary.json").write_text(json.dumps(report, indent=2),
                                       encoding="utf-8")

    print(f"{args.run.name}: {len(samples)} samples -> {len(profiles)} profiles")
    for name, count in sorted(report["flags"].items(), key=lambda kv: -kv[1]):
        print(f"   {name:34}{count:5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
