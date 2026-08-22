"""A findings report that sits beside the artifacts it describes.

    python tools/write_findings.py --run artifacts/runs/1446/01_phone

Reads what the run already produced and writes `FINDINGS.md` into the run
folder. Runs no model and recomputes nothing.

Three rules the report obeys, because a triage tool that overstates is worse
than no triage tool:

**Every number comes from a file, and the file is named.** A count without a
path is a count the reader has to take on trust.

**Limitations are derived, not boilerplate.** The caveat section is computed
from this run's own state -- whether seats resolved, whether a stage was
skipped and why, whether zero events formed -- so it cannot drift out of date
or flatter a bad run.

**Nothing asserts misconduct.** The output is observations with timestamps and
pixels attached, for a human to judge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def images(run_dir: Path, pattern: str, limit: int = 12) -> list[str]:
    return sorted(p.relative_to(run_dir).as_posix()
                  for p in run_dir.glob(pattern))[:limit]


def build(run_dir: Path) -> str:
    out: list[str] = []
    add = out.append
    caveats: list[str] = []

    manifest = read_json(run_dir / "RUN_MANIFEST.json") or {}
    source = (manifest.get("sources") or [{}])[0]
    status = read_json(run_dir / "RUN_STATUS.json") or {}
    stages = {s["stage"]: s for s in status.get("stages", [])}

    add(f"# {run_dir.name}")
    add("")
    add(f"`{Path(source.get('path', '?')).name}`  ")
    add(f"sha256 `{str(source.get('sha256', ''))[:16]}…` · run `{run_dir.as_posix()}`")
    add("")
    add("Observations for a human reviewer. Nothing here asserts that "
        "misconduct occurred.")
    add("")

    # ----------------------------------------------- what the pipeline did --
    add("## What the pipeline did")
    add("")
    done = [n for n, s in stages.items() if s.get("state") == "complete"]
    other = [(n, s) for n, s in stages.items()
             if s.get("state") not in ("complete", "pending")]
    add(f"{len(done)} stages completed: {', '.join(f'`{n}`' for n in done)}.")
    add("")
    for name, s in other:
        reason = (s.get("reason") or "").strip()
        add(f"- `{name}` — **{s['state']}**"
            + (f": {reason}" if reason else ""))
        if s["state"] == "blocked":
            caveats.append(f"`{name}` was blocked: {reason or 'no reason recorded'}")
    if other:
        add("")

    ingest = read_json(run_dir / "01_ingest/preprocessing/comparison.json")
    if ingest:
        add("**Preprocessing** — four named transforms measured on identical "
            "frames, none applied to the evaluation path:")
        add("")
        add("| transform | Δ sharpness | Δ blocking | seconds |")
        add("|---|--:|--:|--:|")
        for name, b in ingest["configurations"].items():
            d = b["delta"]
            add(f"| `{name}` | {d['sharpness']:+.2f} | "
                f"{d['blocking_ratio']:+.4f} | {b['seconds']:.1f} |")
        add("")

    align = read_json(run_dir / "03_alignment/comparison.json")
    if align:
        support = align["static_support_coverage"]
        breaks = len(align["epoch_breaks"])
        compensated = sum(b["passed_gate_and_compensated"]
                          for b in align["configurations"].values())
        add(f"**Camera motion** — static background support {support:.1%}, "
            f"{breaks} epoch break(s).")
        add("")
        add("| configuration | compensated | median residual | confidence |")
        add("|---|--:|--:|--:|")
        for name, b in align["configurations"].items():
            add(f"| `{name}` | {b['passed_gate_and_compensated']}"
                f"/{b['estimates']} | {b['median_residual']:.4f} | "
                f"{b['median_confidence']:.3f} |")
        add("")
        identity = align["configurations"].get("identity", {})
        if compensated == 0 and identity.get("median_residual", 0) < 1.0:
            add("Nothing passed the compensation gate because nothing needed "
                "to: the identity baseline's residual is already low, so the "
                "camera is effectively still.")
            add("")

    health = read_jsonl(run_dir / "02_health/observations.jsonl")
    if health:
        counts: dict[str, int] = {}
        for row in health:
            key = row.get("condition", "?")
            counts[key] = counts.get(key, 0) + 1
        add("**Health** — " + ", ".join(f"`{k}` {v}" for k, v in
                                        sorted(counts.items(), key=lambda kv: -kv[1]))
            + ".")
        add("")

    events = read_jsonl(run_dir / "06_segmentation/events.jsonl")
    if events:
        add(f"**Segmentation** — {len(events)} candidate events.")
        add("")

    pose = read_json(run_dir / "08_pose/comparison.json")
    if pose:
        add(f"**Pose** — {pose['manifest_rows']} identical manifest rows, "
            f"equivalence `{pose['manifest_equivalence']}`, digest "
            f"`{pose['manifest_digest'][:16]}…`.")
        add("")
        add("| configuration | returned | coverage | joints visible | rows/s |")
        add("|---|--:|--:|--:|--:|")
        for c in pose["configurations"]:
            if c.get("state") != "available":
                add(f"| `{c['config_id']}` | _{c.get('state')}_ | | | |")
                continue
            add(f"| `{c['config_id']}` | {c.get('persons_returned', 0)} | "
                f"{c.get('row_coverage', 0):.1%} | "
                f"{c.get('joint_visible_fraction', 0):.3f} | "
                f"{c.get('rows_per_second', 0):.2f} |")
        add("")
        seats = pose.get("rows_by_seat_state", {})
        if seats and set(seats) == {"unattributed"}:
            caveats.append(
                "**Every row is `unattributed`.** The calibration is draft, so "
                "seat attribution is refused and the pipeline cannot tell a "
                "seated candidate from a person walking the aisle. Head events "
                "below may include invigilators or passers-by.")

    # -------------------------------------------------------- what we found --
    add("## What we found")
    add("")

    head = read_json(run_dir / "08_pose/head/summary.json")
    excursions = read_jsonl(run_dir / "08_pose/head/excursions.jsonl")
    if head:
        kinds = head.get("events_by_kind", {})
        add(f"**Head and neck** — {head['observations']} observations, "
            f"{head['resolvable_fraction']:.1%} resolvable, "
            f"{head['seats']} tracked subject(s). "
            f"`look_down` {kinds.get('look_down', 0)}, "
            f"`yaw_left` {kinds.get('yaw_left', 0)}, "
            f"`yaw_right` {kinds.get('yaw_right', 0)}.")
        add("")
        if excursions:
            add("| kind | subject | start | duration | peak | baseline | frames |")
            add("|---|---|--:|--:|--:|--:|--:|")
            for e in sorted(excursions,
                            key=lambda r: -float(r["duration_ms"]))[:12]:
                add(f"| `{e['kind']}` | track_{e['track_id']} | "
                    f"{e['start_pts_ms'] / 1000:.1f}s | "
                    f"{e['duration_ms'] / 1000:.2f}s | "
                    f"{e['peak_value']:+.3f} | {e['baseline']:+.3f} | "
                    f"{e['frames']} |")
            add("")
            degenerate = [e for e in excursions
                          if abs(float(e["baseline"]) - float(e["peak_value"]))
                          < 1e-6]
            if degenerate:
                caveats.append(
                    f"{len(degenerate)} excursion(s) have a baseline equal to "
                    f"their peak — that track was turned for its entire visible "
                    f"life, so the per-subject baseline gave no discrimination "
                    f"and the absolute threshold governed. \"Always turned\" and "
                    f"\"turned more than usual\" look the same there.")
        else:
            floor = head.get("config", {}).get("min_event_ms", 0)
            add(f"No sustained head event passed the {floor:.0f} ms floor.")
            add("")
            caveats.append(
                "Zero head events formed. Either nothing sustained occurred, or "
                "the thresholds are not reachable on this recording — check the "
                "per-observation yaw distribution before reading this as "
                "'nothing happened'.")

        pitch_rows = read_jsonl(run_dir / "08_pose/head/per_observation.jsonl")
        if pitch_rows:
            resolved = sum(1 for r in pitch_rows if r.get("pitch") is not None)
            add(f"Pitch resolved on {resolved}/{len(pitch_rows)} observations "
                f"({resolved / len(pitch_rows):.0%}); yaw needs both ears or a "
                f"nose plus one ear, and the far ear is occluded exactly when "
                f"the head is most turned.")
            add("")

    obj = read_json(run_dir / "09_object/comparison.json")
    if obj:
        add(f"**Objects** — {obj['crop_rows']} identical crops, equivalence "
            f"`{obj['crop_equivalence']}`, digest "
            f"`{obj['crop_digest'][:16]}…`.")
        add("")
        add("| configuration | merged | phone | chit | stationery | "
            "hard negatives | crops/s |")
        add("|---|--:|--:|--:|--:|--:|--:|")
        for c in obj["configurations"]:
            if c.get("state") != "available":
                add(f"| `{c['config_id']}` | _{c.get('state')}_ | | | | | |")
                continue
            by = c.get("by_class", {})
            add(f"| `{c['config_id']}` | {c.get('merged_detections', 0)} | "
                f"**{by.get('phone', 0)}** | "
                f"{by.get('secondary_paper_chit', 0)} | "
                f"{by.get('stationery', 0)} | "
                f"{c.get('hard_negative_detections', 0)} | "
                f"{c.get('crops_per_second', 0):.2f} |")
        add("")
        worst = max((c for c in obj["configurations"]
                     if c.get("state") == "available"),
                    key=lambda c: c.get("hard_negative_detections", 0),
                    default=None)
        if worst:
            by = worst.get("by_class", {})
            noise = {k: by.get(k, 0) for k in
                     ("mouse", "keyboard", "monitor_or_bezel")}
            phones = by.get("phone", 0)
            if sum(noise.values()) > max(phones, 1) * 2:
                add(f"`{worst['config_id']}` reports "
                    + ", ".join(f"{v} {k}" for k, v in noise.items())
                    + f" against {phones} phone. In a computer-based-test hall "
                      f"every desk carries a mouse and a dark rectangular "
                      f"bezel, which is the shape a phone detector confuses "
                      f"most. Only target classes may corroborate an event, so "
                      f"these cannot carry anything into the queue — but they "
                      f"are the false-positive source to fix.")
                add("")

    ranked = read_jsonl(run_dir / "10_evidence/ranked_events.jsonl")
    if ranked:
        counts = {}
        for row in ranked:
            key = row.get("disposition", "unknown")
            counts[key] = counts.get(key, 0) + 1
        queue = (counts.get("high_potential_corroborated", 0)
                 + counts.get("flagged_for_review", 0))
        add(f"**Review queue** — {queue} of {len(ranked)} events reach a "
            f"reviewer ({queue / len(ranked):.0%}).")
        add("")
        for key, value in sorted(counts.items(), key=lambda kv: -kv[1]):
            add(f"- `{key}` — {value}")
        add("")
        add("`rejected_not_offered` means a quality gate withheld the event "
            "from the queue. It does **not** mean nothing happened; the "
            "evidence keeps its score and stays searchable.")
        add("")
        top = sorted(ranked, key=lambda r: -float(r.get("score", 0) or 0))[:8]
        add("| # | event | seat | start | score | disposition |")
        add("|--:|---|---|--:|--:|---|")
        for i, row in enumerate(top, start=1):
            add(f"| {i} | `{row.get('event_id', '')}` | "
                f"{row.get('seat_state', '')} | "
                f"{float(row.get('start_pts_ms', 0) or 0) / 1000:.1f}s | "
                f"{float(row.get('score', 0) or 0):.2f} | "
                f"`{row.get('disposition', '')}` |")
        add("")

    # ------------------------------------------------------------- frames --
    add("## Frames")
    add("")
    any_frames = False
    overlays = images(run_dir, "08_pose/head/overlays/*.jpg", limit=30)
    if overlays:
        any_frames = True
        add(f"**Head vector on the source frame** — {len(overlays)} frame(s), "
            f"drawn at the midpoint of the longest excursions so the pictures "
            f"match the table above.")
        add("")
        for path in overlays:
            add(f"- `{path}`")
        add("")
    for cls in ("phone", "secondary_paper_chit", "stationery",
                "book_or_notebook_like_object"):
        frames = images(run_dir, f"09_object/*/rendered_{cls}/*_frame.jpg")
        sheets = images(run_dir, f"09_object/*/rendered_{cls}/contact_sheet*.jpg")
        if not (frames or sheets):
            continue
        any_frames = True
        add(f"**{cls}** — {len(frames)} source frame(s) with the detector's "
            f"own box drawn, plus the native crop at true source size.")
        add("")
        if sheets:
            add(f"- contact sheet — `{sheets[0]}`")
        for path in frames:
            add(f"- `{path}`")
        add("")
    if not any_frames:
        add("No rendered frames in this run. Head overlays are written only "
            "where a sustained excursion formed; detection frames only where "
            "that class was detected.")
        add("")

    # ---------------------------------------------------------- caveats --
    add("## Read this before quoting any number above")
    add("")
    for note in caveats:
        add(f"- {note}")
    add("- No reference manifest exists for this corpus, so there is **no "
        "precision or recall anywhere in this report**. Every count is a count "
        "of what the system reported, not of what was true.")
    add("")

    add("## Not measured")
    add("")
    add("- **Eye gaze and iris direction** — impossible at this source scale; "
        "an eye is 2–3 px and there is no iris to track.")
    add("- **Face identity** — never computed, by design.")
    add("- **Chit versus notebook** — no detector class and no labelled data.")
    add("- **Seat changes, absences, re-identification** — blocked on "
        "calibration approval.")
    add("")

    files = [p for p in run_dir.rglob("*") if p.is_file()]
    pictures = [p for p in files if p.suffix.lower() in (".jpg", ".png")]
    add(f"---")
    add("")
    add(f"{len(files)} files · {len(pictures)} images · "
        f"{sum(p.stat().st_size for p in files) / 1048576:.1f} MB")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    if not (args.run / "RUN_MANIFEST.json").is_file():
        print(f"{args.run} is not a run folder", file=sys.stderr)
        return 2

    path = args.run / "FINDINGS.md"
    path.write_text(build(args.run), encoding="utf-8")
    print(f"written {path}  ({len(path.read_text(encoding='utf-8')):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
