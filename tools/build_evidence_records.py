"""Turn a finished run into evidence records and fused track outcomes.

    python tools/build_evidence_records.py --run artifacts/runs/1505/03_mobile

Writes, into the run's `14_person_timeline` stage directory:

    evidence_records.jsonl   one record per object proposal, rejections included
    track_outcomes.json      one fused state per track, with its reason chain
    contract_summary.json    the funnel, plus the self-check below

### This tool invents nothing

Every field is read from artifacts already on disk, or recomputed by calling
the same gate the original run called (`chit_evidence.gate_detection`). Nothing
is re-inferred, no model is run, and no threshold is chosen here.

### The self-check, and why it is a hard failure

The gate is recomputed rather than read, because the per-detection rejection
reason was never stored -- only per-track totals were. Recomputation is only
legitimate if it reproduces those totals exactly, so the tool compares its own
kept-count against `chit_evidence.json`'s `kept_detections` per track and exits
non-zero on any mismatch.

A mismatch means the run used a different `EvidenceConfig` than this invocation
(the newsclip run raised `max_size_ratio` to 0.20 for its 360x450 footage), and
the records would carry rejection reasons that never happened. Reporting that
as a successful conversion is exactly the class of quiet error this project has
already been bitten by once, on VFR timestamps.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import chit_evidence as ce            # noqa: E402
from pipeline import evidence_record as er          # noqa: E402
from pipeline import fusion                         # noqa: E402
from pipeline import reason_codes as rc             # noqa: E402

CHIT_SOURCE = "roboflow_chit_paper_new_v2"
CHIT_MODEL = ("chit-paper-new", "2")

# SAM 3 verdict -> reason code. `unsupported` deliberately maps to
# SAM3_NOT_CONFIRMED and not to anything that reads as a negative finding.
SAM3_VERDICT_CODE = {
    "corroborated": rc.SAM3_SUPPORTED.code,
    "reclassified_as_phone": rc.SAM3_PHONE_NAMED.code,
    # A D-FINE phone proposal the referee agreed with. Same code as the
    # escalation above: what matters downstream is that SAM 3 named a phone,
    # not which detector raised the question.
    # `phone_confirmed` is accepted for historic run folders only. New runs
    # write `phone_supported`: SAM 3 supports an object claim, never a human
    # policy decision.
    "phone_confirmed": rc.SAM3_PHONE_NAMED.code,
    "phone_supported": rc.SAM3_PHONE_NAMED.code,
    # COCO said phone, SAM 3 said paper. The phone claim is not confirmed;
    # the paper reading is already carried by the chit path.
    "phone_was_paper": rc.SAM3_NOT_CONFIRMED.code,
    "suppressed": rc.SAM3_EQUIPMENT_CONTEXT.code,
    "unsupported": rc.SAM3_NOT_CONFIRMED.code,
    "not_adjudicated": rc.SAM3_NOT_RUN.code,
}

GATE_REASON_CODE = {
    "too_large": rc.OBJECT_TOO_LARGE.code,
    "far_from_hands": rc.OBJECT_AWAY_FROM_WRIST.code,
    "no_wrist": rc.OBJECT_NO_WRIST_RESOLVED.code,
    "low_confidence": rc.TARGET_BELOW_RESOLUTION.code,
}


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def key_of(track_id, pts_ms, box) -> tuple:
    """Identity of one detection across the raw and adjudicated files."""
    return (track_id, round(float(pts_ms), 1),
            tuple(round(float(v), 1) for v in box))


def sam3_index(rows: list) -> dict:
    """Every SAM 3 verdict, keyed by the detection it was passed."""
    out = {}
    for row in rows:
        for item in (row.get("near_hand_objects") or []):
            if not isinstance(item, dict) or not item.get("sam3"):
                continue
            out[key_of(row["track_id"], row["pts_ms"], item.get("box") or [0] * 4)] \
                = item["sam3"]
    return out


def build(run_dir: Path, cfg: ce.EvidenceConfig) -> tuple:
    stage = run_dir / "14_person_timeline"
    manifest = read_json(run_dir / "RUN_MANIFEST.json", {})
    raw = read_jsonl(stage / "samples_with_chits.jsonl")
    if not raw:
        raise SystemExit(f"{stage}: no samples_with_chits.jsonl — this run "
                         f"never had the chit detector attached.")

    verdicts = sam3_index(read_jsonl(stage / "samples_sam3_adjudicated.jsonl"))
    summary = read_json(stage / "sam3_adjudication_summary.json", {})
    adjudicated_run = bool(summary)

    video = str(manifest.get("video") or manifest.get("source") or run_dir.name)
    run_id = f"{run_dir.parent.name}/{run_dir.name}"

    records, kept_count = [], collections.Counter()
    for row in raw:
        track_id = row["track_id"]
        person_box = row.get("box") or [0.0] * 4
        person_w = max(person_box[2] - person_box[0], 1.0)
        person_h = max(person_box[3] - person_box[1], 1.0)

        for item in (row.get("near_hand_objects") or []):
            if not isinstance(item, dict):
                continue
            source = item.get("source") or ""
            is_chit = source == CHIT_SOURCE

            kept, gate_reason, size_ratio, distance = ce.gate_detection(
                row, item, cfg) if is_chit else (False, "", 0.0, None)

            stored_distance = item.get("wrist_distance_norm")
            if stored_distance is not None:
                try:
                    stored_distance = float(stored_distance)
                except (TypeError, ValueError):
                    stored_distance = None
            wrist_distance = stored_distance if stored_distance is not None else distance
            record = er.EvidenceRecord(
                provenance=er.Provenance(
                    run_id=run_id, video=video, camera=run_dir.name,
                    pts_ms=float(row["pts_ms"]), stage="14_person_timeline"),
                detector=er.Detector(
                    name=CHIT_MODEL[0] if is_chit else "d-fine",
                    version=CHIT_MODEL[1] if is_chit else "coco",
                    class_name=str(item.get("cls") or ""),
                    confidence=float(item.get("confidence") or 0.0)),
                geometry=er.Geometry(
                    box=tuple(float(v) for v in (item.get("box") or [0.0] * 4)),
                    person_box=tuple(float(v) for v in person_box),
                    size_ratio_to_person=round(float(size_ratio), 5),
                    torso_scale_px=float(row.get("torso_scale_px") or 0.0)),
                association=er.Association(
                    track_id=track_id,
                    seat=str(row.get("seat_state") or "unattributed"),
                    seat_calibrated=bool(row.get("seat_state")
                                         and row["seat_state"] != "unattributed"),
                    wrist_resolved=bool(row.get("left_wrist")
                                        or row.get("right_wrist")),
                    wrist_distance_norm=(round(wrist_distance, 4)
                                         if wrist_distance is not None else -1.0),
                    nearest_wrist=str(item.get("nearest_wrist") or "")),
                quality=er.Quality(
                    person_box_px=int(min(person_w, person_h)),
                    head_resolvable=row.get("yaw") is not None,
                    hands_resolvable=bool(row.get("left_wrist")
                                          or row.get("right_wrist"))),
            )

            if not is_chit:
                # D-FINE is workstation context -- with one exception. Its
                # COCO `phone` class is a *proposal*, which SAM 3 then verifies
                # against mouse/keyboard/monitor. It still may not route anyone
                # to review on its own; only SAM3_PHONE_NAMED does that, and
                # only when the association condition also passes.
                record.add_reason(rc.DFINE_OBJECT_CONTEXT.code)
                if str(item.get("cls") or "") == "phone":
                    record.add_reason(rc.OBJECT_PROPOSAL_PHONE.code)
                    verdict = verdicts.get(key_of(
                        track_id, row["pts_ms"], item.get("box") or [0.0] * 4))
                    if verdict:
                        record.sam3 = er.Sam3Response(
                            attempted=True, responded=True,
                            prompt=str(summary.get("prompt") or ""),
                            model=str(summary.get("workflow") or ""))
                        code = SAM3_VERDICT_CODE.get(verdict.get("verdict", ""))
                        if code:
                            record.add_reason(code)
                        if verdict.get("as"):
                            record.sam3.negative_claims.append(er.Sam3Claim(
                                phrase=str(verdict["as"]),
                                confidence=float(verdict.get("confidence") or 0.0)))
                        if verdict.get("verdict") in ("phone_confirmed",
                                                      "phone_supported"):
                            record.sam3.target_claims.append(er.Sam3Claim(
                                phrase="mobile phone",
                                confidence=float(verdict.get("confidence") or 0.0)))
                    elif adjudicated_run:
                        record.add_reason(rc.SAM3_NOT_RUN.code)
                records.append(record.assign_id())
                continue

            record.add_reason(rc.OBJECT_PROPOSAL_PAPER.code)
            if kept:
                kept_count[track_id] += 1
                record.add_reason(rc.OBJECT_AT_WRIST.code)
            elif gate_reason in GATE_REASON_CODE:
                record.add_reason(GATE_REASON_CODE[gate_reason])

            verdict = verdicts.get(
                key_of(track_id, row["pts_ms"], item.get("box") or [0.0] * 4))
            if verdict:
                record.sam3 = er.Sam3Response(
                    attempted=True, responded=True,
                    prompt=str(summary.get("prompt") or ""),
                    model=str(summary.get("workflow") or ""))
                code = SAM3_VERDICT_CODE.get(verdict.get("verdict", ""))
                if code:
                    record.add_reason(code)
                for claim in ([verdict["as"]] if verdict.get("as") else []):
                    record.sam3.negative_claims.append(
                        er.Sam3Claim(phrase=str(claim),
                                     confidence=float(verdict.get("confidence")
                                                      or 0.0)))
                if verdict.get("verdict") in ("corroborated",
                                              "reclassified_as_phone",
                                              "phone_supported"):
                    record.sam3.target_claims.append(
                        er.Sam3Claim(phrase=str(verdict.get("verdict")),
                                     confidence=float(verdict.get("confidence")
                                                      or 0.0)))
            elif kept and adjudicated_run:
                # Gated but the referee never saw it. Not the same as cleared.
                record.sam3 = er.Sam3Response(attempted=True, responded=False,
                                              error="not returned by the run")
                record.add_reason(rc.SAM3_UNAVAILABLE.code)
            elif kept:
                record.add_reason(rc.SAM3_NOT_RUN.code)

            records.append(record.assign_id())

    return records, kept_count, adjudicated_run, summary


def verify(kept_count, evidence_rows, cfg) -> list:
    """Recomputed gate must reproduce the run's own kept-counts, exactly."""
    problems = []
    for row in evidence_rows:
        expected = int(row.get("kept_detections") or 0)
        got = kept_count.get(row["track_id"], 0)
        if expected != got:
            problems.append(f"track {row['track_id']}: run recorded "
                            f"{expected} kept, recompute gives {got}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--max-size-ratio", type=float,
                    default=ce.EvidenceConfig.max_size_ratio,
                    help="must match the value the run was gated with; the "
                         "self-check fails loudly if it does not")
    args = ap.parse_args()

    run_dir = args.run.resolve()
    stage = run_dir / "14_person_timeline"
    cfg = ce.EvidenceConfig(max_size_ratio=args.max_size_ratio)

    records, kept_count, adjudicated, summary = build(run_dir, cfg)
    evidence_rows = read_json(stage / "chit_evidence.json", [])

    problems = verify(kept_count, evidence_rows, cfg)
    if problems:
        print(f"ERROR: recomputed gate does not reproduce this run "
              f"({len(problems)} track(s) disagree). The run was gated with a "
              f"different EvidenceConfig than --max-size-ratio "
              f"{args.max_size_ratio}. Refusing to write records that would "
              f"carry rejection reasons the run never applied.", file=sys.stderr)
        for line in problems[:10]:
            print(f"  {line}", file=sys.stderr)
        return 2

    by_track = collections.defaultdict(list)
    for record in records:
        by_track[record.association.track_id].append(record)

    profiles = read_json(stage / "profiles.json", [])
    evidence_by_track = {row["track_id"]: row for row in evidence_rows}

    outcomes = [
        fusion.route(profile,
                     evidence_by_track.get(profile["track_id"]),
                     # Both proposal sources reach fusion: the chit model's
                     # papers, and D-FINE's phone proposals. Every other D-FINE
                     # class stays out -- it is desk context, not a claim about
                     # the candidate.
                     [r for r in by_track.get(profile["track_id"], [])
                      if r.detector.name == CHIT_MODEL[0]
                      or r.has(rc.OBJECT_PROPOSAL_PHONE.code)])
        for profile in profiles
    ]
    for outcome in outcomes:
        fusion.assert_machine_state(outcome.state)

    written = er.write_jsonl(records, stage / "evidence_records.jsonl")
    (stage / "track_outcomes.json").write_text(
        json.dumps([{
            "track_id": o.track_id, "seat": o.seat, "state": o.state,
            "reason_codes": o.reason_codes, "modalities": o.modalities,
            "priority": round(o.priority, 3),
            "conditions": [{"name": c.name, "passed": c.passed,
                            "detail": c.detail} for c in o.conditions],
            "funnel": {k: getattr(o, k) for k in
                       ("proposals", "gated", "sam3_supported",
                        "sam3_equipment", "sam3_not_confirmed",
                        "sam3_unavailable", "sam3_phone")},
        } for o in outcomes], indent=1), encoding="utf-8")

    report = {
        "contract_version": er.CONTRACT_VERSION,
        "run": f"{run_dir.parent.name}/{run_dir.name}",
        "records_written": written,
        "gate_config": {"max_size_ratio": cfg.max_size_ratio,
                        "max_wrist_distance": cfg.max_wrist_distance,
                        "min_confidence": cfg.min_confidence},
        "sam3_adjudicated": adjudicated,
        "sam3_api_errors": int(summary.get("api_errors") or 0),
        "gate_self_check": "passed — recomputed kept-counts match the run",
        **fusion.funnel(outcomes),
    }
    (stage / "contract_summary.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")

    print(f"{report['run']}: {written} evidence records, "
          f"{len(outcomes)} tracks")
    for state, count in report["states"].items():
        if count:
            print(f"  {state:22s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
