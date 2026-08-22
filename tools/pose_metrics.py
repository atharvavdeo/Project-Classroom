"""Per-model pose metrics: angles, confidence, and temporal stability (PRD 9).

    python tools/pose_metrics.py --run artifacts/runs/corpus_03

Reads the persisted `08_pose/<config>/observations.jsonl` rows. Computes nothing
the models did not produce; every figure is derived from stored keypoints.

Three families of measure, because "joints visible" alone ranks a model that
confidently invents keypoints above one that declines:

**Angles.** Head pitch (nose below the shoulder line, normalised by shoulder
width), torso lean, and both elbow flexion angles. These are the observable
temporal quantities PRD 9 permits pose to report.

**Confidence.** Median and spread per joint group. A model whose wrist
confidence sits just above the floor is not the same as one well clear of it,
and the mean hides that.

**Temporal stability.** The frame-to-frame jitter of each joint on the same
track, normalised by torso scale so a distant candidate is not penalised for
being small. This is the "stable results over a long run" question: a model that
finds a wrist every frame but puts it somewhere different each time is not
producing usable temporal evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip")
VISIBLE = "visible"


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def joint_map(row):
    return {j["name"]: j for j in row.get("joints", [])}


def angle(a, b, c):
    """Interior angle at b, in degrees, or None if any point is missing."""
    if not (a and b and c):
        return None
    v1 = (a["x"] - b["x"], a["y"] - b["y"])
    v2 = (c["x"] - b["x"], c["y"] - b["y"])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def summarise(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    values.sort()
    return {
        "n": len(values),
        "median": round(statistics.median(values), 2),
        "p10": round(values[int(0.10 * (len(values) - 1))], 2),
        "p90": round(values[int(0.90 * (len(values) - 1))], 2),
        "iqr": round(values[int(0.75 * (len(values) - 1))]
                     - values[int(0.25 * (len(values) - 1))], 2),
    }


def analyse(rows):
    """Angles, per-joint confidence, and jitter for one configuration."""
    angles = {"head_pitch_norm": [], "torso_lean_deg": [],
              "left_elbow_deg": [], "right_elbow_deg": []}
    confidence = {j: [] for j in JOINTS}
    observability = {j: {} for j in JOINTS}
    torso_scales = []

    by_track = {}
    for row in rows:
        jm = joint_map(row)
        for name in JOINTS:
            joint = jm.get(name)
            if not joint:
                continue
            state = joint["observability"]
            observability[name][state] = observability[name].get(state, 0) + 1
            if state == VISIBLE:
                confidence[name].append(joint["confidence"])

        def vis(name):
            j = jm.get(name)
            return j if j and j["observability"] == VISIBLE else None

        ls, rs = vis("left_shoulder"), vis("right_shoulder")
        scale = row.get("torso_scale_px")
        if scale:
            torso_scales.append(scale)
        if row.get("head_pitch") is not None:
            angles["head_pitch_norm"].append(row["head_pitch"])
        if ls and rs:
            angles["torso_lean_deg"].append(
                abs(math.degrees(math.atan2(ls["y"] - rs["y"],
                                            ls["x"] - rs["x"]))) % 180.0)
        angles["left_elbow_deg"].append(
            angle(vis("left_shoulder"), vis("left_elbow"), vis("left_wrist")))
        angles["right_elbow_deg"].append(
            angle(vis("right_shoulder"), vis("right_elbow"), vis("right_wrist")))

        key = row.get("pose_row_id", "")
        by_track.setdefault(round(row["pts_ms"], 1), []).append((key, jm, scale))

    # Frame-to-frame jitter, normalised by torso scale.
    times = sorted(by_track)
    jitter = {j: [] for j in JOINTS}
    for earlier, later in zip(times, times[1:]):
        if later - earlier > 2000.0:
            continue                      # not consecutive; a gap is not jitter
        for _k, a_map, a_scale in by_track[earlier]:
            for _k2, b_map, b_scale in by_track[later]:
                scale = a_scale or b_scale
                if not scale:
                    continue
                for name in JOINTS:
                    a, b = a_map.get(name), b_map.get(name)
                    if not (a and b):
                        continue
                    if a["observability"] != VISIBLE or b["observability"] != VISIBLE:
                        continue
                    d = math.hypot(a["x"] - b["x"], a["y"] - b["y"]) / scale
                    if d < 3.0:           # beyond this it is a different person
                        jitter[name].append(d)
                break

    return {
        "observations": len(rows),
        "median_torso_scale_px": round(statistics.median(torso_scales), 1)
        if torso_scales else None,
        "angles": {k: summarise(v) for k, v in angles.items()},
        "confidence": {k: summarise(v) for k, v in confidence.items()},
        "observability": observability,
        "jitter_torso_normalised": {k: summarise(v) for k, v in jitter.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    configs = sorted(d.name for d in (args.run / "08_pose").iterdir()
                     if d.is_dir() and (d / "observations.jsonl").is_file())
    if not configs:
        print("no pose observations in this run", file=sys.stderr)
        return 2

    report = {}
    for config_id in configs:
        rows = read_jsonl(args.run / f"08_pose/{config_id}/observations.jsonl")
        report[config_id] = analyse(rows)

    (args.run / "08_pose/pose_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'configuration':28} {'obs':>5} {'head pitch':>18} "
          f"{'R elbow deg':>18} {'wrist conf':>14}")
    print("-" * 88)
    for config_id, r in report.items():
        hp = r["angles"]["head_pitch_norm"]
        el = r["angles"]["right_elbow_deg"]
        wc = r["confidence"]["right_wrist"]
        print(f"{config_id:28} {r['observations']:5} "
              f"{(f'{hp[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]:+.3f} [{hp[chr(112)+chr(49)+chr(48)]:+.2f},{hp[chr(112)+chr(57)+chr(48)]:+.2f}]' if hp else 'none'):>18} "
              f"{(f'{el[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]:.1f} +/-{el[chr(105)+chr(113)+chr(114)]/2:.1f}' if el else 'none'):>18} "
              f"{(f'{wc[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]:.3f}' if wc else 'none'):>14}")

    print(f"\n{'configuration':28} {'wrist jitter (torso-normalised)':>34} "
          f"{'elbow jitter':>16}")
    print("-" * 82)
    for config_id, r in report.items():
        wj = r["jitter_torso_normalised"]["right_wrist"]
        ej = r["jitter_torso_normalised"]["right_elbow"]
        print(f"{config_id:28} "
              f"{(f'median {wj[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]:.3f}  p90 {wj[chr(112)+chr(57)+chr(48)]:.3f}  n={wj[chr(110)]}' if wj else 'no paired frames'):>34} "
              f"{(f'{ej[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]:.3f}' if ej else '--'):>16}")

    print("\nJitter is frame-to-frame joint displacement divided by shoulder "
          "width, so it is comparable between a near and a distant candidate. "
          "Lower is steadier. A model with high visibility and high jitter is "
          "returning a keypoint every frame without agreeing with itself.")
    print(f"\nwritten {args.run / '08_pose/pose_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
