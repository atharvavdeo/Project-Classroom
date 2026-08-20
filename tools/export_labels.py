"""Export the label registry as JSON for the frontend.

    python tools/export_labels.py --out artifacts/labels.json

One source of truth. A frontend that hardcodes its own strings drifts from the
pipeline the first time a class is renamed, and the drift is silent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/labels.json")
    args = ap.parse_args()

    registry = labels.registry()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    total = sum(len(v) for v in registry["groups"].values())
    print(f"{total} labels in {len(registry['groups'])} groups")
    for name, group in registry["groups"].items():
        by_sev = {}
        for row in group:
            by_sev[row["severity"]] = by_sev.get(row["severity"], 0) + 1
        print(f"  {name:16}{len(group):4}  {by_sev}")
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
