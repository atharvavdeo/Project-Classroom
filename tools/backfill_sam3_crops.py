"""Link the SAM 3 annotated crops a run already produced into the console.

Why this exists
---------------

The referee's annotated crops were being written and then stranded. Measured
across the corpus:

    run                  annotated on disk   linked in manifest
    1511/01_phone               30                  0
    1509/04_talking             48                  0
    1512/12_paper              213                 16

So the console showed "not adjudicated" in the SAM 3 column for rows whose own
Backing column said "10 of 224 frames" -- two statements about the same track
that flatly contradict each other. The adjudication had happened; nothing
carried the picture across.

`not adjudicated` has a specific meaning in this system: the referee was never
called. Printing it where the referee *was* called and its output is sitting on
disk is a false statement about provenance, and it is exactly the kind of
"absence of evidence read as evidence of absence" the output contract exists to
prevent.

What it does
------------

For every track in a console crop manifest, find the annotated crop for that
same track nearest in time to the track's key frame, copy it next to the other
review crops, and record its filename. Nothing is generated: if a track has no
annotated crop, it keeps `sam3_crop: null` and the console goes on saying "not
adjudicated", which for that track is true.

    python tools/backfill_sam3_crops.py                # every mapped run
    python tools/backfill_sam3_crops.py 1511_01_phone  # just one
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "console/public/crops"

#: console crop-set name -> run directory, relative to artifacts/runs.
RUNS = {
    "1505_03_mobile": "1505/03_mobile",
    "1507_06_phone": "1507/06_phone_last10",
    "1509_04_talking": "1509/04_talking",
    "1511_01_phone": "1511/01_phone",
    "1512_12_paper": "1512/12_paper",
}

#: `52750ms_trk1.jpg` -> (52750.0, 1)
NAME = re.compile(r"^(\d+)ms_trk(\d+)\.jpg$", re.I)

#: A crop further from the key frame than this is a different moment, and
#: showing it under the key frame's claim would misattribute the evidence.
#: 12s is generous but bounded -- the sampler runs at 2 Hz, so a gap this large
#: means the referee simply was not called near this instant.
MAX_GAP_MS = 12_000.0


def index(annotated: Path) -> dict[int, list[tuple[float, str]]]:
    """Annotated crops per track, sorted by time."""
    by_track: dict[int, list[tuple[float, str]]] = {}
    if not annotated.is_dir():
        return by_track
    for f in annotated.iterdir():
        m = NAME.match(f.name)
        if not m:
            continue
        by_track.setdefault(int(m.group(2)), []).append((float(m.group(1)), f.name))
    for rows in by_track.values():
        rows.sort()
    return by_track


def backfill(name: str, run_rel: str) -> None:
    manifest_path = CROPS / name / "manifest.json"
    annotated = ROOT / "artifacts/runs" / run_rel / "14_person_timeline/sam3_adjudication"

    if not manifest_path.exists():
        print(f"{name:20} no manifest; skipped")
        return

    by_track = index(annotated)
    if not by_track:
        print(f"{name:20} no annotated crops in {annotated.relative_to(ROOT)}")
        return

    # utf-8-sig: some of these were written by a Windows tool that prepends a
    # BOM, and json.load refuses it.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    out_dir = CROPS / name / "sam3"
    out_dir.mkdir(parents=True, exist_ok=True)

    linked = already = missing = far = 0
    for key, entry in manifest.items():
        if entry.get("sam3_crop"):
            already += 1
            continue
        track = int(entry.get("track_id", key))
        rows = by_track.get(track)
        if not rows:
            missing += 1
            continue
        target = float(entry.get("key_pts_ms") or 0.0)
        pts, filename = min(rows, key=lambda r: abs(r[0] - target))
        if abs(pts - target) > MAX_GAP_MS:
            far += 1
            continue
        shutil.copyfile(annotated / filename, out_dir / filename)
        entry["sam3_crop"] = filename
        # What the picture is *of*, so a reviewer is never guessing whether the
        # crop belongs to the row above it.
        entry["sam3_crop_pts_ms"] = pts
        linked += 1

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"{name:20} linked {linked:3}  already {already:3}  "
          f"no-crop {missing:3}  too-far {far:3}  "
          f"(from {sum(len(v) for v in by_track.values())} annotated)")


def main() -> int:
    wanted = sys.argv[1:] or list(RUNS)
    for name in wanted:
        run_rel = RUNS.get(name)
        if not run_rel:
            print(f"{name}: not a known crop set; known: {', '.join(RUNS)}")
            continue
        backfill(name, run_rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
