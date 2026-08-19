"""Render the actual frames behind detection rows (PRD 10, PRD 17.12).

    python tools/render_detections.py --run artifacts/runs/full \
        --config dfine_native --class secondary_paper_chit --top 12 VIDEO

PRD 17.12 requires that "every reported item needs PTS-native overlay/crop
provenance", and PRD 10 requires the crop itself to be persisted. Detection
rows carry source-frame boxes and PTS but no pixels, so until now a detection
could be counted but not *looked at*. This closes that: it re-decodes the exact
source frames and draws the exact boxes the detector produced.

Two panels per detection, because either alone misleads:

  * the **full frame** with the box drawn, so the reader sees where in the room
    it is and what surrounds it;
  * the **native crop** beside a magnified view, captioned with its true source
    footprint in pixels. Magnification is interpolation, and a crop shown only
    at 8x invites the reader to see detail that is not in the source.

Nothing here re-runs a model or changes a score. It reads persisted rows and
shows the pixels they refer to.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_at(video: Path, wanted_ms: list[float]) -> dict:
    """Decode exactly the frames named, in one forward pass."""
    import av

    wanted = sorted(set(round(t, 1) for t in wanted_ms))
    found, index = {}, 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None or index >= len(wanted):
                continue
            pts_ms = float(frame.pts * time_base) * 1000.0
            while index < len(wanted) and pts_ms >= wanted[index] - 1.0:
                found[wanted[index]] = frame.to_ndarray(format="bgr24")
                index += 1
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--config", default="dfine_native")
    ap.add_argument("--class", dest="cls", default="secondary_paper_chit")
    ap.add_argument("--top", type=int, default=12,
                    help="highest-confidence N detections of that class")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    rows = read_jsonl(
        paths.dir / f"09_object/{args.config}/merged_detections.jsonl")
    hits = [r for r in rows if r["cls"] == args.cls and not r.get("abstained")]
    if not hits:
        print(f"no {args.cls!r} detections in {args.config}", file=sys.stderr)
        return 2

    hits.sort(key=lambda r: -r["confidence"])
    hits = hits[:args.top]
    print(f"{len(rows)} rows, {args.cls}: showing top {len(hits)} by confidence "
          f"({hits[0]['confidence']:.3f} to {hits[-1]['confidence']:.3f})")

    import cv2
    import numpy as np

    frames = decode_at(args.video, [r["pts_ms"] for r in hits])
    out_dir = args.out or (paths.dir / f"09_object/{args.config}/rendered_{args.cls}")
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    for rank, row in enumerate(hits, start=1):
        frame = frames.get(round(row["pts_ms"], 1))
        if frame is None:
            print(f"  rank {rank}: no frame decoded at {row['pts_ms']:.0f} ms")
            continue
        height, width = frame.shape[:2]
        x1, y1 = int(max(row["x1"], 0)), int(max(row["y1"], 0))
        x2, y2 = int(min(row["x2"], width)), int(min(row["y2"], height))
        if x2 <= x1 or y2 <= y1:
            continue

        native_w, native_h = x2 - x1, y2 - y1

        marked = frame.copy()
        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked, f"{row['cls']} {row['confidence']:.2f}",
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)

        # Context crop around the box, then the native box magnified.
        pad = max(native_w, native_h) * 2
        cx1, cy1 = max(int(x1 - pad), 0), max(int(y1 - pad), 0)
        cx2, cy2 = min(int(x2 + pad), width), min(int(y2 + pad), height)
        context = marked[cy1:cy2, cx1:cx2]
        native = frame[y1:y2, x1:x2]

        panel_h = 260
        def fit(image, target_h):
            scale = target_h / max(image.shape[0], 1)
            return cv2.resize(image, (max(int(image.shape[1] * scale), 1), target_h),
                              interpolation=cv2.INTER_NEAREST if scale > 1
                              else cv2.INTER_AREA)

        strip = np.zeros((30, 900, 3), dtype=np.uint8)
        cv2.putText(strip,
                    f"#{rank}  {row['cls']}  conf {row['confidence']:.3f}  "
                    f"{row['pts_ms'] / 1000.0:.2f}s  native {native_w}x{native_h}px "
                    f"({args.config})",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        pieces = [fit(context, panel_h), fit(native, panel_h)]
        row_img = np.hstack(pieces)
        if row_img.shape[1] < 900:
            pad_img = np.zeros((panel_h, 900 - row_img.shape[1], 3), dtype=np.uint8)
            row_img = np.hstack([row_img, pad_img])
        else:
            row_img = row_img[:, :900]
        panels.append(np.vstack([strip, row_img]))

        prefix = artifacts.pts_prefix(row["pts_ms"])
        cv2.imwrite(str(out_dir / f"{prefix}_r{rank:02d}_frame.jpg"), marked)
        cv2.imwrite(str(out_dir / f"{prefix}_r{rank:02d}_native.png"), native)

    if panels:
        sheet = np.vstack(panels)
        sheet_path = out_dir / f"contact_sheet_{args.cls}.jpg"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"\nwritten {sheet_path}")
        print(f"        {len(panels)} full frames + native crops in {out_dir}")
        print("\nLeft panel is context with the box drawn; right panel is the "
              "native crop at its true source size, upscaled with nearest "
              "neighbour so no interpolation invents detail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
