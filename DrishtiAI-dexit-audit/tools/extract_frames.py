"""Extract evenly spaced frames from every source file, by PTS."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import cv2

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/DrishtiAI")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/frames")
PER_FILE = int(sys.argv[3]) if len(sys.argv) > 3 else 8

SHORT = {
    "01.Candidate": "cam04a", "02.Candidate": "cam04b", "03.CCTV": "cam12",
    "04.CCTV": "talk", "05.Crowd": "crowd", "Seat No. 12": "ahmd3",
}


def short_name(name: str) -> str:
    for key, tag in SHORT.items():
        if name.startswith(key):
            return tag
    return name[:8].replace(" ", "_")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(list(CORPUS.glob("*.mkv")) + list(CORPUS.glob("*.mp4")))
    total = 0
    for path in files:
        tag = short_name(path.name)
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            tb = float(st.time_base)
            dur = float(st.duration * tb) if st.duration else 0.0
            if dur <= 0:
                dur = float(c.duration / av.time_base) if c.duration else 60.0
            times = [dur * (i + 0.5) / PER_FILE for i in range(PER_FILE)]
            for t in times:
                c.seek(max(int(t / tb), 0), stream=st)
                for frame in c.decode(st):
                    if frame.pts is None:
                        continue
                    if frame.pts * tb >= t:
                        img = frame.to_ndarray(format="bgr24")
                        out = OUT / f"{tag}_t{int(t):04d}.jpg"
                        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
                        total += 1
                        break
        print(f"  {tag:8s} {path.name[:44]:44s} {PER_FILE} frames  {dur:6.1f}s")
    print(f"\n{total} frames -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
