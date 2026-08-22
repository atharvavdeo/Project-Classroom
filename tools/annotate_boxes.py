"""Object-level ground truth for the detector fine-tune.

    python tools/annotate_boxes.py data/frames --db data/classroom.db --annotator hs

Draws boxes on extracted stills. Two things about this corpus shape the tool:

  * The objects are tiny. A phone measures roughly 30x20 px at a near seat
    (PRD 5.2), which is smaller than the mouse cursor on most displays. A loupe
    is therefore not a convenience -- without magnification the annotator is
    guessing, and guessed boxes are worse than no boxes.

  * The negatives matter more than the positives. Stock weights scored about
    40% precision here, and the errors were computer mice, keyboard edges and
    monitor bezels read as phones (PRD 8.1.2). Boxing those confusers by name is
    the highest-value labelling available, so they have their own class keys.

A frame that is opened and advanced past is recorded as reviewed even when it
yields nothing. That is deliberate: a reviewed empty frame is a background
training example, and without the record the exporter cannot tell "nothing here"
from "not looked at yet".

Keys
----
    drag        draw a box
    1-7         classify the box just drawn (see the legend in the window)
    u           undo the last box on this frame
    c           clear every box on this frame

    n / space   next frame (records this one as reviewed)
    p           previous frame
    g           go to a frame number, typed at the terminal

    +  -        loupe magnification
    h           hide / show existing boxes
    w           write everything to the database and exit
    q           quit without writing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import db  # noqa: E402
from classroom.annotate import (  # noqa: E402
    BOX_CLASSES, NEGATIVE_CLASSES, Box, clear_boxes, dataset_summary,
    mark_frame_reviewed, save_box,
)

CLASS_KEYS = {str(i + 1): name for i, name in enumerate(BOX_CLASSES)}

COLOURS = {
    "phone_like": (60, 60, 255),
    "paper_like": (80, 220, 120),
    "person": (255, 210, 60),
    "mouse": (200, 60, 220),
    "keyboard": (240, 150, 40),
    "monitor": (120, 120, 120),
    "other": (200, 200, 200),
}

WINDOW = "boxes"

# Below this the box is a stray click rather than an object. The abstention
# floor in the detector is 300 px^2, and a label smaller than a tenth of that
# cannot be a deliberate annotation of anything in 720p footage.
MIN_BOX_PX = 30.0


class Canvas:
    """Mouse state for one frame."""

    def __init__(self) -> None:
        self.dragging = False
        self.origin = (0, 0)
        self.cursor = (0, 0)
        self.rubber: tuple[int, int, int, int] | None = None
        self.pending: tuple[int, int, int, int] | None = None

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.origin = (x, y)
            self.rubber = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            ox, oy = self.origin
            self.rubber = (min(ox, x), min(oy, y), max(ox, x), max(oy, y))
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            ox, oy = self.origin
            box = (min(ox, x), min(oy, y), max(ox, x), max(oy, y))
            self.rubber = None
            width, height = box[2] - box[0], box[3] - box[1]
            if width * height >= MIN_BOX_PX:
                self.pending = box
            else:
                print(f"  box of {width}x{height} px ignored as a stray click")


def draw_loupe(
    frame: np.ndarray, view: np.ndarray, cursor: tuple[int, int], zoom: int
) -> None:
    """Paint a magnified inset of the region under the cursor.

    Drawn from the *source* frame, not from the annotated view, so existing
    boxes never obscure the pixels being judged.
    """
    h, w = frame.shape[:2]
    size = 72
    cx, cy = cursor
    x1, y1 = max(cx - size // 2, 0), max(cy - size // 2, 0)
    x2, y2 = min(x1 + size, w), min(y1 + size, h)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return
    big = cv2.resize(patch, (patch.shape[1] * zoom, patch.shape[0] * zoom),
                     interpolation=cv2.INTER_NEAREST)
    bh, bw = big.shape[:2]
    # Park the loupe in the corner furthest from the cursor so it never covers
    # what is being looked at.
    px = 8 if cx > w // 2 else w - bw - 8
    py = 8 if cy > h // 2 else h - bh - 8
    px, py = max(px, 0), max(py, 0)
    if py + bh > view.shape[0] or px + bw > view.shape[1]:
        return
    view[py:py + bh, px:px + bw] = big
    cv2.rectangle(view, (px, py), (px + bw, py + bh), (255, 255, 255), 1)
    # Crosshair at the magnified cursor position.
    mx = px + (cx - x1) * zoom
    my = py + (cy - y1) * zoom
    cv2.line(view, (mx - 8, my), (mx + 8, my), (0, 255, 255), 1)
    cv2.line(view, (mx, my - 8), (mx, my + 8), (0, 255, 255), 1)


def render(
    frame: np.ndarray, boxes: list[Box], canvas: Canvas, zoom: int,
    index: int, total: int, name: str, show_boxes: bool, reviewed: bool,
) -> np.ndarray:
    view = frame.copy()
    if show_boxes:
        for box in boxes:
            colour = COLOURS[box.cls]
            cv2.rectangle(view, (int(box.x1), int(box.y1)),
                          (int(box.x2), int(box.y2)), colour, 1)
            label = box.cls + (" -" if box.cls in NEGATIVE_CLASSES else "")
            cv2.putText(view, label, (int(box.x1), max(int(box.y1) - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    if canvas.rubber:
        x1, y1, x2, y2 = canvas.rubber
        cv2.rectangle(view, (x1, y1), (x2, y2), (255, 255, 255), 1)
    if canvas.pending:
        x1, y1, x2, y2 = canvas.pending
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)

    draw_loupe(frame, view, canvas.cursor, zoom)

    bar = np.zeros((80, view.shape[1], 3), dtype=np.uint8)
    head = (f"[{index + 1}/{total}] {name}   boxes {len(boxes)}   "
            f"loupe {zoom}x   {'reviewed' if reviewed else 'new'}")
    cv2.putText(bar, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if canvas.pending:
        cv2.putText(bar, "press 1-7 to classify the highlighted box",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
    x = 10
    for key, name_ in CLASS_KEYS.items():
        text = f"{key} {name_}"
        cv2.putText(bar, text, (x, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLOURS[name_], 1, cv2.LINE_AA)
        x += 12 * len(text) - 20
    return np.vstack([view, bar])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, help="directory of extracted stills")
    ap.add_argument("--db", type=Path, default=Path("data/classroom.db"))
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(args.frames.glob(args.glob))
    if not paths:
        print(f"no frames matching {args.glob} under {args.frames}", file=sys.stderr)
        return 2

    conn = db.connect(args.db)
    already = {
        r["frame_path"] for r in
        db.all_rows(conn, "SELECT frame_path FROM reviewed_frame")
    }

    # Boxes are held per frame in memory and written in one transaction at the
    # end, so quitting with 'q' really does leave the database untouched.
    per_frame: dict[str, list[Box]] = {}
    for row in db.all_rows(conn, "SELECT * FROM box_annotation"):
        per_frame.setdefault(row["frame_path"], []).append(
            Box(row["frame_path"], row["cls"], row["x1"], row["y1"],
                row["x2"], row["y2"], row["source_video_id"], row["t"],
                row["seat_label"], row["visibility"], row["annotator"])
        )
    touched: set[str] = set()

    canvas = Canvas()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, canvas.on_mouse)

    index = max(0, min(args.start, len(paths) - 1))
    zoom = 6
    show_boxes = True
    print(__doc__)

    # The display loop redraws about fifty times a second so the loupe tracks
    # the cursor. Decoding the JPEG on every one of those passes is pure waste;
    # one frame is held instead and replaced only when the index moves.
    loaded: tuple[str, np.ndarray] | None = None

    while True:
        path = paths[index]
        key_path = str(path)
        if loaded is None or loaded[0] != key_path:
            image = cv2.imread(key_path)
            loaded = (key_path, image) if image is not None else None
        frame = loaded[1] if loaded else None
        if frame is None:
            print(f"  cannot read {path}, skipping")
            index = min(index + 1, len(paths) - 1)
            continue
        boxes = per_frame.setdefault(key_path, [])

        view = render(frame, boxes, canvas, zoom, index, len(paths),
                      path.name, show_boxes, key_path in already)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFFFF
        if key == 0xFFFF:
            continue

        char = chr(key) if 32 <= key < 127 else ""

        if char in CLASS_KEYS:
            if canvas.pending is None:
                print("  draw a box first")
            else:
                x1, y1, x2, y2 = canvas.pending
                box = Box(key_path, CLASS_KEYS[char], float(x1), float(y1),
                          float(x2), float(y2), annotator=args.annotator)
                box.validate()
                boxes.append(box)
                touched.add(key_path)
                canvas.pending = None
                print(f"  {box.cls} {int(box.px_area)} px^2")
            continue

        if char in ("q", "\x1b"):
            print("quit without writing")
            cv2.destroyAllWindows()
            return 1

        if char in ("n", " "):
            already.add(key_path)
            touched.add(key_path)
            canvas.pending = None
            if index + 1 >= len(paths):
                print("  at the last frame; press 'w' to write")
            index = min(index + 1, len(paths) - 1)
        elif char == "p":
            canvas.pending = None
            index = max(index - 1, 0)
        elif char == "g":
            raw = input(f"  frame number 1-{len(paths)}: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(paths):
                index = int(raw) - 1
            else:
                print("  out of range")
        elif char == "u":
            if boxes:
                dropped = boxes.pop()
                touched.add(key_path)
                print(f"  removed {dropped.cls}")
            else:
                print("  nothing to undo")
        elif char == "c":
            if boxes:
                print(f"  cleared {len(boxes)} boxes")
                boxes.clear()
                touched.add(key_path)
        elif char == "h":
            show_boxes = not show_boxes
        elif char == "+" or char == "=":
            zoom = min(zoom + 1, 12)
        elif char == "-":
            zoom = max(zoom - 1, 2)
        elif char == "w":
            written = 0
            for frame_path in sorted(touched):
                clear_boxes(conn, frame_path)
                for box in per_frame.get(frame_path, []):
                    save_box(conn, box)
                    written += 1
                mark_frame_reviewed(conn, frame_path, annotator=args.annotator)
            all_boxes = [b for bs in per_frame.values() for b in bs]
            print(f"\nwrote {written} boxes across {len(touched)} frames "
                  f"to {args.db}\n")
            print(dataset_summary(all_boxes, sorted(already | touched)))
            cv2.destroyAllWindows()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
