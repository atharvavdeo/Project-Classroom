"""Interactive calibration authoring.

    python tools/calibrate.py VIDEO --at 60 --out calib/camera04.json

Click to place polygon points. Enter closes the current polygon and files it
under the active kind. The active kind is chosen with a single key:

    s  seat footprint (prompts for the placard label on the terminal)
    c  screen zone      -- excluded from motion scoring entirely
    d  desk zone        -- typing baseline lives here
    t  torso zone       -- head and shoulder evidence
    o  overlay mask     -- the burned-in date/time, which changes every second
    r  reflection mask  -- glass, mirrors, interior windows
    e  environment mask -- fans, curtains, anything persistently moving
    k  staff mask       -- invigilator paths and reception traffic

    l  set the desk line for the active seat by clicking one point
    u  undo the last point, or the last closed polygon if none pending
    w  write JSON and exit
    q  quit without writing

Zones attach to the most recently defined seat, so define a seat before its
zones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom.calibration import Calibration, MaskRegion, Seat  # noqa: E402

KIND_KEYS = {
    ord("s"): "seat",
    ord("c"): "screen",
    ord("d"): "desk",
    ord("t"): "torso",
    ord("o"): "overlay",
    ord("r"): "reflection",
    ord("e"): "environment",
    ord("k"): "staff",
}

COLOURS = {
    "seat": (255, 210, 60),
    "screen": (60, 60, 255),
    "desk": (80, 220, 120),
    "torso": (240, 150, 40),
    "overlay": (0, 0, 220),
    "reflection": (200, 60, 220),
    "environment": (120, 120, 120),
    "staff": (60, 200, 240),
}


def grab_frame(video: Path, at_s: float) -> np.ndarray:
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if at_s > 0:
            container.seek(int(at_s / float(stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            return frame.to_ndarray(format="bgr24")
    raise RuntimeError(f"could not decode a frame from {video}")


class Editor:
    def __init__(self, frame: np.ndarray, camera_id: int):
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.kind = "seat"
        self.pending: list[tuple[float, float]] = []
        self.seats: list[Seat] = []
        self.masks: list[MaskRegion] = []
        self.camera_id = camera_id
        self.awaiting_desk_line = False
        self.status = "kind=seat"

    # ------------------------------------------------------------- events --

    def on_mouse(self, event, x, y, flags, param):  # noqa: ANN001, ARG002
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.awaiting_desk_line:
            if self.seats:
                self.seats[-1].desk_line_y = float(y)
                self.status = f"desk line for seat {self.seats[-1].label} at y={y}"
            self.awaiting_desk_line = False
            return
        self.pending.append((float(x), float(y)))

    def close_polygon(self) -> None:
        if len(self.pending) < 3:
            self.status = "need at least 3 points"
            return
        polygon = list(self.pending)
        self.pending = []

        if self.kind == "seat":
            label = input("seat placard label: ").strip()
            if not label:
                self.status = "seat discarded, no label"
                return
            self.seats.append(Seat(label=label, polygon=polygon))
            self.status = f"seat {label} added"
        elif self.kind in ("screen", "desk", "torso"):
            if not self.seats:
                self.status = "define a seat before its zones"
                return
            self.seats[-1].zones[self.kind] = polygon
            self.status = f"{self.kind} zone -> seat {self.seats[-1].label}"
        else:
            self.masks.append(MaskRegion(kind=self.kind, polygon=polygon))
            self.status = f"{self.kind} mask added"

    def undo(self) -> None:
        if self.pending:
            self.pending.pop()
            self.status = "point removed"
        elif self.masks:
            removed = self.masks.pop()
            self.status = f"{removed.kind} mask removed"
        elif self.seats:
            removed = self.seats.pop()
            self.status = f"seat {removed.label} removed"

    # -------------------------------------------------------------- render --

    def render(self) -> np.ndarray:
        canvas = self.frame.copy()
        overlay = self.frame.copy()

        for mask in self.masks:
            pts = np.asarray(mask.polygon, np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], COLOURS[mask.kind])
        for seat in self.seats:
            for kind, polygon in seat.zones.items():
                pts = np.asarray(polygon, np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], COLOURS[kind])
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        for seat in self.seats:
            pts = np.asarray(seat.polygon, np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, COLOURS["seat"], 2)
            cx, cy = np.asarray(seat.polygon, np.int32).mean(axis=0).astype(int)
            cv2.putText(canvas, seat.label, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, COLOURS["seat"], 2, cv2.LINE_AA)
            if seat.desk_line_y is not None:
                y = int(seat.desk_line_y)
                xs = np.asarray(seat.polygon, np.int32)[:, 0]
                cv2.line(canvas, (int(xs.min()), y), (int(xs.max()), y),
                         (255, 255, 255), 1, cv2.LINE_AA)

        for i, point in enumerate(self.pending):
            cv2.circle(canvas, (int(point[0]), int(point[1])), 4,
                       COLOURS[self.kind], -1)
            if i:
                prev = self.pending[i - 1]
                cv2.line(canvas, (int(prev[0]), int(prev[1])),
                         (int(point[0]), int(point[1])), COLOURS[self.kind], 1)

        bar = (
            f"kind={self.kind}  seats={len(self.seats)}  masks={len(self.masks)}  "
            f"| {self.status}"
        )
        cv2.rectangle(canvas, (0, 0), (self.w, 26), (20, 20, 20), -1)
        cv2.putText(canvas, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
        return canvas

    def to_calibration(self) -> Calibration:
        cal = Calibration(
            camera_id=self.camera_id,
            frame_w=self.w,
            frame_h=self.h,
            seats=self.seats,
            masks=self.masks,
        )
        cal.validate()
        return cal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--at", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--camera-id", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frame = grab_frame(Path(args.video), args.at)
    editor = Editor(frame, args.camera_id)

    window = "calibrate"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(frame.shape[1], 1600), min(frame.shape[0], 900))
    cv2.setMouseCallback(window, editor.on_mouse)
    print(__doc__)

    while True:
        cv2.imshow(window, editor.render())
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key == ord("q"):
            print("quit without writing")
            cv2.destroyAllWindows()
            return 1
        if key == ord("w"):
            break
        if key in (13, 10):
            editor.close_polygon()
        elif key == ord("u"):
            editor.undo()
        elif key == ord("l"):
            editor.awaiting_desk_line = True
            editor.status = "click the desk line"
        elif key in KIND_KEYS:
            editor.kind = KIND_KEYS[key]
            editor.pending = []
            editor.status = f"kind={editor.kind}"

    cv2.destroyAllWindows()
    try:
        cal = editor.to_calibration()
    except ValueError as exc:
        print(f"not written: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    cal.write_json(out)
    print(f"wrote {out}  seats={len(cal.seats)}  masks={len(cal.masks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
