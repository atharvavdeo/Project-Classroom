"""Temporal ground-truth annotation.

    python tools/annotate_timeline.py VIDEO --db data/classroom.db --annotator hs

Produces the two things evaluation cannot run without: labelled actions, and a
record of which footage was reviewed exhaustively enough for their absence to
mean something.

Every time written here comes from the frame's presentation timestamp, never
from a frame index divided by a nominal frame rate. Three of the six corpus
files are variable frame rate with declared rates that are simply wrong (PRD
5.1), so an index-derived label would drift against the events it is meant to
score -- silently, and worst on exactly the files that most need labelling.

Keys
----
    space       play / pause
    <- ->       step one frame
    , .         jump 5 s
    [ ]         jump 30 s
    home        back to the start

    s           start a review span here
    e           end the review span here (this is the recall denominator)

    i           mark the start of an action
    o           mark its end, then describe it at the terminal
    x           cancel a half-marked action

    u           undo the last thing recorded
    w           write everything to the database and exit
    q           quit without writing

Describing an action
--------------------
After `o` the terminal asks for one line: seat, type, visibility.

    19 1 c      seat 19, object handling, clearly visible
    61 3 p      seat 61, sustained lookaway, partially visible
    . 6 c       seat unknown, benign movement, clearly visible

Types      1 object_handling  2 neighbour_interaction  3 sustained_lookaway
           4 leaving_seat     5 staff_interaction      6 benign_movement
           7 unclear
Visibility c clear  p partial  o occluded  i insufficient
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import db  # noqa: E402
from classroom.annotate import (  # noqa: E402
    EVENT_TYPES, VISIBILITY, Annotation, Span, save_annotation, save_span,
)

TYPE_KEYS = {str(i + 1): name for i, name in enumerate(EVENT_TYPES)}
VIS_KEYS = {name[0]: name for name in VISIBILITY}

WINDOW = "annotate"

# Arrow and Home key codes differ per HighGUI backend, and the difference is not
# cosmetic: masking the Win32 codes to a byte collapses left and right onto the
# same value (2424832 & 0xFFFF == 2555904 & 0xFFFF == 0), so stepping backward
# and forward become the same key. Every known code is listed, and `waitKeyEx`
# is used so none of them is truncated on the way in. The letter aliases work
# everywhere regardless.
LEFT = {81, 0x250000, 2424832, 63234, ord("a")}
RIGHT = {83, 0x270000, 2555904, 63235, ord("d")}
HOME = {80, 0x240000, 2359296, 63273}


class Scrubber:
    """PTS-accurate random access over one video.

    Seeking is keyframe-granular, so every seek lands on or before the target
    and then decodes forward to it. Slower than trusting the seek, and correct:
    on VFR footage the two differ by enough to mislabel a two-second action.
    """

    def __init__(self, path: Path):
        self.path = path
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.time_base = float(self.stream.time_base)
        self.duration = float(
            (self.stream.duration or 0) * self.time_base
        ) or float(self.container.duration or 0) / av.time_base
        self._iter = self.container.decode(self.stream)
        self.t = 0.0
        self.frame: np.ndarray | None = None
        self.advance()
        self.head_unsafe_s = self._probe_head()

    def _pts_seconds(self, frame) -> float:
        if frame.pts is None:
            return self.t
        return float(frame.pts * self.time_base)

    def advance(self) -> bool:
        """Decode the next frame in presentation order."""
        try:
            frame = next(self._iter)
        except StopIteration:
            return False
        self.t = self._pts_seconds(frame)
        self.frame = frame.to_ndarray(format="bgr24")
        return True

    def _probe_head(self) -> float:
        """Find the earliest timestamp a backward seek can actually reach.

        Seeking backward to a timestamp at or before the first keyframe does not
        land on the first frame: the demuxer skips forward to the *next*
        keyframe instead. Measured on this corpus -- seek(0) lands at 0.483 s on
        `03.CCTV Mobile Usage.mkv` and at 1.040 s on the Seat 12 file, skipping
        seven and eighteen frames. Nothing raises. The annotator is simply shown
        a frame from a second later than the one being labelled, and every mark
        made in that region is wrong by that much.

        Rather than guess at a tolerance, the unreachable head is measured once,
        here, and any later seek below it decodes from the start instead. That
        region is at the beginning of the file, so the forward scan is short.
        """
        self.container.seek(0, stream=self.stream, backward=True)
        for frame in self.container.decode(self.stream):
            landed = self._pts_seconds(frame)
            break
        else:
            landed = 0.0
        self._reopen()
        return landed

    def _reopen(self) -> None:
        self.container.close()
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self._iter = self.container.decode(self.stream)

    def _decode_forward_to(self, target_s: float) -> bool:
        """Decode from the current iterator until the target is reached."""
        for frame in self._iter:
            t = self._pts_seconds(frame)
            if t >= target_s - 1e-6:
                self.t = t
                self.frame = frame.to_ndarray(format="bgr24")
                return True
        return False

    def seek(self, target_s: float) -> None:
        """Land on the first frame at or after `target_s`.

        Below the measured unreachable head the file is decoded from the start;
        above it a normal backward seek is exact. Failing to reach the target at
        all leaves the current frame on screen rather than presenting a blank
        one.
        """
        target_s = max(0.0, min(target_s, max(self.duration - 0.05, 0.0)))
        if target_s < self.head_unsafe_s:
            self._reopen()
            self._decode_forward_to(target_s)
            return
        self.container.seek(
            int(target_s / self.time_base), stream=self.stream, backward=True
        )
        self._iter = self.container.decode(self.stream)
        self._decode_forward_to(target_s)

    def close(self) -> None:
        self.container.close()


def prompt_action(t_start: float, t_end: float) -> tuple[str | None, str, str] | None:
    """Read one description line from the terminal."""
    print(f"\n  action {t_start:8.3f} -> {t_end:8.3f} s  ({t_end - t_start:.2f} s)")
    raw = input("  seat type visibility (blank to discard): ").strip()
    if not raw:
        print("  discarded")
        return None
    parts = raw.split()
    if len(parts) != 3:
        print("  need exactly three fields, e.g. '19 1 c' -- discarded")
        return None
    seat, type_key, vis_key = parts
    if type_key not in TYPE_KEYS:
        print(f"  unknown type {type_key!r} -- discarded")
        return None
    if vis_key not in VIS_KEYS:
        print(f"  unknown visibility {vis_key!r} -- discarded")
        return None
    seat_label = None if seat in (".", "-", "?") else seat
    print(f"  recorded: seat {seat_label or 'unknown'} "
          f"{TYPE_KEYS[type_key]} / {VIS_KEYS[vis_key]}")
    return seat_label, TYPE_KEYS[type_key], VIS_KEYS[vis_key]


def draw_hud(
    frame: np.ndarray, t: float, duration: float, playing: bool,
    pending_in: float | None, span_open: float | None,
    n_actions: int, n_spans: int, covered: float,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    bar = np.zeros((78, w, 3), dtype=np.uint8)

    left = f"{t:8.3f} / {duration:.1f} s   {'PLAY' if playing else 'PAUSE'}"
    cv2.putText(bar, left, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    state = []
    if pending_in is not None:
        state.append(f"action open @ {pending_in:.3f}")
    if span_open is not None:
        state.append(f"span open @ {span_open:.3f}")
    if state:
        cv2.putText(bar, "   ".join(state), (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (60, 220, 255), 1, cv2.LINE_AA)

    tally = f"actions {n_actions}   spans {n_spans}   reviewed {covered:.1f} s"
    cv2.putText(bar, tally, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (150, 255, 150), 1, cv2.LINE_AA)

    # Progress line. A span in progress is drawn so it is obvious when review
    # coverage is running, because an unclosed span is the one mistake here
    # that quietly invalidates every rate computed later.
    if duration > 0:
        x = int(w * min(t / duration, 1.0))
        cv2.line(bar, (0, 77), (w, 77), (60, 60, 60), 2)
        if span_open is not None:
            sx = int(w * min(span_open / duration, 1.0))
            cv2.line(bar, (sx, 77), (x, 77), (60, 220, 255), 2)
        cv2.line(bar, (x, 74), (x, 77), (255, 255, 255), 2)

    return np.vstack([canvas, bar])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--db", type=Path, default=Path("data/classroom.db"))
    ap.add_argument("--annotator", required=True,
                    help="who is reviewing; recorded on every row")
    ap.add_argument("--source-video-id", type=int,
                    help="row id in source_video; looked up by path if omitted")
    ap.add_argument("--start", type=float, default=0.0)
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 2

    conn = db.connect(args.db)
    video_id = args.source_video_id
    if video_id is None:
        row = db.one(
            conn, "SELECT id FROM source_video WHERE path = ?", (str(args.video),)
        )
        if row is None:
            row = db.one(
                conn,
                "SELECT id FROM source_video WHERE path LIKE ?",
                (f"%{args.video.name}",),
            )
        if row is None:
            print(f"{args.video.name} is not registered in {args.db}. "
                  f"Ingest it first, or pass --source-video-id.", file=sys.stderr)
            return 2
        video_id = int(row["id"])

    scrub = Scrubber(args.video)
    if args.start > 0:
        scrub.seek(args.start)

    actions: list[Annotation] = []
    spans: list[Span] = []
    history: list[str] = []
    pending_in: float | None = None
    span_open: float | None = None
    playing = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(__doc__)
    print(f"video id {video_id}, duration {scrub.duration:.1f} s")

    while True:
        if scrub.frame is None:
            break
        covered = sum(s.duration for s in spans)
        view = draw_hud(scrub.frame, scrub.t, scrub.duration, playing,
                        pending_in, span_open, len(actions), len(spans), covered)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKeyEx(20 if playing else 0)
        if key == -1:
            if playing and not scrub.advance():
                playing = False
            continue

        if key in (ord("q"), 27):
            print("quit without writing")
            scrub.close()
            cv2.destroyAllWindows()
            return 1

        if key == ord(" "):
            playing = not playing
        elif key in RIGHT:
            scrub.advance()
        elif key in LEFT:
            scrub.seek(max(scrub.t - 0.2, 0.0))
        elif key == ord(","):
            scrub.seek(scrub.t - 5.0)
        elif key == ord("."):
            scrub.seek(scrub.t + 5.0)
        elif key == ord("["):
            scrub.seek(scrub.t - 30.0)
        elif key == ord("]"):
            scrub.seek(scrub.t + 30.0)
        elif key in HOME:
            scrub.seek(0.0)

        elif key == ord("s"):
            if span_open is not None:
                print(f"  span already open at {span_open:.3f} s")
            else:
                span_open = scrub.t
                print(f"  span opened at {span_open:.3f} s")
        elif key == ord("e"):
            if span_open is None:
                print("  no span open")
            elif scrub.t <= span_open:
                print("  a span cannot end before it starts")
            else:
                spans.append(Span(video_id, span_open, scrub.t, args.annotator))
                history.append("span")
                print(f"  span closed: {span_open:.3f} -> {scrub.t:.3f} s "
                      f"({scrub.t - span_open:.1f} s reviewed)")
                span_open = None

        elif key == ord("i"):
            pending_in = scrub.t
            print(f"  action opened at {pending_in:.3f} s")
        elif key == ord("x"):
            if pending_in is not None:
                print("  action cancelled")
            pending_in = None
        elif key == ord("o"):
            if pending_in is None:
                print("  mark the start with 'i' first")
            elif scrub.t <= pending_in:
                print("  an action cannot end before it starts")
            else:
                described = prompt_action(pending_in, scrub.t)
                if described is not None:
                    seat_label, event_type, visibility = described
                    actions.append(Annotation(
                        video_id, pending_in, scrub.t, event_type, visibility,
                        seat_label, args.annotator,
                    ))
                    history.append("action")
                pending_in = None

        elif key == ord("u"):
            if not history:
                print("  nothing to undo")
            else:
                what = history.pop()
                if what == "action":
                    dropped = actions.pop()
                    print(f"  undid action {dropped.event_type} "
                          f"{dropped.t_start:.3f}-{dropped.t_end:.3f}")
                else:
                    dropped_span = spans.pop()
                    print(f"  undid span {dropped_span.t_start:.3f}-"
                          f"{dropped_span.t_end:.3f}")

        elif key == ord("w"):
            if span_open is not None:
                print("  a review span is still open. Close it with 'e' or "
                      "discard it with 'u' before writing.")
                continue
            if pending_in is not None:
                print("  an action is still half-marked. Close it with 'o' or "
                      "cancel it with 'x' before writing.")
                continue
            for span in spans:
                save_span(conn, span)
            for action in actions:
                save_annotation(conn, action)
            reviewed = sum(s.duration for s in spans)
            print(f"\nwrote {len(spans)} spans ({reviewed:.1f} s reviewed) "
                  f"and {len(actions)} actions to {args.db}")
            scrub.close()
            cv2.destroyAllWindows()
            return 0

    scrub.close()
    cv2.destroyAllWindows()
    print("reached the end of the video without writing; press 'w' to save")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
