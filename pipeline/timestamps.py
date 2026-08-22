"""Timestamp truth (PRD 6.1).

Timing comes from **decoded presentation timestamps**, never from a container's
declared frame rate. That is not a stylistic preference on this corpus: **four
of six** source files are variable frame rate, and two of them declare a rate
that is simply wrong. File 03 declares 12.1 fps against a measured 15.0; the
Seat 12 file declares 25.0 against a measured 21.9. A label derived from
`frame_index / declared_fps` drifts against the events it is meant to locate,
silently, and worst on exactly the files that most need locating.

Four conditions are detected here rather than downstream, because they are
properties of the timestamp sequence itself:

  * **Variable frame rate** — the spread of inter-frame intervals.
  * **Repeated PTS** — two frames claiming the same instant. Ordering between
    them is undefined, so an event boundary landing there is ambiguous.
  * **Gaps** — an interval far longer than the local median, meaning frames are
    missing from the stream. PRD 6.2 requires these kept visible as
    `decode_gap`, never bridged.
  * **Non-monotonic PTS** — a timestamp that goes backwards. Rare, and fatal to
    any assumption that a scan proceeds forward in time.

The index is built from a single decode pass and holds one row per frame, so
every later artifact can map itself back to a source frame and timestamp.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

# A gap is an interval this many times the local median before it counts as
# missing data rather than variable-rate jitter. VFR files legitimately vary by
# a factor of two or three between adjacent frames; measured on this corpus, the
# largest legitimate ratio inside a VFR file is under 4. Above 8 the frames are
# genuinely absent.
GAP_RATIO = 8.0

# Below this, inter-frame intervals are effectively constant and the stream is
# constant frame rate. Expressed as the coefficient of variation of the
# intervals: standard deviation over mean.
#
# Measured across the whole corpus, this separates cleanly with a very wide
# margin -- the two populations differ by a factor of 150 and nothing sits
# between them:
#
#     file                       cv       distinct intervals (ms)
#     01 mobile phone         0.0000     40.0 x3282                      CFR
#     02 mobile phone         0.0007     40.0 x5300, 41.0 x2, 39.0 x2    CFR
#     04 candidate talking    0.1061     120.0 x1001, 160.0 x144         VFR
#     05 reception crowd      0.2041     33.3 x3611, 50.0 x2408          VFR
#     Seat 12 paper           0.3131     40.0 x1666, 80.0 x265, ...      VFR
#     03 mobile usage         0.3571     83.0 x1409, 84.0 x1409, 33.0 x1408  VFR
#
# **This corrects an earlier measurement.** The previous implementation used the
# standard deviation of PTS deltas and classified file 04 as constant rate. It
# is not: 144 of its frames are held for 160 ms against a 120 ms majority, a
# third longer, in two exactly discrete durations. Four of six corpus files are
# variable rate, not three.
VFR_CV_THRESHOLD = 0.05


@dataclass
class FrameRecord:
    """One decoded frame, as the source presented it."""

    index: int                  # decode order, zero based
    pts_ms: float               # presentation timestamp in milliseconds
    keyframe: bool
    pict_type: str = ""
    # Set when this frame's timestamp is unusable in some way, so that a later
    # stage can see the problem rather than inheriting it.
    repeated_pts: bool = False
    backwards_pts: bool = False


@dataclass
class Gap:
    """A stretch where frames are missing from the stream."""

    after_index: int
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class TimestampIndex:
    """Every frame of one source, plus what is wrong with the sequence."""

    frames: list[FrameRecord] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    decode_errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ measures --

    @property
    def count(self) -> int:
        return len(self.frames)

    @property
    def first_pts_ms(self) -> float:
        return self.frames[0].pts_ms if self.frames else 0.0

    @property
    def last_pts_ms(self) -> float:
        return self.frames[-1].pts_ms if self.frames else 0.0

    @property
    def span_ms(self) -> float:
        """Wall-clock span from the first presented frame to the last.

        Note this is the span, not the duration: the final frame is displayed
        for some interval after its timestamp, which the span does not include.
        `duration_ms` adds it back using the median interval.
        """
        return self.last_pts_ms - self.first_pts_ms

    @property
    def duration_ms(self) -> float:
        if self.count < 2:
            return 0.0
        return self.span_ms + self.median_interval_ms

    @property
    def intervals_ms(self) -> list[float]:
        return [b.pts_ms - a.pts_ms for a, b in zip(self.frames, self.frames[1:])]

    @property
    def median_interval_ms(self) -> float:
        intervals = [i for i in self.intervals_ms if i > 0]
        return statistics.median(intervals) if intervals else 0.0

    @property
    def measured_fps(self) -> float:
        """Frames per second from counted frames over measured duration.

        The only frame rate this system trusts. Compare it to the container's
        declared rate to see whether the declaration can be believed.
        """
        if self.duration_ms <= 0:
            return 0.0
        return self.count / (self.duration_ms / 1000.0)

    @property
    def interval_cv(self) -> float:
        """Coefficient of variation of inter-frame intervals.

        Gap intervals are excluded: a missing stretch is absent data, not
        variable pacing, and including it would report every file with a dropped
        frame as variable rate.
        """
        intervals = [i for i in self.intervals_ms if i > 0]
        if len(intervals) < 2:
            return 0.0
        median = statistics.median(intervals)
        clean = [i for i in intervals if i <= median * GAP_RATIO]
        if len(clean) < 2:
            return 0.0
        mean = statistics.fmean(clean)
        if mean <= 0:
            return 0.0
        return statistics.pstdev(clean) / mean

    @property
    def is_vfr(self) -> bool:
        return self.interval_cv > VFR_CV_THRESHOLD

    @property
    def repeated_pts_count(self) -> int:
        return sum(1 for f in self.frames if f.repeated_pts)

    @property
    def backwards_pts_count(self) -> int:
        return sum(1 for f in self.frames if f.backwards_pts)

    @property
    def keyframe_count(self) -> int:
        return sum(1 for f in self.frames if f.keyframe)

    @property
    def gap_ms(self) -> float:
        return sum(g.duration_ms for g in self.gaps)

    @property
    def decodable_ms(self) -> float:
        """Duration actually covered by decoded frames.

        PRD 18 requires the complete decodable duration to be processed or every
        missing range explicitly failed. This is the denominator of that claim.
        """
        return max(self.duration_ms - self.gap_ms, 0.0)

    # ------------------------------------------------------------ lookups --

    def nearest(self, pts_ms: float) -> FrameRecord | None:
        """The decoded frame closest to a timestamp.

        Used to map an artifact back to a source frame. Returns None on an empty
        index rather than inventing a frame, because an artifact that cannot
        name its source frame must not claim one.
        """
        if not self.frames:
            return None
        return min(self.frames, key=lambda f: abs(f.pts_ms - pts_ms))

    def inside_gap(self, pts_ms: float) -> Gap | None:
        for gap in self.gaps:
            if gap.start_ms < pts_ms < gap.end_ms:
                return gap
        return None

    def spans_gap(self, start_ms: float, end_ms: float) -> list[Gap]:
        """Gaps overlapping an interval.

        PRD 8 forbids bridging an event across a decode gap, and this is how a
        segmenter asks whether it is about to.
        """
        return [g for g in self.gaps
                if g.start_ms < end_ms and g.end_ms > start_ms]

    def to_rows(self) -> list[dict]:
        return [
            {"index": f.index, "pts_ms": round(f.pts_ms, 3), "keyframe": f.keyframe,
             "pict_type": f.pict_type, "repeated_pts": f.repeated_pts,
             "backwards_pts": f.backwards_pts}
            for f in self.frames
        ]

    def summary(self) -> dict:
        return {
            "frames": self.count,
            "first_pts_ms": round(self.first_pts_ms, 3),
            "last_pts_ms": round(self.last_pts_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "decodable_ms": round(self.decodable_ms, 3),
            "measured_fps": round(self.measured_fps, 4),
            "median_interval_ms": round(self.median_interval_ms, 4),
            "interval_cv": round(self.interval_cv, 5),
            "is_vfr": self.is_vfr,
            "keyframes": self.keyframe_count,
            "repeated_pts": self.repeated_pts_count,
            "backwards_pts": self.backwards_pts_count,
            "gaps": len(self.gaps),
            "gap_ms": round(self.gap_ms, 3),
            "decode_errors": len(self.decode_errors),
        }


def build_index(records: list[tuple[float, bool, str]],
                gap_ratio: float = GAP_RATIO) -> TimestampIndex:
    """Assemble an index from `(pts_ms, keyframe, pict_type)` in decode order.

    Separated from decoding so that the sequence analysis can be tested against
    constructed timestamp patterns without producing a video file for each one.
    """
    index = TimestampIndex()
    previous_pts: float | None = None
    seen: set[float] = set()

    for i, (pts_ms, keyframe, pict_type) in enumerate(records):
        frame = FrameRecord(index=i, pts_ms=pts_ms, keyframe=bool(keyframe),
                            pict_type=pict_type)
        if pts_ms in seen:
            frame.repeated_pts = True
        seen.add(pts_ms)
        if previous_pts is not None and pts_ms < previous_pts:
            frame.backwards_pts = True
        previous_pts = pts_ms
        index.frames.append(frame)

    # Gaps are found against the median interval, which needs the whole sequence
    # first -- hence a second pass rather than doing it inline above.
    median = index.median_interval_ms
    if median > 0:
        threshold = median * gap_ratio
        for a, b in zip(index.frames, index.frames[1:]):
            delta = b.pts_ms - a.pts_ms
            if delta > threshold:
                # The gap starts one nominal interval after the last good frame:
                # that frame is displayed for its interval before anything is
                # actually missing.
                index.gaps.append(
                    Gap(after_index=a.index,
                        start_ms=a.pts_ms + median,
                        end_ms=b.pts_ms))
    return index
