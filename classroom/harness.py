"""Long-form test recording synthesised from real footage (PRD 17.1).

The corpus is pre-cut incident clips: every frame is an event, so there is no
negative time and event recall cannot be measured. This builds the missing
denominator.

A base frame is held for long quiet stretches. At chosen timestamps one seat
region is cross-faded to the *same region from a different real timestamp*, so
the motion is genuine -- the candidate really did move between those two
moments -- while the rest of the scene stays still. The result is a recording of
arbitrary length with real people, real camera geometry, real compression, and
exactly known event times.

This is not a substitute for continuous footage. It is what makes event recall,
temporal IoU and false-events-per-hour measurable before that footage exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass
class PlantedEvent:
    """One synthesised activity burst."""

    seat_label: str
    region: tuple[int, int, int, int]   # x1, y1, x2, y2 in frame coordinates
    t_start: float
    t_end: float
    source_index: int                   # which alternate frame to fade toward

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def _feathered_mask(shape: tuple[int, int], region: tuple[int, int, int, int],
                    feather: int = 12) -> np.ndarray:
    """Soft-edged region mask, so a composite does not create a hard rectangle.

    A hard edge would be a step change in every block along the boundary and
    would register as motion in its own right -- an artefact of the harness
    rather than of the scene.
    """
    x1, y1, x2, y2 = region
    mask = np.zeros(shape[:2], dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    k = feather * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), feather / 2.0)[..., None]


def build(
    out_path: str | Path,
    base_frame: np.ndarray,
    alternates: list[np.ndarray],
    events: list[PlantedEvent],
    duration_s: float,
    fps: int = 25,
    codec: str = "libx264",
    bitrate: int = 1_900_000,
    oscillations: float = 3.0,
    sensor_noise: float = 2.0,
    # With the triangle wave below, |velocity| is constant at
    # shift_px * (2/pi) * 2*pi*oscillations / (duration*fps). At 36 px over a
    # 6 s event at 25 fps with three oscillations that is about 2.9 px/frame,
    # comfortably clear of the 2.0 px/frame block threshold for the whole swing
    # rather than only at its fastest instant.
    shift_px: float = 36.0,
    seed: int = 17,
) -> Path:
    """Render the recording.

    `bitrate` defaults to the corpus median (~1.9 Mbps at 720p), so the
    compression artefacts the motion gate has to tolerate are representative
    rather than absent.

    `oscillations` sets how many times the composite swings in and out across
    an event. A single slow cross-fade is the wrong shape: spread over six
    seconds it changes each frame by well under the codec's noise floor, and the
    motion gate correctly sees nothing. Real hand movement is fast and
    intermittent, so the alpha oscillates instead of ramping once.

    `sensor_noise` adds mild per-frame grain. Compositing onto a held still
    frame produces an unnaturally perfect background -- every block exactly
    static -- which gives the per-seat MAD nothing to measure and leaves the
    robust z-score dominated by its own floor. Real footage always carries a
    noise floor, and the gate has to work above it rather than in its absence.

    `shift_px` translates the composited content inside its window, and it is
    the difference between exercising the whole motion path and exercising half
    of it. A cross-fade changes pixel *values* without moving anything, so the
    encoder codes it as residual and emits almost no motion vectors. Measured on
    the harness before this existed: during planted events the mean area ratio
    was 0.0000-0.0008 -- essentially no block cleared the 2 px/frame threshold --
    while the residual rose 1.8x to 4.7x. Real events on real footage run at an
    area ratio of 0.15-0.69.

    The consequence was invisible and serious. The harness validated the
    statistical path (residual -> robust z) and never the kinematic one
    (motion vectors -> moving blocks), so a guard keyed on moving blocks could
    be added, be correct on real footage, and take the harness's recall from 5/5
    to 0/5 -- which is exactly what happened when `min_area_ratio` arrived. The
    composite now translates as well as fades, so a planted event displaces
    pixels the way a moving person does.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = base_frame.shape[:2]
    masks = {
        (e.seat_label, e.region): _feathered_mask(base_frame.shape, e.region)
        for e in events
    }

    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = bitrate
        stream.gop_size = fps * 2

        for index in range(int(duration_s * fps)):
            t = index / fps
            frame = base_frame.astype(np.float32)

            for event in events:
                if not (event.t_start <= t < event.t_end):
                    continue
                # Oscillate rather than ramp once: several fast swings across
                # the event, enveloped so it starts and ends at the base frame.
                phase = (t - event.t_start) / max(event.duration, 1e-6)
                envelope = float(np.sin(np.pi * phase))
                swing = 0.5 * (1.0 - float(np.cos(2 * np.pi * oscillations * phase)))
                alpha = envelope * swing
                alt = alternates[event.source_index].astype(np.float32)
                if shift_px > 0:
                    # A triangle wave, not a sine. A sinusoidal displacement has
                    # zero velocity at each turning point and peaks a quarter
                    # cycle away from the alpha peak, so the frames where the
                    # composite is most visible are the frames where nothing is
                    # moving -- which produced a spiky signal of two to four
                    # qualifying windows per event and no sustained detection at
                    # all. A triangle wave holds |velocity| constant across the
                    # whole swing. Diagonal, because a purely horizontal shift is
                    # the case block matching handles best and would flatter the
                    # gate.
                    travel = shift_px * envelope * (2.0 / np.pi) * float(
                        np.arcsin(np.sin(2 * np.pi * oscillations * phase)))
                    matrix = np.float32([[1, 0, travel], [0, 1, travel * 0.4]])
                    alt = cv2.warpAffine(
                        alt, matrix, (width, height),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                mask = masks[(event.seat_label, event.region)] * alpha
                frame = frame * (1.0 - mask) + alt * mask

            if sensor_noise > 0:
                frame = frame + rng.normal(0.0, sensor_noise, frame.shape)

            img = np.clip(frame, 0, 255).astype(np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(img, format="bgr24")
            ):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return out_path


def ground_truth(events: list[PlantedEvent]) -> list[dict]:
    """Insertion timestamps, for scoring recall against."""
    return [
        {"seat_label": e.seat_label, "t_start": e.t_start, "t_end": e.t_end,
         "duration": e.duration}
        for e in sorted(events, key=lambda e: e.t_start)
    ]


def score_recall(
    detected: list[tuple[str, float, float]],
    truth: list[dict],
    tolerance_s: float = 2.0,
    ranks: list[float] | None = None,
    duration_s: float | None = None,
) -> dict:
    """Match detections to planted events and report the standard metrics.

    A detection matches when it overlaps the planted span for the same seat,
    within `tolerance_s`.

    `ranks` is the score or peak deviation per detection, in the same order.
    When supplied, Precision@K is reported for K equal to the number of planted
    events. That is the metric a reviewer actually experiences (PRD 7.4): the
    system ranks rather than eliminates, so a marginal false positive sitting
    below every true event costs nothing, while the same count of false
    positives ranked above them would be disqualifying.
    """
    overlaps: dict[int, list[int]] = {}   # truth index -> detection indices
    for t_index, event in enumerate(truth):
        for d_index, (seat, start, end) in enumerate(detected):
            if seat != event["seat_label"]:
                continue
            if start < event["t_end"] + tolerance_s and end > event["t_start"] - tolerance_s:
                overlaps.setdefault(t_index, []).append(d_index)

    matched_detections = {d for group in overlaps.values() for d in group}
    matched_truth = sorted(overlaps)

    ious = []
    for t_index in matched_truth:
        event = truth[t_index]
        # Best-overlapping detection defines the boundary quality.
        best = 0.0
        for d_index in overlaps[t_index]:
            _, start, end = detected[d_index]
            inter = max(0.0, min(end, event["t_end"]) - max(start, event["t_start"]))
            union = max(end, event["t_end"]) - min(start, event["t_start"])
            best = max(best, inter / union if union > 0 else 0.0)
        ious.append(best)

    spurious = len(detected) - len(matched_detections)

    # Fragmentation is detections *per matched truth event*, which is what the
    # word means. Counting every unmatched detection as fragmentation conflates
    # it with false positives, which have a different cause and a different fix.
    fragmentation = (
        len(matched_detections) / len(matched_truth) if matched_truth else 0.0
    )

    metrics = {
        "planted": len(truth),
        "detected": len(detected),
        "matched": len(matched_truth),
        "recall": len(matched_truth) / len(truth) if truth else 0.0,
        "precision": len(matched_detections) / len(detected) if detected else 0.0,
        "false_events": spurious,
        "mean_temporal_iou": float(np.mean(ious)) if ious else 0.0,
        "fragmentation": fragmentation,
    }

    if duration_s:
        metrics["false_events_per_hour"] = spurious / (duration_s / 3600.0)

    if ranks is not None:
        # Reported even when nothing was detected. Omitting the key made a
        # caller reading `metrics["precision_at_k"]` raise KeyError on a run
        # that found no events -- so a total-recall failure surfaced as a crash
        # in the reporting code rather than as the failing check it is.
        k = len(truth)
        order = sorted(range(len(detected)), key=lambda i: -ranks[i])[:k]
        hits = sum(1 for i in order if i in matched_detections)
        metrics["precision_at_k"] = hits / k if k else 0.0
        metrics["k"] = k

    return metrics
