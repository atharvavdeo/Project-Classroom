"""Video-health observations (PRD 6.2).

Ten named conditions, each with detection evidence and a required action. The
actions matter as much as the detections: PRD 8's gate protocol is explicit that
gates route evidence rather than erase it, so nothing here deletes a span. A
condition marks footage, changes what may be learned from it, and travels with
every downstream record drawn from that time range.

Three principles run through all of them.

**Nothing is bridged.** A decode gap, a blackout or an uncertain camera movement
is a hole in what is known, and an event that spans one is two events with a
hole between them. PRD 8 forbids bridging across any of these boundaries.

**Baselines never learn through bad quality.** PRD 6.3.9 requires baseline
learning to exclude quality-affected spans and to record which windows were
included and excluded. A baseline that absorbs a blackout learns that darkness
is normal, and then fails to notice the next one.

**Thresholds are relative to the camera wherever the quantity is
scene-dependent.** Sharpness varies by an order of magnitude between a well-lit
near seat and a distant one, so blur is defined against a per-camera baseline
(PRD 6.2). An absolute sharpness number calibrated on one camera means nothing
on another -- the same lesson the head-pitch baseline taught on real footage,
where every upright candidate scored between -0.35 and -2.3 because the camera
looked down.

Seven of these ten conditions have **no example anywhere in the available
corpus**. They are built to specification and exercised by fixtures that inject
the condition into real frames. A fixture proves the detector fires; it does not
prove the threshold is right for real footage. Every observation therefore
carries `validated_on` so a report can distinguish the two, and the two
conditions that do occur here -- `compression_noise` and `environmental_motion`
-- are marked as measured on source.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

# ------------------------------------------------------------- conditions --

BLACKOUT = "blackout"
WHITEOUT = "whiteout"
FROZEN_VIDEO = "frozen_video"
DECODE_GAP = "decode_gap"
BLUR = "blur"
EXPOSURE_CHANGE = "exposure_change"
CAMERA_MOTION = "camera_motion"
CAMERA_REPOSITIONED = "camera_repositioned"
COMPRESSION_NOISE = "compression_noise"
ENVIRONMENTAL_MOTION = "environmental_motion"

CONDITIONS = (BLACKOUT, WHITEOUT, FROZEN_VIDEO, DECODE_GAP, BLUR,
              EXPOSURE_CHANGE, CAMERA_MOTION, CAMERA_REPOSITIONED,
              COMPRESSION_NOISE, ENVIRONMENTAL_MOTION)

# The required action per PRD 6.2, carried on the observation so that a
# downstream stage reads the instruction rather than reimplementing the table.
RETAIN_UNAVAILABLE = "retain_mark_unavailable"     # keep, do not learn baseline
PAUSE_BASELINE = "pause_baseline_adaptation"
PREVENT_FALSE_MOTION = "prevent_false_motion"
KEEP_GAP_VISIBLE = "keep_gap_visible"
REDUCE_MODEL_RELIABILITY = "reduce_object_pose_reliability"
FLAG_RECOVERY_NO_BASELINE = "flag_recovery_no_baseline_update"
COMPENSATE_OR_NO_ATTRIBUTION = "compensate_on_success_else_no_local_attribution"
NEW_EPOCH_REVIEW = "new_camera_epoch_calibration_review"
RETAIN_DEPRIORITISE = "retain_diagnostic_deprioritise"
RETAIN_PENALISE = "retain_named_observation_penalise_ranking"

ACTION = {
    BLACKOUT: RETAIN_UNAVAILABLE,
    WHITEOUT: PAUSE_BASELINE,
    FROZEN_VIDEO: PREVENT_FALSE_MOTION,
    DECODE_GAP: KEEP_GAP_VISIBLE,
    BLUR: REDUCE_MODEL_RELIABILITY,
    EXPOSURE_CHANGE: FLAG_RECOVERY_NO_BASELINE,
    CAMERA_MOTION: COMPENSATE_OR_NO_ATTRIBUTION,
    CAMERA_REPOSITIONED: NEW_EPOCH_REVIEW,
    COMPRESSION_NOISE: RETAIN_DEPRIORITISE,
    ENVIRONMENTAL_MOTION: RETAIN_PENALISE,
}

# Conditions that make a span ineligible for baseline learning (PRD 6.3.9).
EXCLUDES_FROM_BASELINE = frozenset({
    BLACKOUT, WHITEOUT, FROZEN_VIDEO, DECODE_GAP, EXPOSURE_CHANGE,
    CAMERA_MOTION, CAMERA_REPOSITIONED,
})

# Conditions across which an event may never be bridged (PRD 8).
BLOCKS_BRIDGING = frozenset({
    DECODE_GAP, BLACKOUT, CAMERA_MOTION, CAMERA_REPOSITIONED,
})

VALIDATED_ON_SOURCE = "source"
VALIDATED_ON_FIXTURE = "fixture_only"


@dataclass
class HealthConfig:
    """Thresholds, all named so they can be versioned and hashed (PRD 17.10)."""

    # A blackout is a sustained near-black frame. 0.90 rather than 1.0 because a
    # real CCTV blackout is sensor floor plus noise plus a burned-in timestamp
    # overlay, which never goes fully black -- the overlay alone occupies about
    # 2% of the frame on every file in this corpus.
    blackout_near_black_fraction: float = 0.90
    blackout_min_duration_ms: float = 500.0

    whiteout_near_white_fraction: float = 0.90
    whiteout_min_duration_ms: float = 500.0

    # Freeze detection uses the mean absolute inter-frame difference of the
    # statistics thumbnail, on a 0-255 scale.
    #
    # A difference *hash* was tried first and was badly wrong: on a fixed mount
    # the frame is dominated by unchanging furniture, so consecutive hashes are
    # identical whether or not anyone moved, and it reported `frozen_video` over
    # 99.6% of a file with people walking through it.
    #
    # The threshold is set from measurement. Real footage always carries sensor
    # noise and compression variation, so a genuinely quiet frame never reaches
    # zero, while a duplicated frame does:
    #
    #     file                    min     p01     p05    median     max
    #     03 mobile usage       0.0043  0.0113  0.0226  0.1745   58.2665
    #     04 candidate talking  0.0417  0.0681  0.1020  0.4705    3.0872
    #
    #   frames below 0.005:  file 03 -> 1 of 4226,  file 04 -> 0 of 1145
    #   frames below 0.001:  neither file has any
    #
    # 0.005 sits below every real frame in one file and below all but one in the
    # other, and that single frame cannot form a run long enough to qualify.
    frozen_max_frame_delta: float = 0.005
    frozen_min_duration_ms: float = 1000.0

    # Blur is relative to the camera's own median sharpness (PRD 6.2). A frame
    # under this fraction of the baseline is blurred *for this camera*.
    blur_relative_threshold: float = 0.40
    blur_min_duration_ms: float = 300.0

    # An exposure change is a scene-wide luma shift between consecutive frames.
    # Expressed in luma levels on a 0-255 scale.
    exposure_step_luma: float = 8.0
    # How long after a step the scene is still settling and must not be learned
    # from. Auto-exposure on CCTV hunts for roughly a second.
    exposure_recovery_ms: float = 2000.0

    # Compression noise is isolated, low-area, non-persistent activity. Both
    # numbers are the measured shape of the phenomenon on this corpus, where an
    # unoccupied desk row produced activity at 0.1% area while a genuinely
    # occupied seat ran at 15-69%.
    compression_noise_max_area: float = 0.05
    compression_noise_max_duration_ms: float = 500.0

    # Environmental motion is periodic or persistent. A fan, a curtain, a
    # monitor. Persistence is the fraction of the recording a region is active
    # for; anything a candidate does is far below this.
    environmental_min_persistence: float = 0.50


@dataclass
class HealthObservation:
    """One condition, over one PTS range, with the evidence that produced it."""

    condition: str
    start_pts_ms: float
    end_pts_ms: float
    evidence: dict = field(default_factory=dict)
    action: str = ""
    validated_on: str = VALIDATED_ON_FIXTURE
    region: tuple[float, float, float, float] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"unknown health condition {self.condition!r}; "
                f"expected one of {CONDITIONS}")
        if self.end_pts_ms < self.start_pts_ms:
            raise ValueError(
                f"{self.condition}: ends before it starts "
                f"({self.start_pts_ms} -> {self.end_pts_ms})")
        if not self.action:
            self.action = ACTION[self.condition]

    @property
    def duration_ms(self) -> float:
        return self.end_pts_ms - self.start_pts_ms

    @property
    def excludes_from_baseline(self) -> bool:
        return self.condition in EXCLUDES_FROM_BASELINE

    @property
    def blocks_bridging(self) -> bool:
        return self.condition in BLOCKS_BRIDGING

    def covers(self, pts_ms: float) -> bool:
        return self.start_pts_ms <= pts_ms <= self.end_pts_ms

    def overlaps(self, start_ms: float, end_ms: float) -> bool:
        return self.start_pts_ms < end_ms and self.end_pts_ms > start_ms

    def to_row(self) -> dict:
        return {
            "condition": self.condition,
            "start_pts_ms": round(self.start_pts_ms, 3),
            "end_pts_ms": round(self.end_pts_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "action": self.action,
            "validated_on": self.validated_on,
            "excludes_from_baseline": self.excludes_from_baseline,
            "blocks_bridging": self.blocks_bridging,
            "region": list(self.region) if self.region else None,
            "evidence": self.evidence,
            "note": self.note,
        }


# -------------------------------------------------------------- detection --

def _runs(flags: list[bool], times_ms: list[float],
          min_duration_ms: float) -> list[tuple[int, int, float, float]]:
    """Contiguous true runs lasting at least `min_duration_ms`.

    Returns `(start_index, end_index, start_ms, end_ms)`. A single frame above a
    threshold is not a condition; sustained evidence is what PRD 6.2 asks for on
    every one of these.
    """
    out = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if times_ms[i - 1] - times_ms[start] >= min_duration_ms:
                out.append((start, i - 1, times_ms[start], times_ms[i - 1]))
            start = None
    if start is not None and flags:
        last = len(flags) - 1
        if times_ms[last] - times_ms[start] >= min_duration_ms:
            out.append((start, last, times_ms[start], times_ms[last]))
    return out


def detect_blackout(stats, cfg: HealthConfig,
                    validated_on: str = VALIDATED_ON_FIXTURE) -> list[HealthObservation]:
    times = [s.pts_ms for s in stats]
    flags = [s.near_black_fraction >= cfg.blackout_near_black_fraction for s in stats]
    return [
        HealthObservation(
            BLACKOUT, t0, t1, validated_on=validated_on,
            evidence={"near_black_fraction_mean": round(statistics.fmean(
                [stats[i].near_black_fraction for i in range(a, b + 1)]), 4),
                "frames": b - a + 1},
        )
        for a, b, t0, t1 in _runs(flags, times, cfg.blackout_min_duration_ms)
    ]


def detect_whiteout(stats, cfg: HealthConfig,
                    validated_on: str = VALIDATED_ON_FIXTURE) -> list[HealthObservation]:
    times = [s.pts_ms for s in stats]
    flags = [s.near_white_fraction >= cfg.whiteout_near_white_fraction for s in stats]
    return [
        HealthObservation(
            WHITEOUT, t0, t1, validated_on=validated_on,
            evidence={"near_white_fraction_mean": round(statistics.fmean(
                [stats[i].near_white_fraction for i in range(a, b + 1)]), 4),
                "frames": b - a + 1},
        )
        for a, b, t0, t1 in _runs(flags, times, cfg.whiteout_min_duration_ms)
    ]


def hamming(a: int, b: int) -> int:
    """Bit distance between two scene fingerprints.

    Used for camera repositioning, where "is this a different view" is exactly
    the question. Not used for freezes -- see HealthConfig.
    """
    return bin(a ^ b).count("1")


def detect_frozen(stats, cfg: HealthConfig,
                  validated_on: str = VALIDATED_ON_FIXTURE) -> list[HealthObservation]:
    """Image repeats while PTS advances (PRD 6.2).

    The PTS advancing is the whole point: a still image with a still clock is a
    paused player, while a still image with a moving clock is a camera or
    encoder that has stopped producing new pictures. Motion scoring over that
    span would read exactly zero and be believed.
    """
    if len(stats) < 2:
        return []
    times = [s.pts_ms for s in stats]
    flags = [False] + [
        stats[i].frame_delta <= cfg.frozen_max_frame_delta
        and times[i] > times[i - 1]
        for i in range(1, len(stats))
    ]
    out = []
    for a, b, t0, t1 in _runs(flags, times, cfg.frozen_min_duration_ms):
        deltas = [stats[i].frame_delta for i in range(max(a, 1), b + 1)]
        out.append(HealthObservation(
            FROZEN_VIDEO, t0, t1, validated_on=validated_on,
            evidence={"max_frame_delta": round(max(deltas), 5) if deltas else 0.0,
                      "frames": b - a + 1},
            note="image repeats while PTS advances; motion here would read zero "
                 "and be believed",
        ))
    return out


def detect_decode_gaps(index, validated_on: str = VALIDATED_ON_FIXTURE
                       ) -> list[HealthObservation]:
    """Gaps come from the timestamp index; nothing image-based is involved."""
    return [
        HealthObservation(
            DECODE_GAP, g.start_ms, g.end_ms, validated_on=validated_on,
            evidence={"after_frame_index": g.after_index,
                      "gap_ms": round(g.duration_ms, 3)},
            note="frames are absent from the stream; no event may be bridged across it",
        )
        for g in index.gaps
    ]


def detect_blur(stats, cfg: HealthConfig, baseline_sharpness: float | None = None,
                validated_on: str = VALIDATED_ON_FIXTURE) -> list[HealthObservation]:
    """Sharpness relative to this camera's own baseline (PRD 6.2)."""
    if not stats:
        return []
    baseline = baseline_sharpness if baseline_sharpness is not None \
        else statistics.median([s.sharpness for s in stats])
    if baseline <= 0:
        return []
    cut = baseline * cfg.blur_relative_threshold
    times = [s.pts_ms for s in stats]
    flags = [s.sharpness < cut for s in stats]
    return [
        HealthObservation(
            BLUR, t0, t1, validated_on=validated_on,
            evidence={"baseline_sharpness": round(baseline, 3),
                      "threshold": round(cut, 3),
                      "min_sharpness": round(min(
                          stats[i].sharpness for i in range(a, b + 1)), 3),
                      "frames": b - a + 1},
        )
        for a, b, t0, t1 in _runs(flags, times, cfg.blur_min_duration_ms)
    ]


def detect_exposure_change(stats, cfg: HealthConfig,
                           validated_on: str = VALIDATED_ON_FIXTURE
                           ) -> list[HealthObservation]:
    """A scene-wide luma step, plus the recovery window that follows it.

    The recovery window is the part that matters for correctness. The step
    itself is one frame; what damages a baseline is the second or two afterwards
    while auto-exposure hunts, which looks like sustained scene change and is
    not.
    """
    out = []
    for i in range(1, len(stats)):
        delta = stats[i].luma_mean - stats[i - 1].luma_mean
        if abs(delta) >= cfg.exposure_step_luma:
            out.append(HealthObservation(
                EXPOSURE_CHANGE,
                stats[i].pts_ms,
                stats[i].pts_ms + cfg.exposure_recovery_ms,
                validated_on=validated_on,
                evidence={"luma_delta": round(delta, 3),
                          "from": round(stats[i - 1].luma_mean, 3),
                          "to": round(stats[i].luma_mean, 3),
                          "recovery_ms": cfg.exposure_recovery_ms},
                note="baseline adaptation is paused through the recovery window",
            ))
    return merge(out)


def merge(observations: list[HealthObservation], gap_ms: float = 0.0
          ) -> list[HealthObservation]:
    """Collapse overlapping observations of the same condition.

    Auto-exposure produces a step on many consecutive frames, and one
    observation per frame would bury the timeline in near-identical rows.
    Evidence from the merged rows is kept as a count rather than discarded.
    """
    by_condition: dict[str, list[HealthObservation]] = {}
    for observation in observations:
        by_condition.setdefault(observation.condition, []).append(observation)

    out: list[HealthObservation] = []
    for condition, group in by_condition.items():
        group.sort(key=lambda o: o.start_pts_ms)
        current = group[0]
        merged_count = 1
        for nxt in group[1:]:
            if nxt.start_pts_ms <= current.end_pts_ms + gap_ms:
                current = HealthObservation(
                    condition, current.start_pts_ms,
                    max(current.end_pts_ms, nxt.end_pts_ms),
                    evidence={**current.evidence},
                    validated_on=current.validated_on,
                    region=current.region, note=current.note)
                merged_count += 1
            else:
                current.evidence["merged_observations"] = merged_count
                out.append(current)
                current, merged_count = nxt, 1
        current.evidence["merged_observations"] = merged_count
        out.append(current)
    out.sort(key=lambda o: (o.start_pts_ms, o.condition))
    return out


def observe(ingest_result, cfg: HealthConfig | None = None,
            baseline_sharpness: float | None = None,
            validated_on: str = VALIDATED_ON_FIXTURE) -> list[HealthObservation]:
    """Every image- and timestamp-derived condition for one ingested video.

    `camera_motion`, `camera_repositioned`, `compression_noise` and
    `environmental_motion` are not produced here: the first two need the global
    alignment stage and the last two need per-zone motion, so they are raised by
    `alignment.py` and `motion.py` and merged into the same observation stream.
    """
    cfg = cfg or HealthConfig()
    stats = ingest_result.stats
    observations: list[HealthObservation] = []
    observations += detect_blackout(stats, cfg, validated_on)
    observations += detect_whiteout(stats, cfg, validated_on)
    observations += detect_frozen(stats, cfg, validated_on)
    observations += detect_blur(stats, cfg, baseline_sharpness, validated_on)
    observations += detect_exposure_change(stats, cfg, validated_on)
    observations += detect_decode_gaps(ingest_result.index, validated_on)
    return merge(observations)


# --------------------------------------------------------------- timeline --

class HealthTimeline:
    """Answers the questions later stages ask about a time range."""

    def __init__(self, observations: list[HealthObservation]):
        self.observations = sorted(observations, key=lambda o: o.start_pts_ms)

    def at(self, pts_ms: float) -> list[HealthObservation]:
        return [o for o in self.observations if o.covers(pts_ms)]

    def over(self, start_ms: float, end_ms: float) -> list[HealthObservation]:
        return [o for o in self.observations if o.overlaps(start_ms, end_ms)]

    def baseline_eligible(self, start_ms: float, end_ms: float) -> bool:
        """Whether a window may contribute to a learned baseline (PRD 6.3.9)."""
        return not any(o.excludes_from_baseline
                       for o in self.over(start_ms, end_ms))

    def may_bridge(self, start_ms: float, end_ms: float) -> bool:
        """Whether an event may span this interval (PRD 8)."""
        return not any(o.blocks_bridging for o in self.over(start_ms, end_ms))

    def affected_ms(self, condition: str | None = None) -> float:
        chosen = [o for o in self.observations
                  if condition is None or o.condition == condition]
        if not chosen:
            return 0.0
        # Union, not sum: two conditions over the same span affect that span
        # once, and summing would let affected time exceed the recording.
        intervals = sorted((o.start_pts_ms, o.end_pts_ms) for o in chosen)
        total, current_start, current_end = 0.0, *intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end
        return total + (current_end - current_start)

    def counts(self) -> dict[str, int]:
        out = {c: 0 for c in CONDITIONS}
        for observation in self.observations:
            out[observation.condition] += 1
        return out

    def summary(self, total_ms: float) -> dict:
        counts = self.counts()
        return {
            "observations": len(self.observations),
            "by_condition": {k: v for k, v in counts.items() if v},
            "affected_ms": round(self.affected_ms(), 3),
            "affected_fraction": round(self.affected_ms() / total_ms, 5)
            if total_ms > 0 else None,
            "baseline_excluded_ms": round(sum(
                o.duration_ms for o in self.observations
                if o.excludes_from_baseline), 3),
        }
