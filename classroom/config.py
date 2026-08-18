"""Pipeline configuration.

One canonical value per setting. Where a threshold is empirical it carries the
measurement or reasoning that produced it, so it can be re-derived rather than
guessed at when the footage changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class MotionConfig:
    """Tier-1 codec motion-vector scan."""

    # Analysis window. 0.5 s at 25 fps = 12-13 frames: long enough for a robust
    # median, short enough to localise a 2 s gesture.
    window_s: float = 0.5

    # A 16x16 block must move more than this (pixels/frame) to count as moving.
    # At 0.08 bits/pixel the compressor emits small spurious vectors constantly,
    # and this is the threshold that separates real motion from them. Measured
    # peak moving-block counts, active seat versus idle seat in the same frame:
    #
    #     threshold   active   idle
    #        0.25        57      47
    #        0.50        51      42
    #        1.00        44      42
    #        2.00        40       4      <- separation
    #
    # Below 2.0 the compressor's noise floor swamps the signal and the two seats
    # become indistinguishable. Raising it further would start discarding slow
    # deliberate movement, which sits around 1-2 px/frame at this scale.
    block_mag_threshold: float = 2.0

    # Global motion: if this fraction of unmasked blocks moves coherently, the
    # camera moved rather than the scene.
    global_motion_ratio: float = 0.55
    global_motion_mag: float = 1.5

    # Exposure step detection, mean-luma delta between adjacent windows.
    exposure_step_delta: float = 6.0

    # Luma is sampled every Nth frame; full-frame conversion is the dominant
    # per-frame cost and the signal is slow-moving.
    luma_sample_stride: int = 5


@dataclass(frozen=True)
class BaselineConfig:
    """Per-seat activity baseline learned during calibration."""

    # Duration of footage analysed to learn a seat's regimes.
    learn_s: float = 180.0

    # A seat is 'typing' in a window when desk-zone motion exceeds the idle
    # centre by this many MAD. Below it, 'idle'.
    typing_z: float = 2.0

    # A typing regime is only established when the active windows make up at
    # least this fraction of the learning span. Typing in a CBT hall is
    # near-continuous, so a seat that is genuinely working sits above the cut
    # most of the time. Below this fraction the excursions are *events*, and
    # treating them as a regime would train the baseline to ignore exactly what
    # the system exists to find.
    min_regime_fraction: float = 0.25

    # Guards against a degenerate baseline on a seat that was empty or static
    # for the whole learning span.
    min_windows: int = 60

    # The spread floor is *relative* to the seat's own median, not absolute.
    #
    # This matters more than it looks. Residual magnitudes are scene-dependent:
    # on real 720p CCTV a seat's median residual is around 0.01 with a MAD near
    # 0.0016, while a high-contrast synthetic fixture produces values two orders
    # of magnitude larger. An absolute floor tuned on one is catastrophic on the
    # other -- a floor of 0.05 against a true MAD of 0.0016 divides every
    # z-score by 30 and the gate goes blind, detecting nothing at all.
    #
    # Found exactly that way: fixtures passed, real footage produced zero events
    # with peak z of 1.5 where the true value was ~47.
    min_mad_fraction: float = 0.05

    # Absolute floor, and it carries real weight rather than merely avoiding a
    # division by zero. This is the other half of the lesson above: on a seat
    # that is genuinely static both the median and the MAD approach zero, so a
    # purely *relative* floor also collapses, and a trivial fluctuation divided
    # by a near-zero denominator reports an enormous deviation from nothing at
    # all. Measured: a static seat produced z = 16.6 with zero moving blocks.
    #
    # 0.001 is set from the observed scale. Real seats sit at a median residual
    # of 0.011-0.021 with a MAD of 0.0016-0.0025, so the relative term governs
    # them and this floor never binds. A static seat has a median near zero and
    # peak residuals around 0.002, which this floor correctly keeps below the
    # start threshold.
    min_mad_absolute: float = 0.001

    # Minimum blocks in a zone that must exceed the motion threshold before any
    # activity is reported, regardless of the statistics. An absolute count, not
    # a fraction, so it does not shift with zone size.
    min_moving_blocks: int = 1

    # And a *fraction* of the zone, which the absolute count above cannot
    # substitute for. A large zone accumulates stray blocks in proportion to its
    # size, so one moving block means something quite different in a 520-block
    # seat than in a 660-block one -- and the count guard passes both.
    #
    # Found on a full run of `03.CCTV Mobile Usage.mkv`. An unoccupied desk row
    # produced the twelve highest-scoring events in the recording, peaking at
    # z = 29.8, while the one genuinely occupied seat peaked at 9.5. The region
    # is visually static: monitors off, nobody in frame, nothing moving across
    # any sampled peak. Measured over the whole recording:
    #
    #     seat     median resid    MAD      mean area ratio   max z
    #     R-desk      0.0130     0.0081        0.0011          29.8
    #     S-61        0.1835     0.1616        0.0465           9.5
    #
    # The empty seat carries forty times less motion and scores three times
    # higher, because a near-zero MAD in the denominator turns a single block of
    # compression noise into an enormous standardised value. Neither existing
    # floor engages: the seat's own MAD (0.0081) sits well above both
    # min_mad_absolute and min_mad_fraction * median.
    #
    # Across every window above the start threshold, the two seats separate
    # completely with no overlap:
    #
    #     R-desk  n=142   area ratio  median 0.0033   max 0.0333
    #     S-61    n= 39   area ratio  median 0.2625   min 0.1458
    #
    # A 4.4x gap. At 0.05 all 142 spurious windows are removed and all 39
    # genuine ones survive, with wide margin on both sides.
    min_area_ratio: float = 0.05


@dataclass(frozen=True)
class SegmentConfig:
    """Per-seat hysteresis state machine (PRD 7.3)."""

    # Robust z-score against the seat's current regime. Start is deliberately
    # higher than stop: this is what stops one action fragmenting into many.
    start_z: float = 4.0
    stop_z: float = 2.0

    # A candidate must persist this long to become Active.
    min_duration_s: float = 1.2

    # How long a forming candidate may sit below stop_z before it is discarded.
    # Real hand motion is intermittent -- a reach pauses, the hand rests, the
    # motion resumes -- so the raw signal dips to baseline several times inside
    # one action. Without this the machine leaves CANDIDATE on the first dip and
    # a genuine action never reaches ACTIVE.
    #
    # This deliberately bridges *gaps between activity* rather than smoothing
    # the signal: smoothing would manufacture duration from a single spike and
    # defeat impulse rejection, which is what keeps a dropped pen from becoming
    # an incident.
    candidate_bridge_s: float = 1.5

    # Cooling: activity resuming inside this gap re-enters Active.
    merge_gap_s: float = 2.5

    # Forces a split so a long noisy stretch cannot become one unusable event.
    max_duration_s: float = 90.0

    # Clip extraction padding.
    pre_roll_s: float = 3.0
    post_roll_s: float = 3.0


@dataclass(frozen=True)
class ScoreConfig:
    """Hand-set priority weights (PRD 7.4).

    Not learned: the available event count cannot support fitting these without
    overfitting. Every weight is surfaced per-event in the UI.
    """

    w_motion: float = 1.0
    w_persistence: float = 0.8
    w_below_desk: float = 1.2
    w_head_pitch: float = 1.0
    w_object: float = 0.9
    w_cross_camera: float = 0.5
    w_global_motion: float = -1.5
    w_environmental: float = -1.0
    w_uncertainty: float = -0.7


@dataclass(frozen=True)
class DetectConfig:
    """Phase 6. Native-resolution seat crops upscaled into the detector."""

    input_size: int = 640
    # Crops smaller than this are upscaled; larger ones are letterboxed down.
    min_crop_px: int = 96
    confidence: float = 0.35

    # Abstention floor. The phone in the profiled footage measures ~30x20 px
    # (600 px^2) at a near seat, which is already marginal; below 300 px^2 the
    # correct output is 'insufficient visual evidence', never a class label.
    min_object_px_area: float = 300.0

    # An object must appear in this many consecutive sampled frames to count.
    temporal_confirm_frames: int = 3


@dataclass(frozen=True)
class VerifyConfig:
    """Phase 7. Gemma verifier, GBNF-constrained."""

    model_path: str = "models/gemma/gemma-4-E4B_q4_0-it.gguf"
    mmproj_path: str = "models/gemma/gemma-4-E4B-it-mmproj.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1          # all layers; 32 GB target has ample room
    # Sized to the grammar's worst case, not guessed: 5 observations and a
    # review note, each capped at 200 characters, plus structure, is about
    # 1,200 characters or ~350 tokens. 768 leaves comfortable headroom, and
    # anything that still hits the limit is reported as a truncated
    # verification rather than parsed.
    max_tokens: int = 768
    temperature: float = 0.1
    contact_sheet_frames: int = 6


@dataclass(frozen=True)
class Config:
    db_path: Path = Path("data/classroom.db")
    media_root: Path = Path("data/media")
    motion: MotionConfig = field(default_factory=MotionConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config, overlaying a JSON file over the defaults if given."""
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sub = {
            "motion": MotionConfig,
            "baseline": BaselineConfig,
            "segment": SegmentConfig,
            "score": ScoreConfig,
            "detect": DetectConfig,
            "verify": VerifyConfig,
        }
        kwargs: dict = {}
        for key, value in data.items():
            if key in sub:
                kwargs[key] = sub[key](**value)
            elif key in ("db_path", "media_root"):
                kwargs[key] = Path(value)
            else:
                raise ValueError(f"unknown config key: {key}")
        return cls(**kwargs)

    def to_json(self) -> str:
        def default(o):
            if isinstance(o, Path):
                return str(o)
            raise TypeError(type(o))
        return json.dumps(asdict(self), indent=2, default=default)
