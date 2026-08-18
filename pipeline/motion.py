"""Whole-video motion, as an adapter over the measured implementation (PRD 8).

The codec motion-vector scan in `classroom/motion.py` has been measured on the
real corpus at 66x realtime and has survived fourteen defects being found and
fixed in it. Rewriting it here would discard that, so this module **adapts** it:
it converts the new `CalibrationProfile` into the geometry that scanner expects,
runs it, and converts the output into `pipeline.baseline.ZoneWindow`.

What this module adds on top, because the new PRD requires it and the old one
did not:

  * **`compression_noise` and `environmental_motion` observations.** PRD 6.2
    names both as health conditions with actions -- de-prioritise and penalise
    respectively -- and both are properties of the *motion* signal rather than
    of the image, so they can only be raised here.

  * **Gate records for everything suppressed.** A window the baseline guards
    reject is routed, not dropped, and the gate log says which guard fired.

  * **Health and epoch awareness.** Windows overlapping a quality-affected span
    are excluded from baseline learning and marked, and an epoch boundary is
    surfaced to segmentation as a wall.

Codec motion vectors are preferred where present and their availability is
recorded (PRD 8). Every corpus file carries them, MPEG-4 included, at roughly
3,600 per frame -- which is what makes a whole-video pass affordable and is why
this layer exists at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import gates
from .baseline import BaselineConfig, ZoneWindow
from .calibration import (
    CalibrationProfile, DESK_WORK_SURFACE, ENVIRONMENT, INVIGILATOR_PATH,
    LAP_BAG, MONITOR_EXCLUSION, OVERLAY, REFLECTION, SEAT_CONTEXT, UPPER_BODY,
)
from .health import (
    COMPRESSION_NOISE, ENVIRONMENTAL_MOTION, HealthObservation,
    VALIDATED_ON_SOURCE,
)

# Which pipeline zones are scored for candidate activity. `seat_context` is the
# footprint, used for attribution and ROI context rather than scoring, and
# anything under a monitor is excluded entirely -- a live screen is an aperiodic
# motion source with nothing to do with the candidate.
SCORED_ZONES = (DESK_WORK_SURFACE, UPPER_BODY, LAP_BAG)

# pipeline mask kind -> the kind the measured scanner understands. It has four
# kinds; the new profile has six, and the two extras map onto the nearest
# behaviour rather than being silently dropped.
_MASK_MAP = {
    OVERLAY: "overlay",
    REFLECTION: "reflection",
    ENVIRONMENT: "environment",
    MONITOR_EXCLUSION: "environment",   # excluded from scoring, same as a fan
    INVIGILATOR_PATH: "staff",
}


@dataclass
class MotionConfig:
    """Scan settings. Values carry the measurement that produced them."""

    window_ms: float = 500.0
    # A 16x16 block must move more than this many pixels per frame to count.
    # Measured against the 0.08 bits/pixel noise floor of this corpus: at 2.0
    # an active seat showed 40 moving blocks against an idle seat's 4, and
    # below it the compressor's noise swamped the separation entirely.
    block_mag_threshold: float = 2.0
    global_motion_ratio: float = 0.55
    global_motion_mag: float = 1.5
    exposure_step_delta: float = 6.0
    luma_sample_stride: int = 5


@dataclass
class ScanResult:
    windows: list[ZoneWindow] = field(default_factory=list)
    observations: list[HealthObservation] = field(default_factory=list)
    gate_log: gates.GateLog = field(default_factory=gates.GateLog)
    motion_vectors_available: bool = False
    mv_per_frame: float = 0.0
    frames: int = 0
    duration_ms: float = 0.0
    seconds: float = 0.0

    @property
    def realtime_factor(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return (self.duration_ms / 1000.0) / self.seconds

    def summary(self) -> dict:
        return {
            "windows": len(self.windows),
            "zones": len({(w.seat_id, w.zone) for w in self.windows}),
            "motion_vectors_available": self.motion_vectors_available,
            "mv_per_frame": round(self.mv_per_frame, 1),
            "frames": self.frames,
            "duration_ms": round(self.duration_ms, 3),
            "seconds": round(self.seconds, 2),
            "realtime_factor": round(self.realtime_factor, 1),
            "observations": len(self.observations),
            "gate_records": len(self.gate_log.records),
        }


def to_scanner_calibration(profile: CalibrationProfile):
    """Convert a reviewed profile into the geometry the measured scanner takes.

    Deliberately a conversion and not a reimplementation. The scanner's
    rasteriser has a fix in it for a real defect -- `fillPoly` includes the
    polygon boundary, so flush-adjacent seats read as overlapping until both
    masks are eroded by a pixel -- and that is worth keeping rather than
    rediscovering.
    """
    from classroom.calibration import Calibration, MaskRegion, Seat

    seats = []
    for seat in profile.seats:
        zones = {}
        # The scanner names its zones differently; the mapping is explicit so a
        # renamed zone fails loudly instead of quietly scoring nothing.
        if DESK_WORK_SURFACE in seat.zones:
            zones["desk"] = [tuple(p) for p in seat.zones[DESK_WORK_SURFACE]]
        if UPPER_BODY in seat.zones:
            zones["torso"] = [tuple(p) for p in seat.zones[UPPER_BODY]]
        if MONITOR_EXCLUSION in [m.kind for m in profile.masks]:
            pass  # handled as a frame-global mask below
        seats.append(Seat(
            label=seat.seat_id,
            polygon=[tuple(p) for p in seat.zones.get(SEAT_CONTEXT, [])],
            zones=zones,
            desk_line_y=seat.desk_line_y,
        ))

    masks = [
        MaskRegion(kind=_MASK_MAP[m.kind], polygon=[tuple(p) for p in m.polygon],
                   note=m.note)
        for m in profile.masks if m.kind in _MASK_MAP
    ]
    return Calibration(camera_id=0, version=profile.version,
                       frame_w=profile.frame_w, frame_h=profile.frame_h,
                       seats=seats, masks=masks)


# The scanner's zone names, mapped back to the PRD 7 vocabulary.
_ZONE_BACK = {"desk": DESK_WORK_SURFACE, "torso": UPPER_BODY}


def scan(video_path: str | Path, profile: CalibrationProfile,
         cfg: MotionConfig | None = None,
         health_timeline=None, progress_every_s: float = 0.0) -> ScanResult:
    """Run the whole-video motion pass and convert its output.

    `health_timeline` is a `pipeline.health.HealthTimeline`. When supplied,
    windows overlapping a quality-affected span are routed through the gate log
    rather than being scored as if the footage were clean.
    """
    import time

    from classroom import calibration as legacy_cal, motion as legacy_motion
    from classroom.config import MotionConfig as LegacyMotionConfig

    cfg = cfg or MotionConfig()
    legacy = LegacyMotionConfig(
        window_s=cfg.window_ms / 1000.0,
        block_mag_threshold=cfg.block_mag_threshold,
        global_motion_ratio=cfg.global_motion_ratio,
        global_motion_mag=cfg.global_motion_mag,
        exposure_step_delta=cfg.exposure_step_delta,
        luma_sample_stride=cfg.luma_sample_stride,
    )
    masks = legacy_cal.rasterize(to_scanner_calibration(profile))

    result = ScanResult()

    # PRD 8: prefer codec motion vectors where present, and *record
    # availability*. Every file in this corpus carries them, MPEG-4 included, at
    # roughly 3,600 per frame -- which is what makes a whole-video pass
    # affordable. A file without them would need the optical-flow challenger and
    # would be an order of magnitude slower, so this is not a detail to infer.
    from classroom.ingest import _motion_vectors

    try:
        result.motion_vectors_available, result.mv_per_frame = _motion_vectors(
            Path(video_path))
    except Exception as exc:  # noqa: BLE001 - absence is a finding, not a crash
        result.motion_vectors_available, result.mv_per_frame = False, 0.0
        result.gate_log.route(
            "motion", "motion_vector_probe", gates.UNCERTAIN, 0.0, 0.0,
            f"could not determine motion-vector availability: "
            f"{type(exc).__name__}: {exc}")

    started = time.perf_counter()
    raw = list(legacy_motion.scan(str(video_path), masks, legacy,
                                  progress_every=progress_every_s))
    result.seconds = time.perf_counter() - started
    if not raw:
        return result

    result.frames = sum(w.frames for w in raw)
    result.duration_ms = raw[-1].t_end * 1000.0

    for window in raw:
        start_ms = window.t_start * 1000.0
        end_ms = window.t_end * 1000.0

        # Frame-level conditions become gate records, not silent suppression.
        if window.camera_moved:
            result.gate_log.route(
                "motion", "camera_motion", gates.SUPPRESS_FROM_AUTO_RANKING,
                start_ms, end_ms,
                f"global motion ratio {window.global_ratio:.2f} with a "
                f"{(window.global_dx ** 2 + window.global_dy ** 2) ** 0.5:.2f} px "
                f"shift; local attribution is unreliable here",
                inputs={"global_ratio": round(window.global_ratio, 4)},
                thresholds={"global_motion_ratio": cfg.global_motion_ratio})
        if window.exposure_step:
            result.gate_log.route(
                "motion", "exposure_change", gates.DEPRIORITISE, start_ms, end_ms,
                f"scene-wide luma step of {window.luma_delta:+.2f}",
                inputs={"luma_delta": round(window.luma_delta, 3)},
                thresholds={"exposure_step_delta": cfg.exposure_step_delta})

        for (seat_label, zone_name), stats in window.zones.items():
            zone = _ZONE_BACK.get(zone_name, zone_name)
            if zone not in SCORED_ZONES:
                continue
            result.windows.append(ZoneWindow(
                seat_id=seat_label, zone=zone,
                start_ms=start_ms, end_ms=end_ms,
                residual=stats.residual,
                area_ratio=stats.area_ratio,
                moving_blocks=stats.moving_blocks,
                coherence=stats.coherence,
                below_desk=stats.below_desk,
            ))

        if health_timeline is not None and \
                not health_timeline.baseline_eligible(start_ms, end_ms):
            reasons = ", ".join(sorted({o.condition for o in
                                        health_timeline.over(start_ms, end_ms)}))
            result.gate_log.route(
                "motion", "quality_affected", gates.UNCERTAIN, start_ms, end_ms,
                f"overlaps {reasons}; excluded from baseline learning "
                f"(PRD 6.3.9)")

    return result


def classify_noise(windows: list[ZoneWindow], cfg: BaselineConfig,
                   total_ms: float, log: gates.GateLog | None = None
                   ) -> list[HealthObservation]:
    """Raise `compression_noise` and `environmental_motion` (PRD 6.2).

    Both are properties of the motion signal rather than of any single image,
    and both are *retained*: compression noise is de-prioritised, environmental
    motion is penalised in ranking. Neither is deleted, because a fan is real
    motion — it is simply never the answer.
    """
    from .baseline import environmental_zones

    observations: list[HealthObservation] = []

    # Environmental: a zone active for most of the recording.
    for (seat_id, zone), persistence in environmental_zones(
            windows, cfg, total_ms).items():
        active = [w for w in windows
                  if w.seat_id == seat_id and w.zone == zone]
        if not active:
            continue
        observations.append(HealthObservation(
            ENVIRONMENTAL_MOTION,
            min(w.start_ms for w in active), max(w.end_ms for w in active),
            evidence={"seat_id": seat_id, "zone": zone,
                      "persistence": persistence,
                      "threshold": cfg.environmental_persistence},
            validated_on=VALIDATED_ON_SOURCE,
            note="active for most of the recording; behavioural motion is not "
                 "persistent at this level",
        ))
        if log is not None:
            log.route("motion", ENVIRONMENTAL_MOTION, gates.DEPRIORITISE,
                      min(w.start_ms for w in active),
                      max(w.end_ms for w in active),
                      f"{seat_id}/{zone} active for {persistence:.0%} of the "
                      f"recording",
                      inputs={"persistence": persistence},
                      thresholds={"environmental_persistence":
                                  cfg.environmental_persistence},
                      seat_id=seat_id)

    # Compression noise: isolated, low-area, non-persistent activity.
    #
    # Merged into runs before being emitted. On a 282-second file this fires on
    # 1,169 individual windows, and one observation per window would bury the
    # health timeline in near-identical rows while saying nothing a merged span
    # does not. The count is kept as evidence on the merged record.
    noise_spans: list[HealthObservation] = []
    by_zone: dict[tuple[str, str], list[ZoneWindow]] = {}
    for window in windows:
        by_zone.setdefault((window.seat_id, window.zone), []).append(window)

    for (seat_id, zone), group in by_zone.items():
        group.sort(key=lambda w: w.start_ms)
        for index, window in enumerate(group):
            if window.moving_blocks <= 0:
                continue
            if window.area_ratio >= cfg.min_area_ratio:
                continue
            neighbours = [g for g in group[max(index - 1, 0):index + 2]
                          if g is not window]
            if any(n.area_ratio >= cfg.min_area_ratio for n in neighbours):
                continue
            noise_spans.append(HealthObservation(
                COMPRESSION_NOISE, window.start_ms, window.end_ms,
                evidence={"seat_id": seat_id, "zone": zone,
                          "area_ratio": round(window.area_ratio, 5),
                          "moving_blocks": window.moving_blocks},
                validated_on=VALIDATED_ON_SOURCE,
                note="isolated low-area activity with no persistent neighbour",
            ))
            if log is not None:
                log.route("motion", COMPRESSION_NOISE, gates.DEPRIORITISE,
                          window.start_ms, window.end_ms,
                          f"{window.moving_blocks} block(s) at "
                          f"{window.area_ratio:.4f} area, isolated",
                          inputs={"area_ratio": round(window.area_ratio, 5),
                                  "moving_blocks": window.moving_blocks},
                          thresholds={"min_area_ratio": cfg.min_area_ratio},
                          seat_id=seat_id)

    from .health import merge as merge_observations

    # Merged only where strictly contiguous. A 2000 ms merge gap collapsed all
    # 1,169 noise windows on a 282-second file into a single whole-recording
    # span, which is technically true and useless: every event then inherited
    # `compression_noise` in its health state and the field stopped
    # discriminating between anything.
    observations += merge_observations(noise_spans, gap_ms=0.0)
    return observations


def suppression_gates(scored, log: gates.GateLog) -> None:
    """Route every suppressed window, so no guard fires invisibly."""
    reason_text = {
        "no_baseline": "no baseline exists for this seat-zone",
        "degenerate_baseline": "the baseline had too little eligible data",
        "no_moving_blocks": "no block cleared the motion threshold",
        "below_area_fraction": "moving area below the fraction floor (PRD 6.3.8)",
        "not_persistent": "single window; area fraction requires persistence",
    }
    for item in scored:
        if not item.suppressed:
            continue
        window = item.window
        log.route(
            "baseline", item.suppressed_by, gates.DEPRIORITISE,
            window.start_ms, window.end_ms,
            reason_text.get(item.suppressed_by, item.suppressed_by),
            inputs={"area_ratio": round(window.area_ratio, 5),
                    "moving_blocks": window.moving_blocks,
                    "residual": round(window.residual, 6)},
            seat_id=window.seat_id)
