"""Per-seat robust baselines, health-aware (PRD 6.3.8, 6.3.9).

What "normal" looks like, learned per seat and per zone, so that a deviation
means "unusual for this seat" rather than "above some number". A fixed pixel
threshold that works for the front row is meaningless for the back.

Four guards, and every one of them exists because its absence was measured
producing wrong output on real footage:

**A relative spread floor.** Residual magnitudes are scene-dependent. On real
720p CCTV a seat's median residual sits around 0.01 with a MAD near 0.0016,
while a high-contrast synthetic fixture produces values two orders of magnitude
larger. An absolute floor of 0.05 against a true MAD of 0.0016 divides every
z-score by 30 and the gate goes blind. Measured exactly that way: fixtures
passed, real footage produced zero events with a peak z of 1.5 where the true
value was ~47.

**An absolute spread floor.** The other half of the same lesson. On a genuinely
static seat both median and MAD approach zero, so a purely relative floor also
collapses and a trivial fluctuation divided by a near-zero denominator reports
an enormous deviation from nothing at all. Measured: a static seat produced
z = 16.6 with zero moving blocks.

**A moving-area fraction, not a block count.** PRD 6.3.8 forbids an absolute
moving-block count alone, and this is why: a large zone accumulates stray blocks
in proportion to its area, so one block of compression noise passes a count
guard. Measured on `03.CCTV Mobile Usage.mkv` — an unoccupied desk row produced
the twelve highest-scoring events in the recording at 29.8 sigma against 9.5 for
the one occupied seat, on 0.1% area motion against 4.65%.

**A regime fraction.** Typing in a CBT hall is near-continuous, so a seat that is
genuinely working sits above its idle centre most of the time. Below a minimum
fraction the excursions are *events*, and treating them as a regime trains the
baseline to ignore exactly what the system exists to find. Measured: two of
three planted events were absorbed and recall fell to 1/3.

On top of those, PRD 6.3.9: baseline learning **excludes quality-affected
spans**, and records which windows were included and which were excluded with
the reason. A baseline that learns through a blackout learns that darkness is
normal.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

IDLE = "idle"
TYPING = "typing"
UNKNOWN = "unknown"
REGIMES = (IDLE, TYPING, UNKNOWN)


@dataclass
class BaselineConfig:
    """Every threshold, named so it can be versioned and hashed (PRD 17.10)."""

    # Duration of footage analysed to learn a seat's regimes.
    learn_ms: float = 180_000.0

    # A window is 'typing' when desk-zone motion exceeds the idle centre by this
    # many robust deviations.
    typing_z: float = 2.0

    # A typing regime is established only when the active windows make up at
    # least this fraction of the learning span.
    min_regime_fraction: float = 0.25

    # Guards a degenerate baseline on a seat empty or static throughout.
    min_windows: int = 60

    # Spread floors: relative to the seat's own median, and absolute.
    min_spread_fraction: float = 0.05
    min_spread_absolute: float = 0.001

    # Motion must be present in the zone, both as a count and as a share of it.
    min_moving_blocks: int = 1
    min_area_ratio: float = 0.05

    # Persistence: PRD 6.3.8 requires area fraction *plus persistence*. A single
    # window above threshold is not activity, whatever its area.
    min_persistent_windows: int = 2

    # A zone active for more than this share of the whole recording is
    # environmental rather than behavioural. Nothing a candidate does persists
    # at this level; a fan or a monitor does.
    environmental_persistence: float = 0.50


@dataclass
class ExcludedWindow:
    """A window kept out of baseline learning, and why (PRD 6.3.9)."""

    start_ms: float
    end_ms: float
    reason: str


@dataclass
class Baseline:
    """The robust centre and spread of one seat-zone's motion distribution."""

    seat_id: str
    zone: str
    median: float
    spread: float                      # MAD
    p95: float
    sample_windows: int
    typing_median: float | None = None
    typing_spread: float | None = None
    excluded_windows: int = 0
    excluded_ms: float = 0.0
    exclusions: list[ExcludedWindow] = field(default_factory=list)
    degenerate: bool = False
    degenerate_reason: str = ""

    def floor(self, centre: float, cfg: BaselineConfig) -> float:
        """The spread actually used, after both floors.

        Both are needed and neither is sufficient. See the module docstring for
        the two measurements that established that.
        """
        return max(self.spread if centre == self.median else (self.typing_spread or 0.0),
                   centre * cfg.min_spread_fraction,
                   cfg.min_spread_absolute)

    def deviation(self, residual: float, cfg: BaselineConfig,
                  regime: str = IDLE) -> tuple[float, str]:
        """Robust z-score of one measurement, and which regime judged it."""
        if self.degenerate:
            return 0.0, UNKNOWN
        if regime == TYPING and self.typing_median is not None:
            centre = self.typing_median
            which = TYPING
        else:
            centre = self.median
            which = IDLE
        spread = self.floor(centre, cfg)
        if spread <= 0:
            return 0.0, UNKNOWN
        return (residual - centre) / spread, which

    def to_row(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "exclusions"},
            "exclusions": [asdict(e) for e in self.exclusions],
        }


@dataclass
class ZoneWindow:
    """One measured window of one seat-zone, as the motion scan produced it."""

    seat_id: str
    zone: str
    start_ms: float
    end_ms: float
    residual: float
    area_ratio: float
    moving_blocks: int
    coherence: float = 0.0
    below_desk: float = 0.0


def learn(windows: list[ZoneWindow], cfg: BaselineConfig,
          eligible=None) -> dict[tuple[str, str], Baseline]:
    """Learn one baseline per seat-zone from eligible windows only.

    `eligible(start_ms, end_ms) -> (bool, reason)` decides whether a window may
    contribute. Pass `HealthTimeline.baseline_eligible` wrapped to return a
    reason; PRD 6.3.9 requires the excluded windows recorded, not merely skipped.
    """
    grouped: dict[tuple[str, str], list[ZoneWindow]] = {}
    for window in windows:
        grouped.setdefault((window.seat_id, window.zone), []).append(window)

    out: dict[tuple[str, str], Baseline] = {}
    for key, group in grouped.items():
        seat_id, zone = key
        group.sort(key=lambda w: w.start_ms)
        horizon = group[0].start_ms + cfg.learn_ms
        considered = [w for w in group if w.start_ms < horizon]

        kept: list[ZoneWindow] = []
        exclusions: list[ExcludedWindow] = []
        for window in considered:
            if eligible is None:
                kept.append(window)
                continue
            allowed, reason = eligible(window.start_ms, window.end_ms)
            if allowed:
                kept.append(window)
            else:
                exclusions.append(ExcludedWindow(window.start_ms, window.end_ms,
                                                 reason))

        excluded_ms = sum(e.end_ms - e.start_ms for e in exclusions)

        if len(kept) < cfg.min_windows:
            out[key] = Baseline(
                seat_id, zone, 0.0, 0.0, 0.0, len(kept),
                excluded_windows=len(exclusions), excluded_ms=excluded_ms,
                exclusions=exclusions, degenerate=True,
                degenerate_reason=(
                    f"only {len(kept)} eligible window(s), need "
                    f"{cfg.min_windows}"
                    + (f"; {len(exclusions)} excluded by quality"
                       if exclusions else "")),
            )
            continue

        residuals = sorted(w.residual for w in kept)
        median = statistics.median(residuals)
        spread = statistics.median([abs(r - median) for r in residuals])
        p95 = residuals[min(int(len(residuals) * 0.95), len(residuals) - 1)]

        baseline = Baseline(seat_id, zone, median, spread, p95, len(kept),
                            excluded_windows=len(exclusions),
                            excluded_ms=excluded_ms, exclusions=exclusions)

        # A typing regime, only if the seat is above its idle centre most of the
        # time. Otherwise the excursions are events and must not become normal.
        cut = median + cfg.typing_z * max(spread, cfg.min_spread_absolute)
        active = [r for r in residuals if r > cut]
        if active and len(active) / len(residuals) >= cfg.min_regime_fraction:
            typing_median = statistics.median(active)
            baseline.typing_median = typing_median
            baseline.typing_spread = statistics.median(
                [abs(r - typing_median) for r in active])

        out[key] = baseline
    return out


def regime_of(window: ZoneWindow, baseline: Baseline, cfg: BaselineConfig) -> str:
    """Which regime a window belongs to."""
    if baseline.degenerate or baseline.typing_median is None:
        return IDLE if not baseline.degenerate else UNKNOWN
    cut = baseline.median + cfg.typing_z * baseline.floor(baseline.median, cfg)
    return TYPING if window.residual > cut else IDLE


@dataclass
class Scored:
    """A window with its deviation and the reason it was or was not suppressed."""

    window: ZoneWindow
    deviation: float
    regime: str
    suppressed_by: str = ""

    @property
    def suppressed(self) -> bool:
        return bool(self.suppressed_by)


def score(windows: list[ZoneWindow], baselines: dict[tuple[str, str], Baseline],
          cfg: BaselineConfig, persistent: dict[tuple[str, str], set[int]] | None = None
          ) -> list[Scored]:
    """Attach a deviation to each window, suppressing what cannot be activity.

    Suppression is recorded by name rather than by silently zeroing, so the
    gate log can say which guard fired.
    """
    ordered = sorted(windows, key=lambda w: (w.seat_id, w.zone, w.start_ms))
    persistent = persistent if persistent is not None else _persistent_indices(
        ordered, cfg)

    out: list[Scored] = []
    index_by_key: dict[tuple[str, str], int] = {}
    for window in ordered:
        key = (window.seat_id, window.zone)
        position = index_by_key.get(key, 0)
        index_by_key[key] = position + 1

        baseline = baselines.get(key)
        if baseline is None:
            out.append(Scored(window, 0.0, UNKNOWN, "no_baseline"))
            continue

        regime = regime_of(window, baseline, cfg)
        deviation, which = baseline.deviation(window.residual, cfg, regime)

        suppressed = ""
        if baseline.degenerate:
            suppressed = "degenerate_baseline"
        elif window.moving_blocks < cfg.min_moving_blocks:
            suppressed = "no_moving_blocks"
        elif window.area_ratio < cfg.min_area_ratio:
            # PRD 6.3.8. The measurement behind this is in the module docstring.
            suppressed = "below_area_fraction"
        elif position not in persistent.get(key, set()):
            # PRD 6.3.8 requires area fraction *plus persistence*.
            suppressed = "not_persistent"

        out.append(Scored(window, max(deviation, 0.0) if not suppressed else 0.0,
                          which, suppressed))
    return out


def _persistent_indices(ordered: list[ZoneWindow], cfg: BaselineConfig
                        ) -> dict[tuple[str, str], set[int]]:
    """Positions belonging to a run of at least `min_persistent_windows`.

    Persistence is measured on the *area* signal rather than the deviation,
    because deviation is what persistence is meant to qualify. Using deviation
    for both would let one loud window vouch for itself.
    """
    runs: dict[tuple[str, str], set[int]] = {}
    by_key: dict[tuple[str, str], list[tuple[int, ZoneWindow]]] = {}
    for window in ordered:
        key = (window.seat_id, window.zone)
        by_key.setdefault(key, []).append((len(by_key.get(key, [])), window))

    for key, entries in by_key.items():
        keep: set[int] = set()
        run: list[int] = []
        for position, window in entries:
            if window.area_ratio >= cfg.min_area_ratio:
                run.append(position)
            else:
                if len(run) >= cfg.min_persistent_windows:
                    keep.update(run)
                run = []
        if len(run) >= cfg.min_persistent_windows:
            keep.update(run)
        runs[key] = keep
    return runs


def environmental_zones(windows: list[ZoneWindow], cfg: BaselineConfig,
                        total_ms: float) -> dict[tuple[str, str], float]:
    """Zones active for so much of the recording that they are not behavioural.

    Returns persistence per zone for those above the threshold. PRD 6.2 requires
    these retained as named observations and penalised in ranking, not deleted:
    a fan is real motion, it is simply never the answer.
    """
    active_ms: dict[tuple[str, str], float] = {}
    for window in windows:
        if window.area_ratio >= cfg.min_area_ratio:
            key = (window.seat_id, window.zone)
            active_ms[key] = active_ms.get(key, 0.0) + (window.end_ms - window.start_ms)
    if total_ms <= 0:
        return {}
    return {k: round(v / total_ms, 4) for k, v in active_ms.items()
            if v / total_ms >= cfg.environmental_persistence}
