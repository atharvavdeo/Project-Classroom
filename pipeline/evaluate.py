"""Post-hoc evaluation against the reference manifest (PRD 15).

Stage 12, and the first place in the pipeline permitted to read reference data.

### Seven outcomes, not two

PRD 15 requires each reference to be classified as `TP`, `FP`, `FN`,
`ambiguous`, `insufficient_visual_evidence`, `quality_unavailable`, or
`not_routed_to_candidate`. The last three are what make the evaluation useful
rather than merely scored.

A reference the motion stage never routed to a candidate is not a detector
failure -- the detector was never shown that frame. A reference falling in a
span the health monitor marked unavailable is not a failure of anything; the
data was not there. Collapsing all of these into "FN" produces a recall number
that blames the detector for the camera, and optimising against it leads
directly to loosening the motion gate until everything is a candidate.

### The stage-loss waterfall

For every reference that did not become a true positive, `attribute_loss` names
the earliest stage that could have carried it and did not. That is the output
this whole pipeline exists to produce: not "recall was 0.6" but "of 40
references, 12 were never routed, 9 had no crop above the resolution floor, 6
were detected but not temporally confirmed".

### Zero detections must still produce metrics

PRD 15: "Zero detections must yield valid metrics rather than crash." Every
division here is guarded and every empty case returns a defined value with a
note, because the zero-detection run is the one most likely to be a real result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Outcomes, exactly as PRD 15 names them.
TP = "TP"
FP = "FP"
FN = "FN"
AMBIGUOUS = "ambiguous"
INSUFFICIENT = "insufficient_visual_evidence"
QUALITY_UNAVAILABLE = "quality_unavailable"
NOT_ROUTED = "not_routed_to_candidate"
OUTCOMES = (TP, FP, FN, AMBIGUOUS, INSUFFICIENT, QUALITY_UNAVAILABLE, NOT_ROUTED)

# Stage-loss attribution, in pipeline order. The earliest stage that could have
# carried a reference and did not is the one credited with the loss.
STAGE_MOTION = "motion_roi"
STAGE_GATE = "gate"
STAGE_CROP = "crop"
STAGE_TRACKER = "tracker_seat"
STAGE_POSE = "pose_hand"
STAGE_DETECTOR = "detector"
STAGE_TEMPORAL = "temporal_confirmer"
STAGE_VLM = "vlm_disagreement"
LOSS_STAGES = (STAGE_MOTION, STAGE_GATE, STAGE_CROP, STAGE_TRACKER, STAGE_POSE,
               STAGE_DETECTOR, STAGE_TEMPORAL, STAGE_VLM)


@dataclass
class MatchConfig:
    # How far a detection may sit from a reference in time and still be the same
    # sighting. One sample interval at 2 Hz, so a reference between two sampled
    # frames can match the nearer one.
    pts_tolerance_ms: float = 500.0

    # Spatial overlap required when the reference carries a box. Low by the
    # standards of object detection, deliberately: at 30x20 px a few pixels of
    # box disagreement destroys IoU without meaning the object was missed.
    min_iou: float = 0.20

    # Whether a class mismatch is a false positive or an ambiguous match. PRD 15
    # requires class agreement, so a phone detected where a chit is referenced
    # is not a true positive -- but it is also not the same failure as detecting
    # nothing, so it is recorded as ambiguous.
    strict_class: bool = True


@dataclass
class Outcome:
    """One reference row, under one configuration, and what happened to it."""

    reference_id: str
    config_id: str
    cls: str
    outcome: str
    source_pts_ms: float
    visibility: str = "unknown"
    seat_id: str | None = None
    matched_detection: dict | None = None
    matched_iou: float | None = None
    matched_pts_delta_ms: float | None = None
    loss_stage: str | None = None
    rank: int | None = None
    reason: str = ""

    def to_row(self) -> dict:
        return asdict(self)


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class RunView:
    """What the run produced, from the artifacts, for one configuration.

    Assembled by the caller and passed in, so this module never reaches into a
    run folder. That keeps the evaluator testable on synthetic data and keeps
    the reference data flowing one way only.
    """

    config_id: str
    candidate_spans: list[tuple[float, float]] = field(default_factory=list)
    failed_spans: list[tuple[float, float, str]] = field(default_factory=list)
    suppressed_spans: list[tuple[float, float, str]] = field(default_factory=list)
    detections: list[dict] = field(default_factory=list)
    abstentions: list[dict] = field(default_factory=list)
    crops: list[dict] = field(default_factory=list)
    ranks: dict = field(default_factory=dict)
    duration_ms: float = 0.0

    def routed(self, pts_ms: float) -> bool:
        return any(lo <= pts_ms <= hi for lo, hi in self.candidate_spans)

    def failed_at(self, pts_ms: float) -> str | None:
        for lo, hi, reason in self.failed_spans:
            if lo <= pts_ms <= hi:
                return reason
        return None

    def suppressed_at(self, pts_ms: float) -> str | None:
        for lo, hi, reason in self.suppressed_spans:
            if lo <= pts_ms <= hi:
                return reason
        return None

    def abstained_at(self, pts_ms: float, tolerance: float) -> dict | None:
        for row in self.abstentions:
            if abs(row["pts_ms"] - pts_ms) <= tolerance:
                return row
        return None

    def crop_at(self, pts_ms: float, tolerance: float) -> dict | None:
        for row in self.crops:
            if abs(row["pts_ms"] - pts_ms) <= tolerance:
                return row
        return None


def match_one(reference, view: RunView, cfg: MatchConfig) -> Outcome:
    """Classify one reference against one configuration's output."""
    outcome = Outcome(
        reference_id=reference.reference_id, config_id=view.config_id,
        cls=reference.cls, outcome=FN,
        source_pts_ms=reference.source_pts_ms,
        visibility=reference.visibility, seat_id=reference.seat_id)

    # 1. Was the data even there? A reference inside a decode gap or blackout is
    #    not a failure of any model.
    failure = view.failed_at(reference.source_pts_ms)
    if failure:
        outcome.outcome = QUALITY_UNAVAILABLE
        outcome.reason = (f"the source is unavailable here ({failure}); no "
                          f"configuration was shown this frame")
        return outcome

    # 2. Did motion route it to a candidate at all? If not, the detector never
    #    saw it, and calling this a detector miss would blame the wrong stage.
    if not view.routed(reference.source_pts_ms):
        outcome.outcome = NOT_ROUTED
        outcome.loss_stage = STAGE_MOTION
        outcome.reason = ("no candidate interval covers this timestamp, so no "
                          "crop, pose or detector ever processed it")
        return outcome

    # 3. Did a gate hold it back?
    suppressed = view.suppressed_at(reference.source_pts_ms)

    # 4. Look for a matching detection.
    best, best_iou, best_delta = None, 0.0, None
    for detection in view.detections:
        delta = abs(detection["pts_ms"] - reference.source_pts_ms)
        if delta > cfg.pts_tolerance_ms:
            continue
        if reference.bbox_xyxy_source is not None:
            overlap = iou(reference.bbox_xyxy_source,
                          (detection["x1"], detection["y1"],
                           detection["x2"], detection["y2"]))
            if overlap < cfg.min_iou:
                continue
        else:
            overlap = 1.0        # unlocalised reference: time and class only
        if overlap > best_iou or (best is None and overlap >= cfg.min_iou):
            best, best_iou, best_delta = detection, overlap, delta

    if best is not None:
        outcome.matched_detection = best
        outcome.matched_iou = round(best_iou, 4)
        outcome.matched_pts_delta_ms = round(best_delta, 1)
        outcome.rank = view.ranks.get(best.get("event_id"))

        if cfg.strict_class and best.get("cls") != reference.cls:
            outcome.outcome = AMBIGUOUS
            outcome.loss_stage = STAGE_DETECTOR
            outcome.reason = (f"an object was found here but classified "
                              f"{best.get('cls')!r} against reference "
                              f"{reference.cls!r}")
            return outcome

        if best.get("status") == "single_frame_object_candidate":
            # Found, but not corroborated. Still a true positive -- it was
            # detected and surfaced -- with the weaker status recorded.
            outcome.outcome = TP
            outcome.reason = ("matched a single-frame candidate; found and "
                              "retained, but not temporally corroborated")
            return outcome

        if suppressed:
            outcome.outcome = TP
            outcome.reason = (f"matched, but a gate marked this span "
                              f"{suppressed!r}, so it is excluded from "
                              f"automatic ordering while remaining searchable")
            return outcome

        outcome.outcome = TP
        outcome.reason = (f"matched at IoU {best_iou:.2f}, "
                          f"{best_delta:.0f} ms from the reference timestamp")
        return outcome

    # 5. No detection. Was the crop even capable of carrying one?
    abstention = view.abstained_at(reference.source_pts_ms, cfg.pts_tolerance_ms)
    if abstention is not None:
        outcome.outcome = INSUFFICIENT
        outcome.loss_stage = STAGE_CROP
        detail = abstention.get("abstain_reason") or "below the resolution floor"
        outcome.reason = f"the crop abstained: {detail}"
        return outcome

    crop = view.crop_at(reference.source_pts_ms, cfg.pts_tolerance_ms)
    if crop is None:
        outcome.outcome = FN
        outcome.loss_stage = STAGE_CROP
        outcome.reason = ("routed to a candidate, but no crop covers this "
                          "timestamp; the loss is before the detector")
        return outcome

    if suppressed:
        outcome.outcome = FN
        outcome.loss_stage = STAGE_GATE
        outcome.reason = f"no detection, and a gate marked this span {suppressed!r}"
        return outcome

    outcome.outcome = FN
    outcome.loss_stage = STAGE_DETECTOR
    outcome.reason = ("a crop of adequate resolution was produced and the "
                      "detector returned nothing matching")
    return outcome


def false_positives(view: RunView, references, cfg: MatchConfig,
                    target_classes: tuple[str, ...] = ()) -> list[Outcome]:
    """Detections matching no reference.

    Restricted to the target classes by default. A detector reporting a keyboard
    where no keyboard is referenced is not a false positive against a manifest
    that only annotates phones and chits -- it is an unlabelled observation, and
    counting it would invent a precision figure the references cannot support.
    """
    out: list[Outcome] = []
    for detection in view.detections:
        if target_classes and detection.get("cls") not in target_classes:
            continue
        matched = False
        for reference in references:
            if abs(detection["pts_ms"] - reference.source_pts_ms) > cfg.pts_tolerance_ms:
                continue
            if reference.bbox_xyxy_source is not None:
                if iou(reference.bbox_xyxy_source,
                       (detection["x1"], detection["y1"],
                        detection["x2"], detection["y2"])) < cfg.min_iou:
                    continue
            if cfg.strict_class and detection.get("cls") != reference.cls:
                continue
            matched = True
            break
        if not matched:
            out.append(Outcome(
                reference_id="", config_id=view.config_id,
                cls=detection.get("cls", "unknown_object"), outcome=FP,
                source_pts_ms=detection["pts_ms"],
                matched_detection=detection,
                reason="no reference row within tolerance for this class"))
    return out


def evaluate(manifest, view: RunView, cfg: MatchConfig | None = None,
             target_classes: tuple[str, ...] = ("phone", "secondary_paper_chit")
             ) -> list[Outcome]:
    """Evaluate one configuration against the reference manifest."""
    cfg = cfg or MatchConfig()
    outcomes = [match_one(row, view, cfg) for row in manifest.rows]
    outcomes += false_positives(view, manifest.rows, cfg, target_classes)
    return outcomes


# ------------------------------------------------------------- metrics --

def _safe(numerator: float, denominator: float) -> float | None:
    """A ratio, or None. Never a zero standing in for an undefined value."""
    return round(numerator / denominator, 4) if denominator else None


def metrics(outcomes: list[Outcome], manifest, view: RunView,
            cfg: MatchConfig | None = None,
            target_classes: tuple[str, ...] = ("phone", "secondary_paper_chit")
            ) -> dict:
    """Every metric PRD 15 requires, per class and overall.

    Defined for zero detections and for zero references, because both are real
    results. An undefined ratio is `None` with the counts beside it, never 0.0 --
    a precision of 0.0 says the detector was wrong, and a precision of None says
    it was never asked.
    """
    cfg = cfg or MatchConfig()
    hours = view.duration_ms / 3_600_000.0 if view.duration_ms else 0.0

    def per_class(cls: str | None) -> dict:
        rows = [o for o in outcomes if cls is None or o.cls == cls]
        references = [o for o in rows if o.reference_id]
        counts = {name: sum(1 for o in rows if o.outcome == name)
                  for name in OUTCOMES}

        tp, fp, fn = counts[TP], counts[FP], counts[FN]
        abstained = counts[INSUFFICIENT]

        # Recall is reported four ways over three different denominators,
        # because they answer different questions and quoting only one is how a
        # pipeline gets optimised in the wrong direction.
        #
        #   available  references the source could answer for at all -- every
        #              row except those inside a decode gap or blackout.
        #   routed     of those, the ones motion routed to a candidate, so a
        #              crop and a detector actually saw them.
        #
        # A reference in a decode gap is neither routed nor not-routed: nothing
        # was there to route. Counting it as routed inflated routing recall to
        # 1.0 on a fixture where a third of the references were unavailable.
        available = [o for o in references if o.outcome != QUALITY_UNAVAILABLE]
        routed = [o for o in available if o.outcome != NOT_ROUTED]

        return {
            "class": cls or "all",
            "reference_count": len(references),
            **counts,
            "precision": _safe(tp, tp + fp),
            # The strictest: of everything annotated, what reached a reviewer.
            "recall": _safe(tp, len(references)),
            "f1": _safe(2 * tp, 2 * tp + fp + fn) if (tp or fp or fn) else None,
            "fp_per_hour": round(fp / hours, 3) if hours else None,
            # Of what the source could support, how much motion routed.
            "candidate_routing_recall": _safe(len(routed), len(available)),
            # Of what was routed, how much the rest of the pipeline found.
            "recall_conditional_on_routing": _safe(tp, len(routed)),
            # Of what the source could support, how much was found end to end.
            "full_pipeline_recall": _safe(tp, len(available)),
            "abstention_rate": _safe(abstained, len(references)),
            "quality_unavailable_rate": _safe(counts[QUALITY_UNAVAILABLE],
                                              len(references)),
        }

    by_visibility: dict[str, dict] = {}
    for outcome in (o for o in outcomes if o.reference_id):
        bucket = by_visibility.setdefault(
            outcome.visibility, {name: 0 for name in OUTCOMES})
        bucket[outcome.outcome] += 1

    waterfall = {stage: 0 for stage in LOSS_STAGES}
    for outcome in outcomes:
        if outcome.loss_stage:
            waterfall[outcome.loss_stage] += 1

    ranks = [o.rank for o in outcomes if o.reference_id and o.rank is not None]

    return {
        "config_id": view.config_id,
        "complete": manifest.complete,
        "completeness_note": manifest.summary()["note"],
        "duration_ms": round(view.duration_ms, 1),
        "overall": per_class(None),
        # PRD 15: "Phone and chit results must be separate."
        "per_class": {cls: per_class(cls) for cls in target_classes},
        "by_visibility": by_visibility,
        "stage_loss_waterfall": waterfall,
        "waterfall_note":
            "the earliest stage that could have carried a reference and did "
            "not. A reference lost at motion_roi was never shown to a detector, "
            "so it is not a detector error.",
        "candidate_duration_ms": round(
            sum(hi - lo for lo, hi in view.candidate_spans), 1),
        "candidate_duration_fraction": _safe(
            sum(hi - lo for lo, hi in view.candidate_spans), view.duration_ms),
        "events_per_hour": round(len(view.candidate_spans) / hours, 2)
        if hours else None,
        # PRD 15 requires the rank of every reference, so the list is given
        # whole rather than summarised away.
        "reference_ranks": {o.reference_id: o.rank for o in outcomes
                            if o.reference_id},
        "ranked_references": len(ranks),
        "unranked_references": sum(1 for o in outcomes
                                   if o.reference_id and o.rank is None),
        "median_reference_rank": sorted(ranks)[len(ranks) // 2] if ranks else None,
        "note": "zero detections yields these metrics with None for undefined "
                "ratios and counts beside them. A precision of None means the "
                "detector was never asked; 0.0 would mean it was wrong.",
    }


def compare(metrics_by_config: dict, taxonomy_approved: bool = False) -> dict:
    """Compare configurations, and recommend only what the numbers support."""
    if not metrics_by_config:
        return {"configurations": {},
                "recommendation": "no configuration produced evaluable output."}

    complete = all(m.get("complete") for m in metrics_by_config.values())
    ranked = []
    for config_id, entry in metrics_by_config.items():
        overall = entry["overall"]
        ranked.append((config_id, overall.get("f1"), overall.get("precision"),
                       overall.get("recall"), overall.get(TP)))

    measurable = [r for r in ranked if r[1] is not None]
    if not taxonomy_approved:
        recommendation = (
            "None. PRD 3 blocks detector selection until the product owner "
            "approves the taxonomy. The comparison below is recorded for review, "
            "not acted on.")
    elif not complete:
        recommendation = (
            "None. The reference manifest has rejected rows, so every metric "
            "above is incomplete and a selection made from it would be a "
            "selection made from an unknown subset (PRD 14).")
    elif not measurable:
        recommendation = (
            "None. No configuration produced a defined F1, which means none was "
            "asked a question the references can answer. The measured result is "
            "that this corpus and source scale do not support a detector claim.")
    else:
        best = max(measurable, key=lambda r: r[1])
        runner_up = sorted(measurable, key=lambda r: -r[1])[1:2]
        margin = best[1] - runner_up[0][1] if runner_up else None
        if margin is not None and margin < 0.05:
            recommendation = (
                f"No clear winner. {best[0]} leads {runner_up[0][0]} by "
                f"{margin:.3f} F1, which is inside the noise this corpus can "
                f"resolve. Both are retained.")
        else:
            recommendation = (
                f"{best[0]} measures best on this corpus (F1 {best[1]:.3f}, "
                f"precision {best[2]}, recall {best[3]}). This is a measurement "
                f"on {sum(1 for _ in metrics_by_config)} configurations over one "
                f"reference manifest, not a general claim.")

    return {
        "configurations": metrics_by_config,
        "complete": complete,
        "taxonomy_approved": taxonomy_approved,
        "recommendation": recommendation,
        "note": "no forced winner. A configuration that could not run stays in "
                "the table as unavailable (PRD 5).",
    }
