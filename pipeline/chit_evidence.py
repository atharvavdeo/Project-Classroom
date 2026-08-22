"""Turning raw `chit-paper` detections into something worth a reviewer's time.

### The problem this solves

`chit_detector` run over every person in `12_paper` produced **1,817
detections on 1,616 of 4,934 crops** -- a third of every person-sighting in a
video where one candidate takes one piece of paper. Rendered and inspected,
the detector fires on:

* white and light-coloured **shirt fabric**,
* the frosted **desk partition panels**,
* a photograph displayed **on a monitor**, at 0.65 -- higher than four of the
  five confidences on hand-verified true positives.

So a confidence floor cannot separate them. The score distribution of the
false positives sits on top of the true positives.

### What was tried and rejected

**Duration/persistence gating** -- the argument that a chit is transient while
a shirt or an answer sheet is permanent. It is a good argument and it is what
`person_timeline.persistent_object_beside_seat` already does for D-FINE. On
this video it **removes the true positive**.

The subject (tracks 118 and 53 -- one man, split by the re-identification gap)
carries a detection in 95% of his sightings, because he genuinely handles the
paper for most of the recording. "Seat 12 was seen taking a piece of paper from
the desk" describes someone who takes it and then keeps it. Ranking by
transience puts him last. Measured: at a duty-cycle cap of 0.25 he is excluded
entirely, and the survivors are five tracks whose detections land on background
furniture.

This is worth stating plainly because the intuition is so reasonable: *for this
class of event, persistence is the signal, not the artefact.*

### What actually separates them

**Where the detection is, relative to the person's hands.** The false positives
are on torsos, panels and screens; the true positives are at the wrists. On the
full run, **972 of 1,817 detections (53%) lie more than 0.35 person-widths from
any resolved wrist**, and removing them:

    track 118   310 -> 219 detections   (the subject, rank 1)
    track  53   101 ->  47              (the same man, rank 3)
    track  16   310 ->   7              (a seated person + monitor, collapses)
    track 100   219 ->  35

Combined with a size cap -- a chit is small; a shirt-sized box is not a chit --
this is what the gates below implement.

### What it still cannot do

It cannot tell a chit from a legitimate answer sheet. Both are small, pale, and
held. Everything here narrows *"someone is handling a small pale object"* down
to the people that is actually true of. Deciding whether that object should be
on the desk is a human's call, and the ranked output is built to be reviewed,
not believed.

There is no ground-truth manifest for this video, so nothing below is a
precision or recall figure. The ranking was checked by rendering the top tracks
and looking at them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Detections carrying this source come from `chit_detector`.
CHIT_SOURCE_PREFIX = "roboflow"


@dataclass
class EvidenceConfig:
    """Gates applied to each detection, then to each track."""

    # A detection must score at least this. Deliberately low: the verified true
    # positives run 0.52-0.74 and the worst false positive reaches 0.65, so
    # this gate does almost no work and must not be relied on. It is here to
    # drop the 0.22-0.28 rubbish, nothing more.
    min_confidence: float = 0.40

    # Detection area as a fraction of the person's box. The chit is small and
    # specific; a box covering a quarter of a person is their shirt. Measured
    # on the run: median 0.099, p95 0.250, max 0.782.
    max_size_ratio: float = 0.12

    # Distance from the detection centre to the nearest resolved wrist, in
    # person-box widths. This is the gate that does the real work -- see the
    # module docstring. 0.35 was chosen because it retains the subject's
    # hand-held detections while dropping torso and monitor hits; it is a
    # relative measure so it transfers between camera distances.
    max_wrist_distance: float = 0.35

    # A detection in a frame where neither wrist resolved cannot be placed
    # relative to the hands. Dropping it silently would understate coverage,
    # so it is counted separately and reported.
    drop_when_no_wrist: bool = True

    # Sightings a track needs before its rates mean anything.
    min_sightings: int = 8

    # A gap of this many samples inside a run of detections is treated as the
    # same episode rather than two, so a single missed frame does not split
    # one handling event in half.
    episode_gap_samples: int = 2


@dataclass
class TrackEvidence:
    """What the chit detector observed about one track, after gating."""

    track_id: int
    sightings: int = 0
    wrist_resolved_sightings: int = 0

    raw_detections: int = 0
    kept_detections: int = 0
    rejected_low_confidence: int = 0
    rejected_too_large: int = 0
    rejected_far_from_hands: int = 0
    rejected_no_wrist: int = 0

    # Sightings carrying at least one surviving detection, over sightings where
    # a wrist was resolved -- the only sightings where the test could run.
    handling_rate: float = 0.0

    episodes: list = field(default_factory=list)      # (start_ms, end_ms)
    longest_episode_ms: float = 0.0
    total_handling_ms: float = 0.0

    peak_confidence: float = 0.0
    median_size_ratio: float = 0.0

    # Sightings where D-FINE independently reported a book/notebook-like
    # object. Stock COCO recognises stationery reliably, so a high rate here
    # says there is real paperwork at this seat -- which is context for a
    # reviewer, not a disqualification.
    dfine_paper_rate: float = 0.0

    disposition: str = ""
    reasons: list = field(default_factory=list)


def _wrists(sample: dict) -> list:
    return [w for w in (sample.get("left_wrist"), sample.get("right_wrist"))
            if w]


def gate_detection(sample: dict, detection: dict, cfg: EvidenceConfig):
    """Judge one detection. Returns (kept, reason, size_ratio, wrist_distance).

    `reason` names the gate that rejected it, or "" when kept, so the counts
    add up and a rejected detection can be explained rather than just missing.
    """
    box = sample.get("box") or [0, 0, 0, 0]
    person_w = max(box[2] - box[0], 1e-6)
    person_area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)

    det = detection.get("box") or [0, 0, 0, 0]
    size_ratio = ((det[2] - det[0]) * (det[3] - det[1])) / person_area

    if float(detection.get("confidence", 0.0)) < cfg.min_confidence:
        return False, "low_confidence", size_ratio, None
    if size_ratio > cfg.max_size_ratio:
        return False, "too_large", size_ratio, None

    wrists = _wrists(sample)
    if not wrists:
        if cfg.drop_when_no_wrist:
            return False, "no_wrist", size_ratio, None
        return True, "", size_ratio, None

    cx = (det[0] + det[2]) / 2.0
    cy = (det[1] + det[3]) / 2.0
    distance = min(math.hypot(cx - w[0], cy - w[1]) for w in wrists) / person_w
    if distance > cfg.max_wrist_distance:
        return False, "far_from_hands", size_ratio, distance
    return True, "", size_ratio, distance


def _episodes(stamps: list, hits: list, gap_samples: int, interval_ms: float):
    """Contiguous stretches of handling, tolerating `gap_samples` misses."""
    out = []
    start = end = None
    missed = 0
    for stamp, hit in zip(stamps, hits):
        if hit:
            if start is None:
                start = stamp
            end = stamp
            missed = 0
        elif start is not None:
            missed += 1
            if missed > gap_samples:
                out.append((start, end + interval_ms))
                start = end = None
                missed = 0
    if start is not None:
        out.append((start, end + interval_ms))
    return out


def evaluate(samples: list, cfg: EvidenceConfig | None = None) -> TrackEvidence:
    """Gate and summarise every chit detection on one track's samples."""
    cfg = cfg or EvidenceConfig()
    rows = sorted(samples, key=lambda s: float(s["pts_ms"]))
    track_id = int(rows[0]["track_id"]) if rows else -1
    out = TrackEvidence(track_id=track_id, sightings=len(rows))

    stamps = [float(r["pts_ms"]) for r in rows]
    interval = 250.0
    if len(stamps) > 2:
        deltas = sorted(b - a for a, b in zip(stamps, stamps[1:]) if b > a)
        if deltas:
            interval = deltas[len(deltas) // 2]

    hits: list[bool] = []
    sizes: list[float] = []
    dfine_paper = 0

    for row in rows:
        if _wrists(row):
            out.wrist_resolved_sightings += 1
        hit = False
        for item in (row.get("near_hand_objects") or []):
            if isinstance(item, str):
                # Pre-existing D-FINE attachment from run 1501, no geometry.
                if item == "book_or_notebook_like_object":
                    dfine_paper += 1
                continue
            source = str(item.get("source", ""))
            if not source.startswith(CHIT_SOURCE_PREFIX):
                if item.get("cls") == "book_or_notebook_like_object":
                    dfine_paper += 1
                continue
            out.raw_detections += 1
            kept, reason, size_ratio, _distance = gate_detection(row, item, cfg)
            if kept:
                out.kept_detections += 1
                hit = True
                sizes.append(size_ratio)
                out.peak_confidence = max(out.peak_confidence,
                                          float(item.get("confidence", 0.0)))
            else:
                setattr(out, f"rejected_{reason}",
                        getattr(out, f"rejected_{reason}") + 1)
        hits.append(hit)

    denominator = out.wrist_resolved_sightings or out.sightings
    out.handling_rate = (sum(hits) / denominator) if denominator else 0.0
    out.dfine_paper_rate = dfine_paper / len(rows) if rows else 0.0
    if sizes:
        sizes.sort()
        out.median_size_ratio = round(sizes[len(sizes) // 2], 4)

    out.episodes = _episodes(stamps, hits, cfg.episode_gap_samples, interval)
    if out.episodes:
        out.longest_episode_ms = max(e - s for s, e in out.episodes)
        out.total_handling_ms = sum(e - s for s, e in out.episodes)
    return out


def classify(evidence: TrackEvidence, cfg: EvidenceConfig | None = None) -> None:
    """Assign a disposition and record why, in place.

    Note what is deliberately absent: no rule punishes a *high* handling rate.
    The subject of this recording handles paper in 95% of his sightings, and an
    earlier version of this function ranked him last for exactly that reason.
    """
    cfg = cfg or EvidenceConfig()
    reasons: list[str] = []

    if evidence.sightings < cfg.min_sightings:
        evidence.disposition = "insufficient_observation"
        evidence.reasons = [f"only {evidence.sightings} sightings"]
        return
    if not evidence.kept_detections:
        evidence.disposition = "no_handling_observed"
        if evidence.raw_detections:
            reasons.append(
                f"{evidence.raw_detections} detections, all rejected "
                f"({evidence.rejected_far_from_hands} away from hands, "
                f"{evidence.rejected_too_large} too large, "
                f"{evidence.rejected_no_wrist} with no wrist resolved)")
        evidence.reasons = reasons
        return

    reasons.append(
        f"{evidence.kept_detections} of {evidence.raw_detections} detections "
        f"survived, at the hands and small enough")
    reasons.append(f"handling in {evidence.handling_rate:.0%} of sightings "
                   f"where a wrist resolved")
    reasons.append(f"longest continuous handling "
                   f"{evidence.longest_episode_ms / 1000:.1f}s across "
                   f"{len(evidence.episodes)} episode(s)")
    if evidence.dfine_paper_rate > 0.2:
        reasons.append(
            f"D-FINE also reports notebook/paper in "
            f"{evidence.dfine_paper_rate:.0%} of sightings -- there is real "
            f"paperwork at this seat, which may be legitimate")

    if evidence.longest_episode_ms >= 5000.0:
        evidence.disposition = "sustained_paper_handling"
    elif len(evidence.episodes) >= 3:
        evidence.disposition = "repeated_brief_paper_handling"
    else:
        evidence.disposition = "brief_paper_handling"
    evidence.reasons = reasons


def rank(all_evidence: list) -> list:
    """Order tracks for review: most handling first.

    Ranking is by total handling time rather than by detection count, so a
    track seen twice as often does not outrank one that handled paper longer.
    """
    reportable = [e for e in all_evidence if e.kept_detections]
    reportable.sort(key=lambda e: (-e.total_handling_ms, -e.peak_confidence))
    return reportable
