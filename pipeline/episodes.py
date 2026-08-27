"""Group per-frame object proposals into episodes, and pick frames to verify.

### The problem this exists to solve

The referee was called once per proposal row. Run 1506 spent 1,996 calls on 169
seconds of video. At that density a three-hour recording needs roughly 128,000
calls: not deployable, and it is what exhausted a Roboflow quota in one
afternoon.

Almost all of those calls are redundant. A phone lying on a desk produces a
proposal every 250 ms for a minute; SAM 3 is asked the same question 240 times
and gives the same answer. What a reviewer needs to know is "is there a phone
at this person's hands, and when" -- one question per *episode*, not per frame.

### What an episode is

Consecutive proposals for the same track, of the same class, whose boxes
overlap and whose timestamps are close. That is deliberately conservative: two
proposals that do not overlap in space become two episodes, so an object that
moves across the desk is not silently merged with one that never moved.

### Which frames get verified

Not the highest-confidence ones. Picking by score would flatter the result and
tell us nothing about the frames a reviewer would actually doubt. The default
picks, in order:

    first       the appearance -- when the object arrives
    strongest   the detector's own best look at it
    last        whether it is still there at the end
    middle      persistence away from both edges, if the budget stretches

capped at `max_per_episode`. A one-frame episode yields one call, which is
correct: there is nothing else to ask about.

### What this does NOT do

It does not decide anything. Episodes carry every member proposal, so the
verdict from a sampled frame can be applied back across the episode by the
caller, and the unsampled members remain in the record as what they always
were -- unverified proposals, never "confirmed" and never "absent".
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # Two proposals more than this far apart in time start a new episode even
    # if they overlap. At 4 Hz sampling this tolerates three missed samples.
    max_gap_ms: float = 1000.0

    # Minimum box overlap to treat two proposals as the same object. Uses the
    # symmetric measure, because detector boxes vary a lot in tightness across
    # frames for one physical object.
    min_overlap: float = 0.30

    # Calls per episode. Four is the appearance, the best look, the middle and
    # the end; below that persistence stops being observable.
    max_per_episode: int = 4


@dataclass
class Episode:
    track_id: int
    cls: str
    members: list = field(default_factory=list)   # (pts_ms, item) in time order

    @property
    def start_ms(self) -> float:
        return self.members[0][0] if self.members else 0.0

    @property
    def end_ms(self) -> float:
        return self.members[-1][0] if self.members else 0.0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def peak_confidence(self) -> float:
        return max((float(i.get("confidence") or 0.0) for _, i in self.members),
                   default=0.0)

    def __len__(self) -> int:
        return len(self.members)


def overlap(box, other) -> float:
    """Symmetric overlap, matching the referee's confirmation test."""
    x1 = max(box[0], other[0])
    y1 = max(box[1], other[1])
    x2 = min(box[2], other[2])
    y2 = min(box[3], other[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    b = max((other[2] - other[0]) * (other[3] - other[1]), 1e-6)
    return max(inter / a, inter / b)


def build(proposals: list, cfg: Config | None = None) -> list:
    """Group `(pts_ms, track_id, item)` triples into episodes.

    Input need not be sorted; output is ordered by track, class, then time.
    """
    cfg = cfg or Config()
    by_key: dict = {}
    for pts_ms, track_id, item in proposals:
        by_key.setdefault((track_id, str(item.get("cls") or "")), []).append(
            (float(pts_ms), item))

    episodes = []
    for (track_id, cls), rows in sorted(by_key.items(), key=lambda kv: kv[0]):
        rows.sort(key=lambda r: r[0])
        open_episodes: list = []
        for pts_ms, item in rows:
            box = item.get("box") or [0.0] * 4
            placed = None
            # Most recent match wins, so a stationary object keeps one episode
            # while a second object elsewhere on the desk keeps its own.
            for episode in reversed(open_episodes):
                last_pts, last_item = episode.members[-1]
                if pts_ms - last_pts > cfg.max_gap_ms:
                    continue
                if overlap(box, last_item.get("box") or [0.0] * 4) >= cfg.min_overlap:
                    placed = episode
                    break
            if placed is None:
                placed = Episode(track_id=track_id, cls=cls)
                open_episodes.append(placed)
                episodes.append(placed)
            placed.members.append((pts_ms, item))
    return episodes


def representatives(episode: Episode, cfg: Config | None = None) -> list:
    """The frames worth asking the referee about. Returns (pts_ms, item)."""
    cfg = cfg or Config()
    members = episode.members
    if len(members) <= cfg.max_per_episode:
        return list(members)

    strongest = max(range(len(members)),
                    key=lambda i: float(members[i][1].get("confidence") or 0.0))
    # Priority order, not chronological: whatever the budget is, spend it on
    # the appearance, the detector's best look, and the end before the middle.
    # With a budget of 3 the middle is what to lose -- persistence is read from
    # the two ends, and dropping the last frame instead answers "was it still
    # there" with silence.
    wanted = [0, strongest, len(members) - 1, len(members) // 2]
    picked, seen = [], set()
    for index in wanted:
        if index not in seen and len(picked) < cfg.max_per_episode:
            seen.add(index)
            picked.append(members[index])
    picked.sort(key=lambda m: m[0])
    return picked


def plan(proposals: list, cfg: Config | None = None) -> tuple:
    """(episodes, frames_to_verify, saving) for a whole run.

    `frames_to_verify` is a set of `(track_id, round(pts_ms, 1), box_key)`, so
    a caller can test membership without rebuilding anything.
    """
    cfg = cfg or Config()
    episodes = build(proposals, cfg)
    keep = set()
    for episode in episodes:
        for pts_ms, item in representatives(episode, cfg):
            keep.add((episode.track_id, round(pts_ms, 1),
                      tuple(round(float(v), 1) for v in
                            (item.get("box") or [0.0] * 4))))
    total = len(proposals)
    saving = 1.0 - (len(keep) / total) if total else 0.0
    return episodes, keep, saving
