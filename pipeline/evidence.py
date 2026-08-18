"""Evidence packets: what a reviewer, or Gemma, is actually shown (PRD 11, 12).

PRD 11 specifies the packet and ends with a prohibition:

> Each call must include a contact sheet of 4-8 chronological source frames with
> marked/labeled subject seat, magnified native subject crops, used
> hand/desk/lap crops, object boxes/config/scores, health state, and PTS on
> every tile. Whole-frame-only packets are prohibited.

The prohibition is the load-bearing part and it is not arbitrary. In an
examination hall of thirty people, a whole-frame tile shows thirty candidates
and marks none of them. A model handed that will describe *somebody* -- and it
will do so fluently and with high confidence. That is a measured failure from an
earlier phase of this project, where the verifier described the wrong person at
confidence 1.0. Nothing in the output looked wrong. The only defence is
structural: a packet without a marked subject and a magnified native crop is not
a valid packet, and `validate()` refuses it before a model ever sees it.

`PacketSpec` describes a packet without rendering it, so the composition rules
can be tested, and a run can be checked for compliance, without decoding video
or writing PNGs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Tile roles. `full_frame` alone is never sufficient (PRD 11).
FULL_FRAME = "full_frame_marked"
SUBJECT_CROP = "magnified_subject_crop"
HAND_CROP = "hand_or_desk_or_lap_crop"
OBJECT_CROP = "object_detection_crop"
TILE_ROLES = (FULL_FRAME, SUBJECT_CROP, HAND_CROP, OBJECT_CROP)


class PacketError(ValueError):
    """A packet does not meet PRD 11's composition requirements."""


@dataclass
class PacketConfig:
    # PRD 11: 4-8 chronological source frames. Below four a reviewer cannot see
    # a movement develop; above eight the tiles are too small to read at the
    # magnification this footage needs.
    min_frames: int = 4
    max_frames: int = 8

    # The subject crop must be magnified relative to the full frame, but the
    # magnification is bounded and recorded: past this it is visibly
    # interpolated and invites the reader to see detail that is not there.
    max_subject_magnification: float = 8.0

    # Pre/post-roll around the event, so the packet shows the movement starting
    # rather than only its follow-through.
    pre_roll_ms: float = 1500.0
    post_roll_ms: float = 1500.0


@dataclass
class Tile:
    """One image in a contact sheet, with the provenance PRD 11 requires."""

    role: str
    pts_ms: float
    # Source-frame rectangle this tile shows. For a full frame it is the frame.
    x1: float
    y1: float
    x2: float
    y2: float
    # Whether the subject seat is drawn and labelled on this tile. A full-frame
    # tile without it shows a room, not a subject.
    subject_marked: bool = False
    subject_label: str = ""
    # Rendered size, so magnification is derivable rather than asserted.
    rendered_w: int = 0
    rendered_h: int = 0
    pts_stamped: bool = True
    caption: str = ""

    def __post_init__(self) -> None:
        if self.role not in TILE_ROLES:
            raise PacketError(f"unknown tile role {self.role!r}")

    @property
    def native_w(self) -> float:
        return self.x2 - self.x1

    @property
    def native_h(self) -> float:
        return self.y2 - self.y1

    @property
    def magnification(self) -> float:
        longest_native = max(self.native_w, self.native_h)
        longest_render = max(self.rendered_w, self.rendered_h)
        return longest_render / longest_native if longest_native > 0 else 0.0

    def to_row(self) -> dict:
        return {**asdict(self),
                "native_w": round(self.native_w, 2),
                "native_h": round(self.native_h, 2),
                "magnification": round(self.magnification, 2)}


@dataclass
class ObjectEvidence:
    """One object observation carried into the packet, with its configuration."""

    cls: str
    confidence: float
    status: str
    config_id: str
    support_frames: int
    x1: float
    y1: float
    x2: float
    y2: float
    native_px: float = 0.0

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class EvidencePacket:
    """Everything shown for one event, and everything needed to re-derive it."""

    event_id: str
    seat_state: str
    source_video: str
    source_video_sha256: str
    start_pts_ms: float
    end_pts_ms: float
    peak_pts_ms: float
    calibration_version: str
    camera_epoch: str

    tiles: list[Tile] = field(default_factory=list)
    objects: list[ObjectEvidence] = field(default_factory=list)
    health_state: list[str] = field(default_factory=list)
    gate_state: str = "pass"
    pose_observations: dict = field(default_factory=dict)
    motion: dict = field(default_factory=dict)
    priority: str = "normal"
    config_hashes: dict = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)
    review_status: str = "unreviewed"

    @property
    def frame_tiles(self) -> list[Tile]:
        return [t for t in self.tiles if t.role == FULL_FRAME]

    @property
    def subject_crops(self) -> list[Tile]:
        return [t for t in self.tiles if t.role == SUBJECT_CROP]

    def validate(self, cfg: PacketConfig | None = None) -> list[str]:
        """Every reason this packet may not be sent to a model or a reviewer.

        Returns problems rather than raising, so a run can report how many
        packets are malformed and why instead of dying on the first one.
        """
        cfg = cfg or PacketConfig()
        problems: list[str] = []

        frames = self.frame_tiles
        if len(frames) < cfg.min_frames:
            problems.append(
                f"{len(frames)} source frames, under PRD 11's minimum of "
                f"{cfg.min_frames}; a movement cannot be seen to develop")
        if len(frames) > cfg.max_frames:
            problems.append(
                f"{len(frames)} source frames, over PRD 11's maximum of "
                f"{cfg.max_frames}")

        unmarked = [t for t in frames if not t.subject_marked]
        if unmarked:
            problems.append(
                f"{len(unmarked)} full-frame tile(s) do not mark the subject "
                f"seat. A frame showing thirty candidates and marking none of "
                f"them invites a confident description of the wrong person")

        unlabelled = [t for t in frames if t.subject_marked and not t.subject_label]
        if unlabelled:
            problems.append(
                f"{len(unlabelled)} tile(s) mark a subject without labelling it")

        if not self.subject_crops:
            problems.append(
                "no magnified subject crop. PRD 11 prohibits whole-frame-only "
                "packets: at this source scale a seat inside a full frame is a "
                "few dozen pixels and nothing can be read from it")

        over = [t for t in self.subject_crops
                if t.magnification > cfg.max_subject_magnification]
        if over:
            problems.append(
                f"{len(over)} subject crop(s) magnified beyond "
                f"{cfg.max_subject_magnification}x "
                f"(max {max(t.magnification for t in over):.1f}x); past this "
                f"the tile shows interpolation and reads as detail")

        unstamped = [t for t in self.tiles if not t.pts_stamped]
        if unstamped:
            problems.append(
                f"{len(unstamped)} tile(s) carry no PTS. PRD 11 requires it on "
                f"every tile so no image can be cited without its timestamp")

        times = [t.pts_ms for t in frames]
        if times != sorted(times):
            problems.append(
                "frame tiles are not in chronological order, so a reviewer "
                "reading left to right sees the movement out of sequence")

        leaked = [k for k in self.__dict__ if "reference" in k.lower()]
        if leaked:
            problems.append(
                f"packet carries reference field(s) {leaked}; PRD 15 makes the "
                f"reference manifest readable only at stage 12")

        return problems

    @property
    def valid(self) -> bool:
        return not self.validate()

    def to_row(self) -> dict:
        return {
            "event_id": self.event_id,
            "seat_state": self.seat_state,
            "source_video": self.source_video,
            "source_video_sha256": self.source_video_sha256,
            "start_pts_ms": round(self.start_pts_ms, 3),
            "end_pts_ms": round(self.end_pts_ms, 3),
            "peak_pts_ms": round(self.peak_pts_ms, 3),
            "calibration_version": self.calibration_version,
            "camera_epoch": self.camera_epoch,
            "tiles": [t.to_row() for t in self.tiles],
            "objects": [o.to_row() for o in self.objects],
            "health_state": self.health_state,
            "gate_state": self.gate_state,
            "pose_observations": self.pose_observations,
            "motion": self.motion,
            "priority": self.priority,
            "config_hashes": self.config_hashes,
            "artifact_paths": self.artifact_paths,
            "review_status": self.review_status,
            "valid": self.valid,
            "validation_problems": self.validate(),
        }


def frame_times(start_ms: float, end_ms: float, peak_ms: float,
                cfg: PacketConfig | None = None) -> list[float]:
    """Chronological timestamps for the contact sheet, including the peak.

    The peak is inserted rather than assumed to fall on a grid point. An evenly
    spaced sheet routinely straddles the one moment the event was flagged for,
    and a reviewer then sees the approach and the aftermath but not the reach.
    """
    cfg = cfg or PacketConfig()
    lo = max(start_ms - cfg.pre_roll_ms, 0.0)
    hi = end_ms + cfg.post_roll_ms
    count = cfg.max_frames - 1
    step = (hi - lo) / max(count - 1, 1)
    times = [lo + i * step for i in range(count)]
    if lo <= peak_ms <= hi:
        times.append(peak_ms)
    return sorted(set(round(t, 3) for t in times))[:cfg.max_frames]


def build(event: dict, seat_box: tuple[float, float, float, float] | None,
          objects: list[ObjectEvidence], frame_w: int, frame_h: int,
          cfg: PacketConfig | None = None, tile_px: int = 480,
          health_state: list[str] | None = None,
          pose_observations: dict | None = None,
          config_hashes: dict | None = None) -> EvidencePacket:
    """Compose one packet for one event.

    Describes the tiles rather than rendering them; `tools/render_evidence_packet.py`
    turns this into PNGs. Separating the two means the composition rules are
    testable without a video, which is how the whole-frame-only prohibition
    can be a gate rather than a code review comment.
    """
    cfg = cfg or PacketConfig()
    packet = EvidencePacket(
        event_id=event["event_id"],
        seat_state=event.get("seat_id") or event.get("seat_state", "unattributed"),
        source_video=event.get("source_video", ""),
        source_video_sha256=event.get("source_video_sha256", ""),
        start_pts_ms=float(event["start_pts_ms"]),
        end_pts_ms=float(event["end_pts_ms"]),
        peak_pts_ms=float(event.get("peak_pts_ms", event["start_pts_ms"])),
        calibration_version=event.get("calibration_version", ""),
        camera_epoch=event.get("camera_epoch", ""),
        objects=list(objects),
        health_state=list(health_state or event.get("health_state", [])),
        gate_state=event.get("gate_state", "pass"),
        pose_observations=dict(pose_observations or {}),
        priority=event.get("priority", "normal"),
        config_hashes=dict(config_hashes or {}),
    )

    times = frame_times(packet.start_pts_ms, packet.end_pts_ms,
                        packet.peak_pts_ms, cfg)
    label = packet.seat_state

    for pts in times:
        packet.tiles.append(Tile(
            role=FULL_FRAME, pts_ms=pts,
            x1=0.0, y1=0.0, x2=float(frame_w), y2=float(frame_h),
            subject_marked=seat_box is not None,
            subject_label=label if seat_box is not None else "",
            rendered_w=tile_px, rendered_h=int(tile_px * frame_h / frame_w),
            caption=f"{pts / 1000.0:.2f}s  subject {label}"))

    # The magnified subject crop, at the peak. Bounded so the tile shows the
    # seat at a readable size without pretending to resolution it does not have.
    if seat_box is not None:
        x1, y1, x2, y2 = seat_box
        native = max(x2 - x1, y2 - y1)
        rendered = int(min(tile_px, native * cfg.max_subject_magnification))
        packet.tiles.append(Tile(
            role=SUBJECT_CROP, pts_ms=packet.peak_pts_ms,
            x1=x1, y1=y1, x2=x2, y2=y2,
            subject_marked=True, subject_label=label,
            rendered_w=rendered, rendered_h=rendered,
            caption=f"{packet.peak_pts_ms / 1000.0:.2f}s  {label} magnified "
                    f"{rendered / native:.1f}x from {native:.0f} px native"))

    for observation in objects:
        native = max(observation.x2 - observation.x1,
                     observation.y2 - observation.y1)
        rendered = int(min(tile_px, max(native, 1.0) * cfg.max_subject_magnification))
        packet.tiles.append(Tile(
            role=OBJECT_CROP, pts_ms=packet.peak_pts_ms,
            x1=observation.x1, y1=observation.y1,
            x2=observation.x2, y2=observation.y2,
            subject_marked=True, subject_label=label,
            rendered_w=rendered, rendered_h=rendered,
            caption=f"{observation.cls} {observation.confidence:.2f} "
                    f"({observation.config_id}, {observation.status})"))

    return packet


def summarise(packets: list[EvidencePacket]) -> dict:
    """Packet compliance, reported so malformed packets are counted not hidden."""
    if not packets:
        return {"packets": 0,
                "note": "no evidence packets were built. With zero high-priority "
                        "events this is a valid outcome, and PRD 12 requires the "
                        "report to say what was analysed rather than 'nothing "
                        "happened'."}
    invalid = [p for p in packets if not p.valid]
    reasons: dict[str, int] = {}
    for packet in invalid:
        for problem in packet.validate():
            key = problem.split(";")[0].split(",")[0][:60]
            reasons[key] = reasons.get(key, 0) + 1
    return {
        "packets": len(packets),
        "valid": len(packets) - len(invalid),
        "invalid": len(invalid),
        "invalid_reasons": reasons,
        "with_objects": sum(1 for p in packets if p.objects),
        "with_subject_crop": sum(1 for p in packets if p.subject_crops),
        "median_tiles": sorted(len(p.tiles) for p in packets)[len(packets) // 2],
        "note": "an invalid packet is never sent to a model. PRD 11 prohibits "
                "whole-frame-only packets because a frame showing thirty "
                "candidates and marking none invites a confident description "
                "of the wrong person.",
    }
