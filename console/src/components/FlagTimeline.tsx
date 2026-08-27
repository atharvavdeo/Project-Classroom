import { useMemo } from "react";
import type { Track, Lane } from "../types";
import { LANES } from "../types";
import { LANE_LABEL, SEVERITY_WEIGHT, timecode } from "../lib/format";
import "./FlagTimeline.css";

interface Mark {
  lane: Lane;
  from_ms: number;
  to_ms: number;
  label: string;
  detail: string;
  weight: number;
  /** Faint marks describe a limit of the system rather than an event. */
  faint: boolean;
}

/**
 * Turn one track's spans into lane marks.
 *
 * Object episodes and absences have real extents. Flags mostly do not -- they
 * are whole-track statements -- so they render as a full-width band at low
 * opacity behind anything with a real span. That distinction is load-bearing:
 * a reviewer must be able to tell "this happened at 30-86s" from "this is true
 * of the whole track".
 */
function marksFor(track: Track, duration: number): Mark[] {
  const marks: Mark[] = [];

  for (const ep of track.episodes) {
    marks.push({
      lane: "object",
      from_ms: ep.from_ms,
      to_ms: ep.to_ms,
      label: "object at hands",
      detail: `${timecode(ep.from_ms)} to ${timecode(ep.to_ms)}`,
      weight: 1,
      faint: false,
    });
  }

  for (const ev of track.orientation_events) {
    marks.push({
      lane: "orientation",
      from_ms: ev.from_ms,
      to_ms: ev.to_ms,
      label: ev.label + (ev.direction ? ` ${ev.direction}` : ""),
      detail: ev.guardrail,
      weight: 0.8,
      faint: ev.dashboard_action === "show_only_with_second_modality",
    });
  }

  for (const ab of track.absences) {
    marks.push({
      lane: "presence",
      from_ms: ab.from_ms,
      to_ms: ab.to_ms,
      label: "absent from frame",
      detail: `${timecode(ab.from_ms)} to ${timecode(ab.to_ms)}`,
      weight: 0.78,
      faint: false,
    });
  }

  // Whole-track flags, drawn across the observed span rather than the whole
  // recording -- claiming a flag covers time the person was never seen would
  // be a lie of extent.
  const from = track.first_seen_ms ?? 0;
  const to = track.last_seen_ms ?? duration;
  for (const flag of track.flags) {
    if (flag.lane === "object" && track.episodes.length) continue;
    marks.push({
      lane: flag.lane,
      from_ms: from,
      to_ms: to,
      label: flag.flag,
      detail: flag.evidence,
      weight: SEVERITY_WEIGHT[flag.severity] ?? 0.4,
      faint: true,
    });
  }

  return marks;
}

export function FlagTimeline({
  track,
  duration_ms,
  playheadMs,
  onSeek,
  compact = false,
}: {
  track: Track;
  duration_ms: number;
  playheadMs?: number;
  onSeek?: (ms: number) => void;
  compact?: boolean;
}) {
  const marks = useMemo(
    () => marksFor(track, duration_ms),
    [track, duration_ms],
  );
  const byLane = useMemo(() => {
    const m = new Map<Lane, Mark[]>();
    for (const lane of LANES) m.set(lane, []);
    for (const mark of marks) m.get(mark.lane)!.push(mark);
    return m;
  }, [marks]);

  const pct = (ms: number) =>
    duration_ms > 0 ? Math.max(0, Math.min(100, (ms / duration_ms) * 100)) : 0;

  return (
    <div className={`tl ${compact ? "tl--compact" : ""}`}>
      {LANES.map((lane) => {
        const laneMarks = byLane.get(lane)!;
        return (
          <div className="tl__lane" key={lane}>
            <span className="tl__label">{LANE_LABEL[lane]}</span>
            <div
              className="tl__track"
              onClick={(e) => {
                if (!onSeek || duration_ms <= 0) return;
                const r = e.currentTarget.getBoundingClientRect();
                onSeek(((e.clientX - r.left) / r.width) * duration_ms);
              }}
            >
              {laneMarks.length === 0 && (
                <span className="tl__none">no marks</span>
              )}
              {laneMarks.map((m, i) => {
                const left = pct(m.from_ms);
                const width = Math.max(pct(m.to_ms) - left, 0.4);
                return (
                  <i
                    key={`${m.label}-${i}`}
                    className={m.faint ? "is-faint" : ""}
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      height: `${Math.round(m.weight * 100)}%`,
                      background: `var(--lane-${lane})`,
                    }}
                    title={`${m.label}\n${m.detail}`}
                  />
                );
              })}
              {playheadMs !== undefined && (
                <span
                  className="tl__playhead"
                  style={{ left: `${pct(playheadMs)}%` }}
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
