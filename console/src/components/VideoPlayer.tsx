import { useCallback, useEffect, useRef, useState } from "react";
import { timecode } from "../lib/format";
import "./VideoPlayer.css";

const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4];
const FPS = 25;

/** Jump buttons, in the order a reviewer reaches for them. Frame steps first,
 *  because the whole point of this player is stepping through the moment a
 *  detection fired rather than watching the recording. */
const JUMPS: { label: string; seconds: number }[] = [
  { label: "1f", seconds: 1 / FPS },
  { label: "5f", seconds: 5 / FPS },
  { label: "1s", seconds: 1 },
  { label: "5s", seconds: 5 },
  { label: "10s", seconds: 10 },
];

export interface VideoHandleProps {
  src: string;
  /** Marks to render on the scrubber, as {ms, kind}. */
  marks?: { ms: number; kind: "supported" | "proposal" }[];
  seekToMs?: number | null;
  onTimeUpdate?: (ms: number) => void;
  overlay?: React.ReactNode;
}

/**
 * A player built for inspection, not for watching.
 *
 * Everything a reviewer needs to land on one frame: symmetric step-back and
 * step-forward at five granularities, speed control, and a scrubber that shows
 * where the evidence is so seeking is aimed rather than blind.
 */
export function VideoPlayer({
  src,
  marks = [],
  seekToMs,
  onTimeUpdate,
  overlay,
}: VideoHandleProps) {
  const ref = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [now, setNow] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState(1);

  const cbRef = useRef(onTimeUpdate);
  cbRef.current = onTimeUpdate;

  const nudge = useCallback((seconds: number) => {
    const v = ref.current;
    if (!v) return;
    // Pause on a step: stepping while playing fights the reviewer, because
    // playback immediately carries them off the frame they aimed at.
    if (Math.abs(seconds) < 0.5 && !v.paused) v.pause();
    v.currentTime = Math.max(0, Math.min(v.duration || 0, v.currentTime + seconds));
  }, []);

  const toggle = useCallback(() => {
    const v = ref.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }, []);

  useEffect(() => {
    if (seekToMs === null || seekToMs === undefined) return;
    const v = ref.current;
    if (v) v.currentTime = seekToMs / 1000;
  }, [seekToMs]);

  useEffect(() => {
    const v = ref.current;
    if (v) v.playbackRate = speed;
  }, [speed]);

  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    const tick = () => {
      setNow(v.currentTime * 1000);
      cbRef.current?.(v.currentTime * 1000);
    };
    const meta = () => setDuration((v.duration || 0) * 1000);
    v.addEventListener("timeupdate", tick);
    v.addEventListener("seeked", tick);
    v.addEventListener("loadedmetadata", meta);
    v.addEventListener("play", () => setPlaying(true));
    v.addEventListener("pause", () => setPlaying(false));
    return () => {
      v.removeEventListener("timeupdate", tick);
      v.removeEventListener("seeked", tick);
      v.removeEventListener("loadedmetadata", meta);
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName)) return;
      if (e.key === " ") {
        e.preventDefault();
        toggle();
      } else if (e.key === "ArrowLeft") nudge(e.shiftKey ? -5 : -1);
      else if (e.key === "ArrowRight") nudge(e.shiftKey ? 5 : 1);
      else if (e.key === ",") nudge(-1 / FPS);
      else if (e.key === ".") nudge(1 / FPS);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, nudge]);

  const pct = duration > 0 ? (now / duration) * 100 : 0;

  return (
    <div className="vp">
      <div className="vp__stage">
        <video ref={ref} src={src} playsInline muted onClick={toggle} />
        {overlay}
      </div>

      <div
        className="vp__scrub"
        onPointerDown={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const v = ref.current;
          if (v && v.duration)
            v.currentTime = ((e.clientX - r.left) / r.width) * v.duration;
        }}
      >
        {marks.map((m, i) => (
          <i
            key={i}
            className={m.kind === "supported" ? "is-strong" : ""}
            style={{ left: `${duration > 0 ? (m.ms / duration) * 100 : 0}%` }}
          />
        ))}
        <span className="vp__played" style={{ width: `${pct}%` }} />
        <span className="vp__head" style={{ left: `${pct}%` }} />
      </div>

      <div className="vp__bar">
        <button type="button" className="vp__play" onClick={toggle}>
          {playing ? "Pause" : "Play"}
        </button>

        <div className="vp__jumps">
          {[...JUMPS].reverse().map((j) => (
            <button
              key={`b${j.label}`}
              type="button"
              title={`Back ${j.label}`}
              onClick={() => nudge(-j.seconds)}
            >
              −{j.label}
            </button>
          ))}
          {JUMPS.map((j) => (
            <button
              key={`f${j.label}`}
              type="button"
              title={`Forward ${j.label}`}
              onClick={() => nudge(j.seconds)}
            >
              +{j.label}
            </button>
          ))}
        </div>

        <label className="vp__speed">
          Speed
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>

        <span className="vp__time mono">
          {timecode(now)} / {timecode(duration)}
        </span>
      </div>
    </div>
  );
}
