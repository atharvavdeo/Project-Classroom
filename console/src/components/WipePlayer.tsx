import { useCallback, useEffect, useRef, useState } from "react";
import type { OverlayBundle, OverlayFrame } from "../overlay";
import { nearestFrame } from "../overlay";
import type { Episode } from "../lib/deviation";
import { heatColor, type HeatField } from "../lib/heatmap";
import { timecode } from "../lib/format";
import "./WipePlayer.css";

export interface Layers {
  boxes: boolean;
  skeletons: boolean;
  ids: boolean;
  objects: boolean;
  wrists: boolean;
  facing: boolean;
  /** Mark people while they are mid-turn, with the arithmetic that says so. */
  turning: boolean;
}

/**
 * What is drawn before anyone touches a control.
 *
 * Skeletons are off. With 236 tracked people a full skeleton pass turned the
 * frame into a wireframe mesh -- every joint of every person, most of whom the
 * system said nothing about. The layer is still there for looking at one
 * person; it is just not what you want to open on.
 */
export const DEFAULT_LAYERS: Layers = {
  boxes: true,
  skeletons: false,
  ids: true,
  objects: true,
  wrists: false,
  facing: true,
  turning: true,
};

export const LAYER_LABEL: Record<keyof Layers, string> = {
  boxes: "person boxes",
  skeletons: "skeletons",
  ids: "track ids",
  objects: "object boxes",
  wrists: "wrist markers",
  facing: "head direction",
  turning: "turned away",
};

/** Resist progressively past a boundary instead of stopping dead. A hard stop
 *  reads as frozen; continuous resistance reads as "responsive, but there is
 *  nothing more here". */
function rubberband(overshoot: number, dimension: number, constant = 0.55) {
  return (
    (overshoot * dimension * constant) /
    (dimension + constant * Math.abs(overshoot))
  );
}

function css(name: string, fallback: string) {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return v || fallback;
}

/**
 * Raw footage on the left, our marks on the right, split by a handle the
 * reviewer drags.
 *
 * A blend slider was the obvious alternative and it is worse: at 50% opacity
 * you cannot tell whether a faint edge is our box or the desk bezel, which is
 * precisely the judgement this component exists to support. A hard wipe keeps
 * both sides at full fidelity and puts the comparison on the boundary, where
 * the eye is most sensitive.
 *
 * The marks are drawn on a canvas rather than baked into a second video, so
 * layers can be switched off and a box can say which track it belonged to.
 */
const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4];
const FPS = 25;

/** Steps, symmetric in both directions. Frame steps come first because the
 *  point of this player is landing on the instant a detection fired, not
 *  watching the recording. */
const STEPS: { label: string; seconds: number }[] = [
  { label: "1f", seconds: 1 / FPS },
  { label: "5f", seconds: 5 / FPS },
  { label: "1s", seconds: 1 },
  { label: "5s", seconds: 5 },
  { label: "10s", seconds: 10 },
];

export function WipePlayer({
  videoSrc,
  overlay,
  layers,
  focusTrack,
  onlyTracks,
  turns,
  heat,
  onTimeUpdate,
  seekToMs,
  below,
  inset,
}: {
  videoSrc: string;
  overlay: OverlayBundle | null;
  layers: Layers;
  /** Every measured turn in the recording. The player marks the ones that
   *  contain the current instant, so the mark appears while the turn is
   *  happening and leaves when it ends -- a box that stayed on for the whole
   *  recording would say nothing about *when*. */
  turns?: Episode[];
  /** Accumulated spatial density, drawn under every other mark. Null draws
   *  none. */
  heat?: HeatField | null;
  focusTrack?: number | null;
  /** Draw only these people. Null draws everyone.
   *
   *  The default is the set the system actually said something about. Drawing
   *  all 236 tracked people put a box on every head in the hall, which is the
   *  same picture whether the run found one thing or nothing -- the marks stop
   *  carrying information at that density. */
  onlyTracks?: Set<number> | null;
  onTimeUpdate?: (ms: number) => void;
  seekToMs?: number | null;
  /** Rendered flush under the picture, inside the same frame. The timeline
   *  belongs to the video, so it is not allowed to drift into a separate
   *  section further down the page. */
  below?: React.ReactNode;
  /** Rendered over the picture, bottom-right: the close-up of whatever the
   *  reviewer selected, so the wide frame and the evidence for it are in one
   *  glance instead of two scroll positions. */
  inset?: React.ReactNode;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [now, setNow] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState(1);

  // Wipe position as a fraction. Kept in a ref for the draw loop and mirrored
  // into state only for the handle's own transform, so dragging never
  // re-renders the tree.
  const wipeRef = useRef(0.5);
  const [wipe, setWipe] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const dragRef = useRef<{ id: number; grabDx: number } | null>(null);
  const layersRef = useRef(layers);
  const focusRef = useRef<number | null | undefined>(focusTrack);
  const onlyRef = useRef<Set<number> | null | undefined>(onlyTracks);
  const turnsRef = useRef<Episode[] | undefined>(turns);
  const heatRef = useRef<HeatField | null | undefined>(heat);
  // The scaled-up density is rebuilt only when the field itself changes;
  // rebuilding it every frame would put a cols x rows ImageData allocation
  // inside the draw loop.
  const heatTileRef = useRef<{ field: HeatField; canvas: HTMLCanvasElement } | null>(null);
  layersRef.current = layers;
  focusRef.current = focusTrack;
  onlyRef.current = onlyTracks;
  turnsRef.current = turns;
  heatRef.current = heat;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !overlay) return;

    const w = overlay.width;
    const h = overlay.height;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);

    const frame: OverlayFrame | null = nearestFrame(
      overlay,
      video.currentTime * 1000,
    );
    if (!frame) return;

    // Everything we draw is confined to the right of the handle, so the left
    // half stays the untouched decoded frame.
    const splitX = wipeRef.current * w;
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, w - splitX, h);
    ctx.clip();

    const L = layersRef.current;
    const focus = focusRef.current;
    const accent = css("--accent-2", "#8acfe6");
    const objectColor = css("--lane-object", "#e5825c");
    const dim = css("--slate", "#93a1ac");

    // Density first, so every other mark sits on top of it rather than being
    // washed out by it.
    const field = heatRef.current;
    if (field && field.peak > 0) {
      let tile = heatTileRef.current;
      if (!tile || tile.field !== field) {
        const off = document.createElement("canvas");
        off.width = field.cols;
        off.height = field.rows;
        const octx = off.getContext("2d");
        if (octx) {
          const img = octx.createImageData(field.cols, field.rows);
          for (let i = 0; i < field.cells.length; i += 1) {
            const [r, g, b, a] = heatColor(field.cells[i]);
            img.data[i * 4] = r;
            img.data[i * 4 + 1] = g;
            img.data[i * 4 + 2] = b;
            img.data[i * 4 + 3] = a;
          }
          octx.putImageData(img, 0, 0);
          tile = { field, canvas: off };
          heatTileRef.current = tile;
        }
      }
      if (tile) {
        // Smoothing on: the grid is 16px cells and the field is already
        // blurred, so bilinear upscaling is the intended look. Nearest
        // neighbour would draw visible squares and imply a precision the
        // measurement does not have.
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(tile.canvas, 0, 0, w, h);
      }
    }

    const only = onlyRef.current;

    // Turns containing this instant, by person. Built per draw rather than
    // cached: the list is a handful of entries and the alternative is an index
    // that has to be invalidated whenever the recording changes.
    const nowMs = video.currentTime * 1000;
    const active = new Map<number, Episode>();
    if (L.turning) {
      for (const e of turnsRef.current ?? []) {
        if (nowMs >= e.from_ms && nowMs <= e.to_ms) active.set(e.track_id, e);
      }
    }
    const turnColor = css("--lane-orientation", "#d9a84a");

    for (const person of frame.p) {
      // A person nobody is interested in is not drawn faintly, they are not
      // drawn. Faint marks at this density still read as clutter.
      const turn = active.get(person.i);
      // Someone mid-turn is drawn whether or not they made the object lane's
      // shortlist. That lane is about things in hands; this one is about where
      // a head went, and a person can be interesting to one and not the other.
      if (only && !only.has(person.i) && focus !== person.i && !turn) continue;

      const isFocus = focus === undefined || focus === null || focus === person.i;
      ctx.globalAlpha = isFocus ? 1 : 0.28;
      const [x1, y1, x2, y2] = person.b;

      if (turn) {
        // Drawn first and thick, so it reads as the reason this person is
        // marked rather than as decoration on a box drawn for another reason.
        ctx.globalAlpha = 1;
        ctx.strokeStyle = turnColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1 - 2, y1 - 2, x2 - x1 + 4, y2 - y1 + 4);

        // The metrics, on the picture. A reviewer should not have to match a
        // box against a row in a table below to find out what was measured.
        const lines = [
          `#${person.i} turned ${turn.away}`,
          `${turn.peak_z.toFixed(1)}x own spread · ${((turn.to_ms - turn.from_ms) / 1000).toFixed(1)}s`,
          turn.with_track != null ? `with #${turn.with_track}` : null,
        ].filter(Boolean) as string[];

        ctx.font = "600 12px ui-monospace, Menlo, monospace";
        const boxW = Math.max(...lines.map((s) => ctx.measureText(s).width)) + 10;
        const boxH = lines.length * 14 + 6;
        // Above the head when there is room, otherwise inside the top of the
        // box: a label clipped off the top of the frame is a label nobody
        // reads.
        const ly = y1 - boxH - 4 >= 0 ? y1 - boxH - 4 : y1 + 2;
        ctx.fillStyle = "rgba(0,0,0,.72)";
        ctx.fillRect(x1 - 2, ly, boxW, boxH);
        ctx.fillStyle = turnColor;
        lines.forEach((s, i) => ctx.fillText(s, x1 + 3, ly + 15 + i * 14));
      }

      if (L.boxes) {
        ctx.strokeStyle = isFocus ? accent : dim;
        ctx.lineWidth = isFocus ? 2 : 1;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      if (L.ids) {
        const label = `#${person.i}`;
        ctx.font = "600 13px ui-monospace, Menlo, monospace";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,.62)";
        ctx.fillRect(x1, Math.max(y1 - 17, 0), tw + 8, 16);
        ctx.fillStyle = isFocus ? accent : dim;
        ctx.fillText(label, x1 + 4, Math.max(y1 - 5, 11));
      }

      if (L.skeletons) {
        ctx.strokeStyle = isFocus ? accent : dim;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (const [a, b] of overlay.bones) {
          const ax = person.j[a * 2];
          const ay = person.j[a * 2 + 1];
          const bx = person.j[b * 2];
          const by = person.j[b * 2 + 1];
          // -1 is the "not visible" sentinel; a bone with a missing end is
          // simply not drawn rather than drawn to the origin.
          if (ax < 0 || ay < 0 || bx < 0 || by < 0) continue;
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
        }
        ctx.stroke();
      }

      if (L.wrists) {
        for (const idx of [9, 10]) {
          const wx = person.j[idx * 2];
          const wy = person.j[idx * 2 + 1];
          if (wx < 0 || wy < 0) continue;
          ctx.beginPath();
          ctx.arc(wx, wy, 4, 0, Math.PI * 2);
          ctx.fillStyle = isFocus ? accent : dim;
          ctx.fill();
        }
      }

      // Head direction, as an arrow off the head.
      //
      // This is the pipeline's own coarse `facing` label, not a computed gaze
      // ray, and it is drawn at four directions because that is all the label
      // distinguishes. Gaze is not recoverable at this source scale -- drawing
      // a precise-looking ray would assert a measurement nothing made. Where
      // the pipeline could not resolve facing, nothing is drawn, because "not
      // resolvable" and "facing the camera" must not look the same.
      if (L.facing && person.f && person.f !== "not_resolvable_at_source_scale") {
        const nose = [person.j[0], person.j[1]];
        const lEar = [person.j[6], person.j[7]];
        const rEar = [person.j[8], person.j[9]];
        const ears = [lEar, rEar].filter(([x, y]) => x >= 0 && y >= 0);

        // Anchor on the head: the ear midpoint if we have one, else the nose.
        let hx = nose[0];
        let hy = nose[1];
        if (ears.length) {
          hx = ears.reduce((s, e) => s + e[0], 0) / ears.length;
          hy = ears.reduce((s, e) => s + e[1], 0) / ears.length;
        }
        if (hx >= 0 && hy >= 0) {
          const span = Math.max(x2 - x1, 24);
          const len = span * 0.55;
          const dir: Record<string, [number, number]> = {
            frontal: [0, 1],
            downward: [0, 1],
            turned_left: [-1, 0.15],
            turned_right: [1, 0.15],
          };
          const [dx, dy] = dir[person.f] ?? [0, 1];
          const tx = hx + dx * len;
          const ty = hy + dy * len * (person.f === "downward" ? 1 : 0.5);

          ctx.strokeStyle = person.f === "downward" ? css("--lane-orientation", "#d9a84a") : accent;
          ctx.fillStyle = ctx.strokeStyle;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(hx, hy);
          ctx.lineTo(tx, ty);
          ctx.stroke();

          const angle = Math.atan2(ty - hy, tx - hx);
          const head = Math.max(span * 0.13, 5);
          ctx.beginPath();
          ctx.moveTo(tx, ty);
          ctx.lineTo(
            tx - head * Math.cos(angle - 0.42),
            ty - head * Math.sin(angle - 0.42),
          );
          ctx.lineTo(
            tx - head * Math.cos(angle + 0.42),
            ty - head * Math.sin(angle + 0.42),
          );
          ctx.closePath();
          ctx.fill();
        }
      }

      if (L.objects && person.o) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = objectColor;
        for (const obj of person.o) {
          const [ox1, oy1, ox2, oy2] = obj.b;
          ctx.strokeRect(ox1, oy1, ox2 - ox1, oy2 - oy1);
        }
      }
    }

    ctx.globalAlpha = 1;
    ctx.restore();
  }, [overlay]);

  // The time callback lives in a ref so it can never be an effect dependency.
  // Passed as one it re-created the rAF loop on every render, and since the
  // callback itself sets state each frame that was a render storm that
  // cancelled the loop faster than it could schedule itself.
  const timeCbRef = useRef(onTimeUpdate);
  timeCbRef.current = onTimeUpdate;

  // One display-synced loop, owned by the component for its whole life.
  // `draw` is read through a ref for the same reason: the loop must survive
  // an overlay arriving, not be torn down and rebuilt by it.
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    let raf = 0;
    let alive = true;
    // Publishing the playhead at ~12 Hz is well under a frame but far above
    // the eye's threshold for a moving cursor, and it keeps the 236-row
    // person list from re-rendering sixty times a second.
    let lastPublish = 0;

    const tick = (now: number) => {
      if (!alive) return;
      drawRef.current();
      const v = videoRef.current;
      if (v && timeCbRef.current && now - lastPublish > 80) {
        lastPublish = now;
        timeCbRef.current(v.currentTime * 1000);
      }
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);

    // rAF is not a complete answer. A hidden tab, a background window and a
    // paused video all stop it, and in every one of those cases the marks
    // still have to be correct the moment the picture changes -- a seek while
    // paused being the obvious one. So the same draw is also bound to the
    // video's own events, which fire regardless of compositing.
    const video = videoRef.current;
    const repaint = () => drawRef.current();
    const events = ["seeked", "loadeddata", "timeupdate", "play", "pause"];
    for (const name of events) video?.addEventListener(name, repaint);

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      for (const name of events) video?.removeEventListener(name, repaint);
    };
  }, []);

  // Repaint when anything that changes the picture changes, so a layer toggle
  // or a wipe drag is reflected immediately even with the loop stalled.
  useEffect(() => {
    draw();
  }, [draw, layers, focusTrack, wipe]);

  useEffect(() => {
    if (seekToMs === null || seekToMs === undefined) return;
    const v = videoRef.current;
    if (v) v.currentTime = seekToMs / 1000;
  }, [seekToMs]);

  const applyWipe = useCallback((fraction: number) => {
    const stage = stageRef.current;
    const width = stage?.clientWidth ?? 1;
    let next = fraction;
    if (next < 0) next = rubberband(next, width) / width;
    if (next > 1) next = 1 + rubberband(next - 1, width) / width;
    next = Math.max(-0.04, Math.min(1.04, next));
    wipeRef.current = Math.max(0, Math.min(1, next));
    setWipe(next);
  }, []);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const stage = stageRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const handleX = rect.left + wipeRef.current * rect.width;
    // Respect where they grabbed: snapping the handle to the pointer would
    // break the illusion on the first pixel of movement.
    dragRef.current = { id: e.pointerId, grabDx: e.clientX - handleX };
    (e.target as Element).setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    const stage = stageRef.current;
    if (!drag || drag.id !== e.pointerId || !stage) return;
    const rect = stage.getBoundingClientRect();
    applyWipe((e.clientX - drag.grabDx - rect.left) / rect.width);
  };

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.id !== e.pointerId) return;
    dragRef.current = null;
    // Settle back inside the bounds if the rubber band was stretched.
    applyWipe(Math.max(0, Math.min(1, wipeRef.current)));
  };

  const toggle = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      void v.play();
      setPlaying(true);
    } else {
      v.pause();
      setPlaying(false);
    }
  }, []);

  /** Step by a signed number of seconds. A sub-second step pauses first:
   *  stepping while playing fights the reviewer, because playback carries them
   *  straight off the frame they aimed at. */
  const nudge = useCallback((delta: number) => {
    const v = videoRef.current;
    if (!v) return;
    if (Math.abs(delta) < 0.5 && !v.paused) {
      v.pause();
      setPlaying(false);
    }
    v.currentTime = Math.max(
      0,
      Math.min(v.duration || Number.MAX_SAFE_INTEGER, v.currentTime + delta),
    );
  }, []);

  useEffect(() => {
    const v = videoRef.current;
    if (v) v.playbackRate = speed;
  }, [speed]);

  // The readout under the picture is driven by the element's own events, not
  // by the rAF loop: it has to stay right while paused, which is exactly when
  // rAF is least reliable.
  //
  // The playhead published to the parent rides on these events too. It used to
  // come only from the rAF loop, and the timeline marker underneath sat at
  // zero through a whole sequence of frame steps -- rAF does not run while the
  // page is not compositing, and a frame step is precisely a seek made while
  // paused. Measured: the clock read 00:05.0 while the ribbon still said
  // 00:00.0.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const tick = () => {
      setNow(v.currentTime * 1000);
      timeCbRef.current?.(v.currentTime * 1000);
    };
    const meta = () => setDuration((v.duration || 0) * 1000);
    const events = ["timeupdate", "seeked", "seeking"];
    for (const e of events) v.addEventListener(e, tick);
    v.addEventListener("loadedmetadata", meta);
    v.addEventListener("durationchange", meta);
    return () => {
      for (const e of events) v.removeEventListener(e, tick);
      v.removeEventListener("loadedmetadata", meta);
      v.removeEventListener("durationchange", meta);
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "SELECT") return;
      const v = videoRef.current;
      if (!v) return;
      if (e.key === " ") {
        e.preventDefault();
        toggle();
      } else if (e.key === "ArrowLeft") {
        v.currentTime = Math.max(0, v.currentTime - 1);
      } else if (e.key === "ArrowRight") {
        v.currentTime = v.currentTime + 1;
      } else if (e.key === ",") {
        v.currentTime = Math.max(0, v.currentTime - 1 / 25);
      } else if (e.key === ".") {
        v.currentTime = v.currentTime + 1 / 25;
      } else if (e.key === "\\") {
        applyWipe(0.5);
      } else if (e.key === "a") {
        applyWipe(wipeRef.current > 0.5 ? 0 : 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, applyWipe]);

  const pct = `${(wipe * 100).toFixed(3)}%`;

  return (
    <div className="wipe">
      <div
        className="wipe__stage"
        ref={stageRef}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <video
          ref={videoRef}
          src={videoSrc}
          playsInline
          muted
          preload="metadata"
          onClick={toggle}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
        <canvas ref={canvasRef} />

        {inset}

        <span className="wipe__tag wipe__tag--l">raw</span>
        <span className="wipe__tag wipe__tag--r">annotated</span>

        <div
          className="wipe__handle"
          style={{ left: pct }}
          onPointerDown={onPointerDown}
          role="slider"
          tabIndex={0}
          aria-label="Reveal annotations"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(wipe * 100)}
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
              e.preventDefault();
              applyWipe(wipeRef.current + (e.key === "ArrowLeft" ? -0.02 : 0.02));
            }
          }}
        >
          <i />
        </div>

        {/* Transport sits on the picture, where the reviewer's eye already is.
            A control strip in a panel below means looking away from the frame
            to press the button that changes the frame. */}
        <div className="wipe__transport" onPointerDown={(e) => e.stopPropagation()}>
          <button
            type="button"
            className="wipe__play"
            onClick={toggle}
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? (
              <svg viewBox="0 0 12 14" aria-hidden="true">
                <rect x="1" y="1" width="3.4" height="12" rx="1" />
                <rect x="7.6" y="1" width="3.4" height="12" rx="1" />
              </svg>
            ) : (
              <svg viewBox="0 0 12 14" aria-hidden="true">
                <path d="M2 1.4v11.2a1 1 0 0 0 1.53.85l8.2-5.6a1 1 0 0 0 0-1.7l-8.2-5.6A1 1 0 0 0 2 1.4Z" />
              </svg>
            )}
          </button>

          <div className="wipe__steps">
            {[...STEPS].reverse().map((s) => (
              <button
                key={`b${s.label}`}
                type="button"
                title={`Back ${s.label}`}
                onClick={() => nudge(-s.seconds)}
              >
                −{s.label}
              </button>
            ))}
            {STEPS.map((s) => (
              <button
                key={`f${s.label}`}
                type="button"
                title={`Forward ${s.label}`}
                onClick={() => nudge(s.seconds)}
              >
                +{s.label}
              </button>
            ))}
          </div>

          <label className="wipe__speed">
            <span className="mono">speed</span>
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

          <span className="wipe__clock mono">
            {timecode(now)} / {timecode(duration)}
          </span>
        </div>
      </div>

      {below}
    </div>
  );
}
