import { useEffect, useMemo, useRef, useState } from "react";
import type { Bundle, Track } from "../types";
import type { CropManifest, Decision } from "../review";
import type { OverlayBundle } from "../overlay";
import {
  WipePlayer,
  DEFAULT_LAYERS,
  LAYER_LABEL,
  type Layers,
} from "../components/WipePlayer";
import { StateChip } from "../components/StateChip";
import { FindingsTable } from "../components/FindingsTable";
import { Spotlight } from "../components/Spotlight";
import { api, type ReviewInterval } from "./../api";
import { deviationEpisodes } from "../lib/deviation";
import { buildHeat, HEAT_LABEL, type HeatSource } from "../lib/heatmap";
import { subjectName, timecode } from "../lib/format";
import { flagLabel } from "../lib/humanize";
import "./VideoScreen.css";

/** Overlay geometry is in source pixels, and the bundle now carries the frame
 *  it was measured against. This used to be hard-coded to 1280x720 next to the
 *  media -- correct for most of the corpus and wrong for the 640x480 recording,
 *  where every box landed at half scale in the top-left quadrant. A bundle
 *  without the size is refused rather than guessed at. */
const FALLBACK = { width: 1280, height: 720 };

interface Mark {
  from_ms: number;
  to_ms: number;
  track_id: number;
  detail: string;
}

interface Kind {
  id: string;
  label: string;
  color: string;
  people: number;
  marks: Mark[];
  /** On unless the reviewer says otherwise. */
  defaultOn: boolean;
  note: string;
}

/**
 * Where each kind of finding actually lands in time.
 *
 * Object evidence has real extents (the episodes), and so do absences and
 * orientation events. Most flags do not -- they are statements about a whole
 * track -- and those are drawn across the span the person was actually on
 * camera rather than the whole recording. Claiming a flag covers time the
 * person was never visible would be a lie of extent.
 */
function buildKinds(
  tracks: Track[],
  crops: CropManifest,
  duration: number,
): Kind[] {
  const kinds = new Map<string, Kind>();
  const add = (
    id: string,
    label: string,
    color: string,
    defaultOn: boolean,
    note: string,
  ) => {
    if (!kinds.has(id)) {
      kinds.set(id, {
        id,
        label,
        color,
        people: 0,
        marks: [],
        defaultOn,
        note,
      });
    }
    return kinds.get(id)!;
  };

  for (const t of tracks) {
    const crop = crops[String(t.track_id)];
    const from = t.first_seen_ms ?? 0;
    const to = t.last_seen_ms ?? duration;

    // The one lane that is a positive result rather than a proposal: the
    // referee looked at this crop and named the object. Marked by default,
    // because it is the finding a reviewer came here for.
    if (crop?.key_supported) {
      const k = add(
        "confirmed",
        "Confirmed by the second model",
        "var(--s-confirm)",
        true,
        "SAM 3 looked at this crop and named the object.",
      );
      k.people += 1;
      k.marks.push({
        from_ms: crop.key_pts_ms,
        to_ms: crop.key_pts_ms + 400,
        track_id: t.track_id,
        detail: `#${t.track_id} · ${timecode(crop.key_pts_ms)}`,
      });
    }

    if (t.state === "review_candidate") {
      const k = add(
        "review",
        "Sent for review",
        "var(--s-review)",
        true,
        "The machine routed this person to a human. It is not a verdict.",
      );
      k.people += 1;
      const spans = t.episodes.length ? t.episodes : [{ from_ms: from, to_ms: to }];
      for (const s of spans) {
        k.marks.push({
          from_ms: s.from_ms,
          to_ms: s.to_ms,
          track_id: t.track_id,
          detail: `#${t.track_id} · ${timecode(s.from_ms)}–${timecode(s.to_ms)}`,
        });
      }
    }

    for (const flag of t.flags) {
      const k = add(
        `flag:${flag.flag}`,
        flagLabel(flag.flag),
        `var(--lane-${flag.lane})`,
        flag.severity === "high",
        flag.why || flag.evidence,
      );
      k.people += 1;

      const spans =
        flag.lane === "object" && t.episodes.length
          ? t.episodes
          : flag.lane === "orientation" && t.orientation_events.length
            ? t.orientation_events
            : flag.lane === "presence" && t.absences.length
              ? t.absences
              : [{ from_ms: from, to_ms: to }];

      for (const s of spans) {
        k.marks.push({
          from_ms: s.from_ms,
          to_ms: s.to_ms,
          track_id: t.track_id,
          detail: `#${t.track_id} · ${flag.evidence}`,
        });
      }
    }
  }

  return [...kinds.values()].sort((a, b) => {
    if (a.defaultOn !== b.defaultOn) return a.defaultOn ? -1 : 1;
    return b.people - a.people;
  });
}

export function VideoScreen({
  bundle,
  videoId,
  crops,
  cropBase,
  videoSrc,
  overlaySrc,
  decisions,
}: {
  bundle: Bundle;
  /** The store's id for this recording, used for reviewer marks. */
  videoId: string;
  crops: CropManifest;
  cropBase: string;
  videoSrc: string;
  overlaySrc: string;
  decisions: Record<number, Decision>;
}) {
  const [overlay, setOverlay] = useState<OverlayBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [layers, setLayers] = useState<Layers>(DEFAULT_LAYERS);
  const [focus, setFocus] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState(0);
  const [seek, setSeek] = useState<number | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  // What the reviewer selected from the findings table, shown as a close-up
  // over the frame. Null until they pick something -- opening with a spotlight
  // would assert that one of the rows matters more than the others.
  const [spot, setSpot] = useState<
    { track_id: number; label: string; standing: string } | null
  >(null);
  const [chosen, setChosen] = useState<Set<string> | null>(null);

  useEffect(() => {
    let live = true;
    fetch(overlaySrc)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((doc) => {
        if (!live) return;
        if (!doc.width || !doc.height) {
          // An older bundle, built before the size was recorded. Say so
          // instead of drawing boxes that may be in the wrong place.
          setError(
            "This overlay was built before frame size was recorded. Rebuild it " +
              "with tools/build_overlay_bundle.py — drawing it against an " +
              "assumed size would put the boxes in the wrong place.",
          );
          setOverlay({ ...doc, ...FALLBACK });
          return;
        }
        setOverlay(doc as OverlayBundle);
      })
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [overlaySrc]);

  // A person a reviewer said was not a violation is gone from here. Leaving
  // an overturned detection on the timeline would ask the same question twice
  // and quietly disagree with the answer.
  const live = useMemo(
    () =>
      bundle.tracks.filter(
        (t) => decisions[t.track_id] !== "human_dismissed",
      ),
    [bundle.tracks, decisions],
  );

  const kinds = useMemo(
    () => buildKinds(live, crops, bundle.duration_ms),
    [live, crops, bundle.duration_ms],
  );

  const on = useMemo(
    () => chosen ?? new Set(kinds.filter((k) => k.defaultOn).map((k) => k.id)),
    [chosen, kinds],
  );

  const toggle = (id: string) => {
    const next = new Set(on);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setChosen(next);
  };

  const shown = kinds.filter((k) => on.has(k.id));

  const people: Track[] = useMemo(
    () => [...live].sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0)),
    [live],
  );

  // Who the overlay draws. Everyone the system said something about, plus
  // anyone a reviewer has answered for -- not all 236 tracked people, which
  // put a box on every head in the hall and made a busy run and a quiet one
  // look identical.
  const [drawAll, setDrawAll] = useState(false);
  // Turns are measured from the overlay geometry, not read from the run: the
  // pipeline's own orientation events need a yaw baseline that fails at this
  // source scale for most people. See lib/deviation.ts.
  const turns = useMemo(() => {
    const dismissed = new Set(
      bundle.tracks
        .filter((t) => decisions[t.track_id] === "human_dismissed")
        .map((t) => t.track_id),
    );
    return deviationEpisodes(overlay).episodes.filter(
      (e) => !dismissed.has(e.track_id),
    );
  }, [overlay, bundle.tracks, decisions]);

  // Skeletons are off by default because a hall of 236 people becomes a
  // wireframe mesh. That reasoning does not apply to a recording with nothing
  // shortlisted: there is no clutter to protect against, and the pose is the
  // only thing the run has to show. Runs once, and only upward, so a reviewer
  // who switches it off keeps it off.
  const poseDefaulted = useRef(false);
  useEffect(() => {
    if (poseDefaulted.current || !overlay) return;
    if (bundle.tracks.length === 0) {
      poseDefaulted.current = true;
      setLayers((l) => ({ ...l, skeletons: true, wrists: true }));
    }
  }, [overlay, bundle.tracks.length]);

  const [intervals, setIntervals] = useState<ReviewInterval[]>([]);
  useEffect(() => {
    let live = true;
    api
      .intervals(videoId)
      .then((r) => live && setIntervals(r.rows))
      // A failure here must not blank the timeline: the machine marks are
      // still valid without the human ones.
      .catch(() => live && setIntervals([]));
    return () => {
      live = false;
    };
  }, [videoId]);

  const [heatSource, setHeatSource] = useState<HeatSource | null>(null);
  const heat = useMemo(
    () => (heatSource ? buildHeat(overlay, heatSource, turns) : null),
    [overlay, heatSource, turns],
  );

  const ofInterest = useMemo(() => {
    const ids = new Set<number>();
    for (const t of live) {
      const crop = crops[String(t.track_id)];
      if (
        crop?.key_supported ||
        t.state === "review_candidate" ||
        decisions[t.track_id] === "human_confirmed" ||
        t.flags.some((f) => f.severity === "high")
      ) {
        ids.add(t.track_id);
      }
    }
    return ids;
  }, [live, crops, decisions]);

  const focused = focus === null ? null : people.find((t) => t.track_id === focus);
  const dismissedCount = Object.values(decisions).filter(
    (d) => d === "human_dismissed",
  ).length;

  const pickerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!pickerOpen) return;
    const away = (e: MouseEvent) => {
      if (!pickerRef.current?.contains(e.target as Node)) setPickerOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [pickerOpen]);

  const pct = (ms: number) =>
    bundle.duration_ms > 0
      ? Math.max(0, Math.min(100, (ms / bundle.duration_ms) * 100))
      : 0;

  return (
    <div className="video">
      <div className="video__main">
        {error ? (
          <p className="video__error">
            Overlay bundle failed to load: <span className="mono">{error}</span>
          </p>
        ) : (
          <WipePlayer
            videoSrc={videoSrc}
            overlay={overlay}
            layers={layers}
            focusTrack={focus}
            onlyTracks={
              // An empty shortlist must not mean an empty frame.
              //
              // `ofInterest` is built from the *bundle* -- people fusion routed
              // somewhere -- so a recording whose later stages never ran has
              // none, the set is empty, and every person gets skipped. The
              // CAMERA3 recording drew nothing at all despite its overlay
              // carrying 807 person-samples with full keypoints: stages 14b-15
              // have not run on it, so its bundle lists zero tracks.
              //
              // The overlay geometry is a separate measurement and is valid on
              // its own. With nothing shortlisted, draw everyone -- boxes, pose
              // and head direction are what the run did produce, and an
              // unmarked frame implies the detector found nobody.
              drawAll || ofInterest.size === 0 ? null : ofInterest
            }
            turns={turns}
            heat={heat}
            onTimeUpdate={setNowMs}
            seekToMs={seek}
            inset={
              spot ? (
                <Spotlight
                  trackId={spot.track_id}
                  crop={crops[String(spot.track_id)]}
                  cropBase={cropBase}
                  label={spot.label}
                  standing={spot.standing}
                  onClose={() => setSpot(null)}
                  onSeek={setSeek}
                />
              ) : null
            }
            below={
              <div className="ribbon">
                <div
                  className="ribbon__band"
                  onClick={(e) => {
                    const r = e.currentTarget.getBoundingClientRect();
                    setSeek(
                      ((e.clientX - r.left) / r.width) * bundle.duration_ms,
                    );
                  }}
                >
                  {shown.length === 0 && (
                    <span className="ribbon__none mono">
                      nothing selected — pick findings on the right
                    </span>
                  )}
                  {shown.map((k) =>
                    k.marks
                      .filter(
                        (m) => focus === null || m.track_id === focus,
                      )
                      .map((m, i) => {
                        const left = pct(m.from_ms);
                        return (
                          <i
                            key={`${k.id}-${i}`}
                            style={{
                              left: `${left}%`,
                              width: `${Math.max(pct(m.to_ms) - left, 0.35)}%`,
                              background: k.color,
                            }}
                            title={`${k.label}\n${m.detail}`}
                          />
                        );
                      }),
                  )}
                  {/* Reviewer-marked stretches, drawn last so they sit over
                      every machine mark. A human saying "here" outranks any
                      proposal underneath it, and the band is full height so it
                      cannot be mistaken for one more lane. */}
                  {intervals.map((iv) => {
                    const left = pct(iv.from_ms);
                    return (
                      <i
                        key={`iv${iv.id}`}
                        className={`ribbon__human is-${iv.kind}`}
                        style={{
                          left: `${left}%`,
                          width: `${Math.max(pct(iv.to_ms) - left, 0.4)}%`,
                        }}
                        title={
                          `${iv.kind} marked by ${iv.reviewer}\n` +
                          `${timecode(iv.from_ms)} – ${timecode(iv.to_ms)}` +
                          (iv.note ? `\n${iv.note}` : "")
                        }
                      />
                    );
                  })}
                  <span
                    className="ribbon__head"
                    style={{ left: `${pct(nowMs)}%` }}
                  />
                </div>
                <div className="ribbon__scale mono">
                  <span>{timecode(0)}</span>
                  <b>{timecode(nowMs)}</b>
                  <span>{timecode(bundle.duration_ms)}</span>
                </div>
              </div>
            }
          />
        )}

        <div className="video__layers">
          <span className="video__legend mono">Draw</span>
          <button
            type="button"
            className={drawAll ? "" : "is-on"}
            aria-pressed={!drawAll}
            onClick={() => setDrawAll(false)}
            title="Only the people this run said something about"
          >
            findings only ({ofInterest.size})
          </button>
          <button
            type="button"
            className={drawAll ? "is-on" : ""}
            aria-pressed={drawAll}
            onClick={() => setDrawAll(true)}
            title="Every tracked person"
          >
            everyone ({live.length})
          </button>
        </div>

        <div className="video__layers">
          <span className="video__legend mono">Layers</span>
          {(Object.keys(LAYER_LABEL) as (keyof Layers)[]).map((key) => (
            <button
              key={key}
              type="button"
              className={layers[key] ? "is-on" : ""}
              aria-pressed={layers[key]}
              onClick={() => setLayers((l) => ({ ...l, [key]: !l[key] }))}
            >
              {LAYER_LABEL[key]}
            </button>
          ))}
        </div>

        <div className="video__layers">
          <span className="video__legend mono">Heatmap</span>
          <button
            type="button"
            className={heatSource === null ? "is-on" : ""}
            aria-pressed={heatSource === null}
            onClick={() => setHeatSource(null)}
          >
            off
          </button>
          {(Object.keys(HEAT_LABEL) as HeatSource[]).map((key) => (
            <button
              key={key}
              type="button"
              className={heatSource === key ? "is-on" : ""}
              aria-pressed={heatSource === key}
              onClick={() => setHeatSource(key)}
            >
              {HEAT_LABEL[key]}
            </button>
          ))}
        </div>

        {/* The scale, in the units it was built from. Without the peak count a
            heatmap looks identical whether it accumulated 3 detections or
            3,000, because it is always normalised to its own maximum. */}
        {heat && heatSource && (
          <p className="video__heatnote mono">
            {heat.total.toLocaleString()} {HEAT_LABEL[heatSource]} accumulated ·
            brightest cell = {heat.peak.toFixed(1)}
            {heatSource === "objects" ? " confidence-weighted proposals" : ""}
            {heatSource === "people" ? " sightings" : ""}
            {heatSource === "turns" ? " turns" : ""}.{" "}
            <b>This is a density of proposals, not of cheating.</b> Compare it
            against &ldquo;where people were&rdquo; before reading a bright
            region as anything — a busy seat is bright in both.
          </p>
        )}

        {!overlay && !error && (
          <p className="video__loading mono">Loading overlay geometry…</p>
        )}

        <FindingsTable
          bundle={bundle}
          crops={crops}
          cropBase={cropBase}
          overlay={overlay}
          decisions={decisions}
          focus={focus}
          onSeek={(ms, trackId, label, standing) => {
            setSeek(ms);
            setFocus(trackId);
            setSpot({ track_id: trackId, label, standing });
          }}
        />

        {focused && (
          <section className="video__focus">
            <header>
              <h3>
                {
                  subjectName(focused.track_id, focused.seat, focused.seat_state)
                    .name
                }
              </h3>
              <StateChip
                state={decisions[focused.track_id] ?? focused.state}
                size="sm"
              />
              <button type="button" onClick={() => setFocus(null)}>
                Show everyone
              </button>
            </header>
            <p>
              The timeline above is showing only this person&rsquo;s marks.
            </p>
          </section>
        )}
      </div>

      <aside className="video__aside">
        {/* The dropdown, and the numbers underneath it. A wall of flag names
            next to the video was unreadable; the counts are the part a
            reviewer scans, so they get the space and the names get folded. */}
        <div className="picker" ref={pickerRef}>
          <button
            type="button"
            className="picker__button"
            aria-expanded={pickerOpen}
            onClick={() => setPickerOpen((o) => !o)}
          >
            <span>
              {shown.length} of {kinds.length} findings shown
            </span>
            <i aria-hidden="true">{pickerOpen ? "▲" : "▼"}</i>
          </button>

          {pickerOpen && (
            <div className="picker__menu">
              {kinds.length === 0 && (
                <p className="picker__none">
                  This run produced no flags to show.
                </p>
              )}
              {kinds.map((k) => (
                <label key={k.id}>
                  <input
                    type="checkbox"
                    checked={on.has(k.id)}
                    onChange={() => toggle(k.id)}
                  />
                  <i style={{ background: k.color }} aria-hidden="true" />
                  <span>{k.label}</span>
                  <b className="mono">{k.people}</b>
                </label>
              ))}
              <div className="picker__all">
                <button
                  type="button"
                  onClick={() => setChosen(new Set(kinds.map((k) => k.id)))}
                >
                  All
                </button>
                <button type="button" onClick={() => setChosen(new Set())}>
                  None
                </button>
                <button type="button" onClick={() => setChosen(null)}>
                  Default
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="ftiles">
          {shown.map((k) => (
            <button
              key={k.id}
              type="button"
              className="ftile"
              style={{ ["--ftile" as string]: k.color }}
              title={k.note}
              onClick={() => {
                const first = k.marks
                  .slice()
                  .sort((a, b) => a.from_ms - b.from_ms)[0];
                if (first) setSeek(first.from_ms);
              }}
            >
              <b className="mono">{k.people}</b>
              <span>{k.label}</span>
              <em className="mono">
                {k.marks.length} mark{k.marks.length === 1 ? "" : "s"}
              </em>
            </button>
          ))}
          {shown.length === 0 && (
            <p className="ftiles__none">
              Nothing selected. Open the dropdown above to choose what the
              timeline shows.
            </p>
          )}
        </div>

        {dismissedCount > 0 && (
          <p className="video__hidden mono">
            {dismissedCount} hidden — marked not a violation
          </p>
        )}

        <p className="video__legend mono">Isolate a person</p>
        <ul className="video__people">
          <li>
            <button
              type="button"
              className={focus === null ? "is-on" : ""}
              onClick={() => setFocus(null)}
            >
              <span>Everyone</span>
            </button>
          </li>
          {people.map((t) => (
            <li key={t.track_id}>
              <button
                type="button"
                className={focus === t.track_id ? "is-on" : ""}
                onClick={() => {
                  setFocus(t.track_id);
                  if (t.first_seen_ms !== null) setSeek(t.first_seen_ms);
                }}
              >
                <span>#{t.track_id}</span>
                <StateChip
                  state={decisions[t.track_id] ?? t.state}
                  size="sm"
                />
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
