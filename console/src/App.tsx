import { useCallback, useEffect, useMemo, useState } from "react";
import type { Bundle } from "./types";
import type { CropManifest, Decision, ProjectRef } from "./review";
import { api, type Assets, type DecisionRow } from "./api";
import { findVideo, firstProcessed, firstVideo } from "./projects";
import { Library } from "./components/Library";
import { UploadPanel } from "./components/UploadPanel";
import { Dashboard } from "./screens/Dashboard";
import { ReviewScreen } from "./screens/ReviewScreen";
import { RLScreen } from "./screens/RLScreen";
import { DismissedScreen } from "./screens/DismissedScreen";
import { VideoScreen } from "./screens/VideoScreen";
import "./App.css";

type Screen = "dashboard" | "review" | "video" | "dismissed" | "rl";

const SCREENS: { id: Screen; label: string }[] = [
  { id: "dashboard", label: "Overview" },
  { id: "review", label: "Review" },
  { id: "video", label: "Video" },
  { id: "dismissed", label: "Not violations" },
];

export default function App() {
  const [showReadOnlyNotice, setShowReadOnlyNotice] = useState(true);
  const [projects, setProjects] = useState<ProjectRef[]>([]);
  const [assets, setAssets] = useState<Assets | null>(null);
  const [videoId, setVideoId] = useState("");
  const [screen, setScreen] = useState<Screen>("dashboard");
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [crops, setCrops] = useState<CropManifest>({});
  const [error, setError] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<Record<number, DecisionRow>>({});
  const [booted, setBooted] = useState(false);
  // The rail is a drawer on a phone. Closed by default there, and closed
  // again after any navigation -- a menu that stays open over the thing you
  // just chose is a menu you have to dismiss twice.
  const [navOpen, setNavOpen] = useState(false);

  // Load the tree once. Nothing is assumed about its contents -- an empty
  // console is a real state, not an error.
  useEffect(() => {
    Promise.all([api.tree(), api.assets()])
      .then(([t, a]) => {
        setProjects(t.projects);
        setAssets(a);
        const start = firstProcessed(t.projects) ?? firstVideo(t.projects);
        if (start) setVideoId(start.video.id);
      })
      .catch((e) => setError(String(e instanceof Error ? e.message : e)))
      .finally(() => setBooted(true));
  }, []);

  const found = useMemo(
    () => findVideo(projects, videoId),
    [projects, videoId],
  );
  const video = found?.video;

  // Run artifacts and the decisions made against them load together, because
  // a queue rendered before its decisions arrive would briefly show people the
  // reviewer has already answered for.
  useEffect(() => {
    if (!video?.bundle) {
      setBundle(null);
      setCrops({});
      setDecisions({});
      return;
    }
    let live = true;
    setError(null);
    Promise.all([
      fetch(video.bundle).then((r) => {
        if (!r.ok) throw new Error(`bundle ${r.status}`);
        return r.json();
      }),
      video.crops
        ? fetch(`${video.crops}/manifest.json`)
          .then((r) => (r.ok ? r.json() : {}))
          .catch(() => ({}))
        : Promise.resolve({}),
      api.decisions(video.id).catch(() => ({ current: {}, revisions: 0 })),
    ])
      .then(([b, c, d]) => {
        if (!live) return;
        setBundle(b);
        setCrops(c);
        setDecisions(
          Object.fromEntries(
            Object.values(d.current).map((r) => [r.track_id, r]),
          ),
        );
      })
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [video]);

  /** Write the answer, then reflect it. The row the server hands back is the
   *  one stored, so the UI shows what was persisted rather than what was
   *  hoped -- a decision that silently failed to save is worse than one that
   *  visibly did not. */
  const decide = useCallback(
    async (trackId: number, decision: Decision, extra?: Partial<DecisionRow>) => {
      if (!video) return;
      const row = await api.decide(video.id, {
        track_id: trackId,
        decision,
        machine_state: extra?.machine_state ?? null,
        key_class: extra?.key_class ?? null,
        key_verdict: extra?.key_verdict ?? null,
        key_pts_ms: extra?.key_pts_ms ?? null,
      });
      setDecisions((d) => ({ ...d, [trackId]: row }));
      setProjects((ps) =>
        ps.map((p) => ({
          ...p,
          centres: p.centres.map((c) => ({
            ...c,
            videos: c.videos.map((v) =>
              v.id === video.id
                ? { ...v, decided: (v.decided ?? 0) + (decisions[trackId] ? 0 : 1) }
                : v,
            ),
          })),
        })),
      );
    },
    [video, decisions],
  );

  const plain: Record<number, Decision> = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(decisions).map(([k, v]) => [Number(k), v.decision]),
      ),
    [decisions],
  );

  const uploadPanel = (
    <UploadPanel
      projects={projects}
      onChanged={setProjects}
      onOpen={(id) => {
        setVideoId(id);
        setScreen("dashboard");
      }}
    />
  );

  return (
    <div className={`app ${navOpen ? "is-nav" : ""}`}>
      <button
        type="button"
        className="app__scrim"
        aria-label="Close navigation"
        tabIndex={navOpen ? 0 : -1}
        onClick={() => setNavOpen(false)}
      />
      <Library
        projects={projects}
        assets={assets}
        videoId={videoId}
        open={navOpen}
        onPick={(id) => {
          setVideoId(id);
          setScreen("dashboard");
          setNavOpen(false);
        }}
        rlOpen={screen === "rl"}
        onOpenRL={() => {
          setScreen("rl");
          setNavOpen(false);
        }}
        onChanged={(next) => {
          setProjects(next);
          if (!findVideo(next, videoId)) {
            setVideoId(
              (firstProcessed(next) ?? firstVideo(next))?.video.id ?? "",
            );
          }
        }}
      />

      <div className="shell">
        <header className="top">
          <button
            type="button"
            className="top__menu"
            aria-expanded={navOpen}
            aria-label="Recordings"
            onClick={() => setNavOpen((o) => !o)}
          >
            <svg viewBox="0 0 16 14" aria-hidden="true">
              <rect y="1" width="16" height="2" rx="1" />
              <rect y="6" width="16" height="2" rx="1" />
              <rect y="11" width="16" height="2" rx="1" />
            </svg>
          </button>
          <div className="top__where">
            <span className="mono">
              {found ? `${found.project.name} / ${found.centre.name}` : " "}
            </span>
            <h1>{video?.name ?? "No recording selected"}</h1>
          </div>
          <nav className="top__tabs">
            {SCREENS.map((s) => (
              <button
                key={s.id}
                className={screen === s.id ? "is-on" : ""}
                onClick={() => setScreen(s.id)}
                disabled={!bundle}
              >
                {s.label}
              </button>
            ))}
          </nav>
        </header>

        <main className="main">
          {error && <p className="state state--err">{error}</p>}

          {booted && projects.length === 0 && !error && (
            <div className="state state--empty">
              <p>
                Nothing here yet. Add a recording below — you can create the
                project and centre it belongs to at the same time.
              </p>
              {uploadPanel}
            </div>
          )}

          {/* A console with a hierarchy but no run to show still needs the
              way in. Without this the panel would only appear once something
              had already been processed. */}
          {booted && projects.length > 0 && !bundle && !error && (
            <div className="state state--empty">{uploadPanel}</div>
          )}

          {video && !video.processed && (
            <p className="state">
              This recording has not been processed yet. Run the pipeline on it
              to see findings — an empty dashboard would read like
              &ldquo;nothing was found&rdquo;, which is not the same thing.
            </p>
          )}

          {video?.processed && !bundle && !error && (
            <p className="state mono">Loading run…</p>
          )}

          {bundle && screen === "dashboard" && (
            <Dashboard
              bundle={bundle}
              crops={crops}
              video={video!}
              decisions={plain}
              onReview={() => setScreen("review")}
              upload={uploadPanel}
            />
          )}

          {bundle && screen === "review" && (
            <ReviewScreen
              bundle={bundle}
              crops={crops}
              cropBase={video!.crops ?? ""}
              decisions={decisions}
              onDecide={decide}
            />
          )}

          {bundle && screen === "video" && (
            <VideoScreen
              bundle={bundle}
              videoId={video!.id}
              crops={crops}
              cropBase={video!.crops ?? ""}
              videoSrc={video!.video ?? ""}
              overlaySrc={video!.overlay ?? ""}
              decisions={plain}
            />
          )}

          {screen === "rl" && <RLScreen />}

          {bundle && screen === "dismissed" && (
            <DismissedScreen
              bundle={bundle}
              crops={crops}
              cropBase={video!.crops ?? ""}
              decisions={decisions}
              onRestore={(trackId, extra) =>
                void decide(trackId, "needs_better_view", extra)
              }
            />
          )}
        </main>
      </div>
      {showReadOnlyNotice && (
        <div className="read-only-notice" role="presentation">
          <section
            className="read-only-notice__card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="read-only-title"
          >
            <div className="read-only-notice__mark" aria-hidden="true">
              <span />
              <span />
              <i />
            </div>
            <p className="mono read-only-notice__eyebrow">Public review console</p>
            <h2 id="read-only-title">Read-only public view</h2>
            <p>
              Existing evidence and the interface remain available to explore.
              Uploads, review decisions, job processing, and AI requests are
              disabled on this public deployment.
            </p>
            <button type="button" onClick={() => setShowReadOnlyNotice(false)}>
              Continue to the console
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
