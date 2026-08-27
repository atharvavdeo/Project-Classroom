import { useState } from "react";
import type { ProjectRef, VideoRef } from "../review";
import { api, type Assets } from "../api";
import "./Library.css";

/**
 * The project rail.
 *
 * Everything in it was typed by someone. A console with no projects shows a
 * sentence saying so and one button, rather than an example hierarchy that a
 * reviewer could mistake for real centres.
 *
 * Attaching a recording picks from what is actually on disk. A free-text path
 * field is a field that can be wrong, and a wrong media path fails silently as
 * a video element that never loads.
 */
export function Library({
  projects,
  assets,
  videoId,
  open: navOpen = true,
  onPick,
  onChanged,
  onOpenRL,
  rlOpen = false,
}: {
  projects: ProjectRef[];
  assets: Assets | null;
  videoId: string;
  /** Drawer state on a phone. Above the breakpoint the rail is permanent and
   *  this is ignored. */
  open?: boolean;
  onPick: (id: string) => void;
  onChanged: (projects: ProjectRef[]) => void;
  /** Opens the cross-recording view of everything reviewers overturned. */
  onOpenRL: () => void;
  rlOpen?: boolean;
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [adding, setAdding] = useState<null | { kind: "project" }
    | { kind: "centre"; projectId: string }
    | { kind: "video"; centreId: string }>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (fn: () => Promise<{ projects: ProjectRef[] }>) => {
    setBusy(true);
    setError(null);
    try {
      onChanged((await fn()).projects);
      setAdding(null);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <nav
      className={`lib ${navOpen ? "is-open" : ""}`}
      style={
        typeof window !== "undefined" && window.matchMedia("(max-width: 54rem)").matches
          ? {
            transform: navOpen ? "translateX(0)" : "translateX(-101%)",
            visibility: navOpen ? "visible" : "hidden",
          }
          : undefined
      }
      aria-hidden={undefined}
    >
      <div className="lib__brand">
        <span className="lib__mark" aria-hidden="true" />
        <b>Drishti</b>
      </div>

      {projects.length === 0 && (
        <p className="lib__empty">
          No projects yet. A project is an exam; inside it you add centres, and
          inside those the recordings from each hall.
        </p>
      )}

      {projects.map((project) => (
        <section key={project.id} className="lib__project">
          <header className="lib__ptitle">
            <span>{project.name}</span>
            <button
              type="button"
              title="Remove this project"
              onClick={() => void run(() => api.remove("projects", project.id))}
            >
              ×
            </button>
          </header>

          {project.centres.length === 0 && (
            <p className="lib__hint">No centres in this project yet.</p>
          )}

          {project.centres.map((centre) => {
            const isOpen = open[centre.id] ?? true;
            return (
              <div key={centre.id}>
                <button
                  className="lib__centre"
                  aria-expanded={isOpen}
                  onClick={() =>
                    setOpen((o) => ({ ...o, [centre.id]: !isOpen }))
                  }
                >
                  <span>{centre.name}</span>
                  <b className="mono">{centre.videos.length}</b>
                </button>

                {isOpen && (
                  <ul className="lib__videos">
                    {centre.videos.map((v) => (
                      <li key={v.id}>
                        <button
                          className={v.id === videoId ? "is-on" : ""}
                          title={
                            v.processed ? v.name : `${v.name} — no run yet`
                          }
                          onClick={() => onPick(v.id)}
                        >
                          <span>{v.name}</span>
                          {v.processed ? (
                            (v.decided ?? 0) > 0 && (
                              <em className="is-count">{v.decided}</em>
                            )
                          ) : (
                            <em>not run</em>
                          )}
                        </button>
                      </li>
                    ))}
                    <li>
                      <button
                        className="lib__add"
                        onClick={() =>
                          setAdding({ kind: "video", centreId: centre.id })
                        }
                      >
                        + Recording
                      </button>
                    </li>
                  </ul>
                )}

                {adding?.kind === "video" && adding.centreId === centre.id && (
                  <VideoForm
                    assets={assets}
                    busy={busy}
                    onCancel={() => setAdding(null)}
                    onSubmit={(body) =>
                      void run(() => api.createVideo(centre.id, body))
                    }
                  />
                )}
              </div>
            );
          })}

          {adding?.kind === "centre" && adding.projectId === project.id ? (
            <NameForm
              label="Centre name"
              placeholder="e.g. the centre this camera is in"
              busy={busy}
              onCancel={() => setAdding(null)}
              onSubmit={(name) =>
                void run(() => api.createCentre(project.id, name))
              }
            />
          ) : (
            <button
              className="lib__add"
              onClick={() => setAdding({ kind: "centre", projectId: project.id })}
            >
              + Centre
            </button>
          )}
        </section>
      ))}

      {error && <p className="lib__err mono">{error}</p>}

      {adding?.kind === "project" ? (
        <NameForm
          label="Project name"
          placeholder="e.g. the exam these recordings belong to"
          busy={busy}
          onCancel={() => setAdding(null)}
          onSubmit={(name) => void run(() => api.createProject(name))}
        />
      ) : (
        <button
          className="lib__new"
          type="button"
          onClick={() => setAdding({ kind: "project" })}
        >
          + New project
        </button>
      )}

      {/* Pinned to the bottom, below the tree, because it is not part of the
          hierarchy: it spans every project and every centre. The label is the
          user's word for it; the screen itself is honest about the fact that
          nothing here is reinforcement learning. */}
      <button
        className={`lib__rl ${rlOpen ? "is-on" : ""}`}
        type="button"
        onClick={onOpenRL}
        title="Everything reviewers overturned, across every recording"
      >
        <span className="lib__rlTag mono">RL</span>
        <span className="lib__rlText">Rejected proposals</span>
      </button>
    </nav>
  );
}

function NameForm({
  label,
  placeholder,
  busy,
  onSubmit,
  onCancel,
}: {
  label: string;
  placeholder: string;
  busy: boolean;
  onSubmit: (name: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  return (
    <form
      className="lib__form"
      onSubmit={(e) => {
        e.preventDefault();
        if (name.trim()) onSubmit(name.trim());
      }}
    >
      <label>
        {label}
        <input
          autoFocus
          value={name}
          placeholder={placeholder}
          onChange={(e) => setName(e.target.value)}
        />
      </label>
      <div className="lib__formrow">
        <button type="submit" disabled={busy || !name.trim()}>
          Create
        </button>
        <button type="button" className="is-plain" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

/**
 * Attach a recording.
 *
 * Media is required; a run bundle is not. A recording with no bundle is a
 * recording the pipeline has not seen, and saying "not run" is the honest
 * answer — an empty dashboard would read as "nothing was found".
 */
function VideoForm({
  assets,
  busy,
  onSubmit,
  onCancel,
}: {
  assets: Assets | null;
  busy: boolean;
  onSubmit: (body: Partial<VideoRef> & { name: string }) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [video, setVideo] = useState("");
  const [bundle, setBundle] = useState("");
  const [overlay, setOverlay] = useState("");
  const [crops, setCrops] = useState("");

  const options = (list: { url: string; name: string }[] | undefined) =>
    (list ?? []).map((a) => (
      <option key={a.url} value={a.url}>
        {a.name}
      </option>
    ));

  return (
    <form
      className="lib__form"
      onSubmit={(e) => {
        e.preventDefault();
        if (name.trim()) onSubmit({ name: name.trim(), video, bundle, overlay, crops });
      }}
    >
      <label>
        What this camera shows
        <input
          autoFocus
          value={name}
          placeholder="e.g. Hall 3 · Camera 2"
          onChange={(e) => setName(e.target.value)}
        />
      </label>
      <label>
        Recording
        <select value={video} onChange={(e) => setVideo(e.target.value)}>
          <option value="">— none —</option>
          {options(assets?.media)}
        </select>
      </label>
      <label>
        Run results
        <select
          value={bundle}
          onChange={(e) => {
            setBundle(e.target.value);
            // The three run artifacts are named off the same run key, so
            // picking the bundle is enough to guess the other two. They stay
            // editable, because a guess that cannot be corrected is a trap.
            const key = e.target.value.split("/").pop()?.replace(/\.json$/, "");
            if (!key) return;
            const ov = assets?.overlays.find((o) => o.name.startsWith(key));
            const cr = assets?.crops.find((c) => c.name === key);
            if (ov) setOverlay(ov.url);
            if (cr) setCrops(cr.url);
          }}
        >
          <option value="">— not run yet —</option>
          {options(assets?.bundles)}
        </select>
      </label>
      <label>
        Frame geometry
        <select value={overlay} onChange={(e) => setOverlay(e.target.value)}>
          <option value="">— none —</option>
          {options(assets?.overlays)}
        </select>
      </label>
      <label>
        Review frames
        <select value={crops} onChange={(e) => setCrops(e.target.value)}>
          <option value="">— none —</option>
          {options(assets?.crops)}
        </select>
      </label>
      <div className="lib__formrow">
        <button type="submit" disabled={busy || !name.trim()}>
          Add
        </button>
        <button type="button" className="is-plain" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}
