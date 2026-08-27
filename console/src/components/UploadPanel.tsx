import { useEffect, useMemo, useRef, useState } from "react";
import type { ProjectRef } from "../review";
import { api } from "../api";
import "./UploadPanel.css";

type Phase =
  | { at: "idle" }
  | { at: "uploading"; fraction: number }
  | { at: "queued"; videoId: string; jobId: number | null; state: string; stage: string | null }
  | { at: "error"; message: string };

/**
 * Add a recording, and decide where it belongs, in one panel.
 *
 * The destination is a required choice rather than a default, because filing a
 * recording under the wrong centre is not visibly wrong later -- it just
 * quietly attributes a hall's findings to the wrong venue. A new project or
 * centre can be created here without leaving the panel, since needing to go
 * elsewhere first is the reason people file things in the nearest existing
 * folder instead of the right one.
 *
 * Uploading and processing are separate steps on purpose. A recording can be
 * attached now and run when the GPU is free, and the run is queued rather than
 * awaited -- the pipeline is minutes of work and a request held open for it is
 * a request that times out with nobody sure whether the work is still going.
 */
export function UploadPanel({
  projects,
  onChanged,
  onOpen,
}: {
  projects: ProjectRef[];
  onChanged: (projects: ProjectRef[]) => void;
  onOpen?: (videoId: string) => void;
}) {
  const [open, setOpen] = useState(projects.length === 0);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [projectId, setProjectId] = useState("");
  const [centreId, setCentreId] = useState("");
  const [newProject, setNewProject] = useState("");
  const [newCentre, setNewCentre] = useState("");
  const [phase, setPhase] = useState<Phase>({ at: "idle" });
  const inputRef = useRef<HTMLInputElement>(null);

  const project = projects.find((p) => p.id === projectId);
  const centres = project?.centres ?? [];

  // Keep the two selects consistent: a centre from a project you have since
  // switched away from would upload into the wrong place.
  useEffect(() => {
    if (projectId && !projects.some((p) => p.id === projectId)) setProjectId("");
    if (centreId && !centres.some((c) => c.id === centreId)) setCentreId("");
  }, [projects, projectId, centreId, centres]);

  const makingProject = projectId === "__new";
  const makingCentre = centreId === "__new" || makingProject;

  const ready = useMemo(() => {
    if (!file || !name.trim()) return false;
    if (makingProject) return newProject.trim() && newCentre.trim();
    if (!projectId) return false;
    if (makingCentre) return Boolean(newCentre.trim());
    return Boolean(centreId);
  }, [file, name, projectId, centreId, newProject, newCentre, makingProject, makingCentre]);

  async function submit(startRun: boolean) {
    if (!file || !ready) return;
    setPhase({ at: "uploading", fraction: 0 });
    try {
      // Resolve the destination first. If this fails nothing has been
      // uploaded yet, which is the cheaper failure by a wide margin.
      let targetCentre = centreId;
      let tree = projects;

      if (makingProject) {
        const created = await api.createProject(newProject.trim());
        tree = created.projects;
        const p = tree.find((x) => x.name === newProject.trim());
        if (!p) throw new Error("Project was created but could not be found.");
        const withCentre = await api.createCentre(p.id, newCentre.trim());
        tree = withCentre.projects;
        targetCentre =
          tree
            .find((x) => x.id === p.id)!
            .centres.find((c) => c.name === newCentre.trim())?.id ?? "";
      } else if (makingCentre) {
        const withCentre = await api.createCentre(projectId, newCentre.trim());
        tree = withCentre.projects;
        targetCentre =
          tree
            .find((x) => x.id === projectId)!
            .centres.find((c) => c.name === newCentre.trim())?.id ?? "";
      }

      if (!targetCentre) throw new Error("Could not resolve a centre to file this under.");

      const up = await api.upload(targetCentre, file, name.trim(), (f) =>
        setPhase({ at: "uploading", fraction: f }),
      );

      const after = await api.tree();
      onChanged(after.projects);

      let jobId: number | null = null;
      let state = "not queued";
      if (startRun) {
        const queued = await api.process(up.video_id);
        jobId = queued.job.id;
        state = queued.job.state;
      }
      setPhase({ at: "queued", videoId: up.video_id, jobId, state, stage: null });
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (e) {
      setPhase({ at: "error", message: String(e instanceof Error ? e.message : e) });
    }
  }

  // Follow the job while the panel is open. Polling, not a socket: the states
  // change every few seconds at most and a dropped socket would need the same
  // recovery anyway.
  const jobId = phase.at === "queued" ? phase.jobId : null;
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    const tick = async () => {
      try {
        const { job } = await api.job(jobId);
        if (!live) return;
        setPhase((p) =>
          p.at === "queued"
            ? { ...p, state: job.state, stage: job.stage }
            : p,
        );
        if (job.state === "complete") {
          const after = await api.tree();
          if (live) onChanged(after.projects);
        }
      } catch {
        /* the console being briefly unreachable is not a job failure */
      }
    };
    const timer = window.setInterval(tick, 4000);
    void tick();
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [jobId, onChanged]);

  if (!open) {
    return (
      <button className="up__open" type="button" onClick={() => setOpen(true)}>
        + Add a recording
      </button>
    );
  }

  return (
    <section className="up">
      <header className="up__top">
        <h3>Add a recording</h3>
        <button type="button" onClick={() => setOpen(false)} aria-label="Close">
          ×
        </button>
      </header>

      <div className="up__grid">
        <label className="up__file">
          <span>Recording</span>
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,video/webm,video/x-matroska,.mkv"
            onChange={(e) => {
              const f = e.target.files?.[0] ?? null;
              setFile(f);
              setPhase({ at: "idle" });
              // Seed the label from the filename, minus the extension. It is a
              // starting point, not a decision -- the field stays editable.
              if (f && !name.trim()) setName(f.name.replace(/\.[^.]+$/, ""));
            }}
          />
          {file && (
            <em className="mono">
              {(file.size / 1048576).toFixed(1)} MB · {file.type || "video"}
            </em>
          )}
        </label>

        <label>
          <span>What this camera shows</span>
          <input
            value={name}
            placeholder="e.g. Hall 3 · Camera 2"
            onChange={(e) => setName(e.target.value)}
          />
        </label>

        <label>
          <span>Project</span>
          <select
            value={projectId}
            onChange={(e) => {
              setProjectId(e.target.value);
              setCentreId("");
            }}
          >
            <option value="">— choose —</option>
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
            <option value="__new">+ New project…</option>
          </select>
        </label>

        {makingProject && (
          <label>
            <span>New project name</span>
            <input
              autoFocus
              value={newProject}
              placeholder="the exam these recordings belong to"
              onChange={(e) => setNewProject(e.target.value)}
            />
          </label>
        )}

        {!makingProject && (
          <label>
            <span>Centre</span>
            <select
              value={centreId}
              disabled={!projectId}
              onChange={(e) => setCentreId(e.target.value)}
            >
              <option value="">— choose —</option>
              {centres.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
              <option value="__new">+ New centre…</option>
            </select>
          </label>
        )}

        {makingCentre && (
          <label>
            <span>New centre name</span>
            <input
              value={newCentre}
              placeholder="the venue this camera is in"
              onChange={(e) => setNewCentre(e.target.value)}
            />
          </label>
        )}
      </div>

      {phase.at === "uploading" && (
        <div className="up__bar" role="status">
          <i style={{ width: `${Math.round(phase.fraction * 100)}%` }} />
          <span className="mono">
            Uploading {Math.round(phase.fraction * 100)}%
          </span>
        </div>
      )}

      {phase.at === "queued" && (
        <div className="up__done" role="status">
          <b>Uploaded.</b>{" "}
          {phase.jobId === null ? (
            <span>
              Filed as not run. Nothing has been processed, so it will say
              &ldquo;not run&rdquo; until you queue it.
            </span>
          ) : (
            <span>
              Job #{phase.jobId} is <b>{phase.state}</b>
              {phase.stage ? ` — ${phase.stage}` : ""}. The pipeline runs on the
              GPU; this page does not wait for it.
            </span>
          )}
          {onOpen && (
            <button type="button" onClick={() => onOpen(phase.videoId)}>
              Open it
            </button>
          )}
        </div>
      )}

      {phase.at === "error" && (
        <p className="up__err mono">{phase.message}</p>
      )}

      <div className="up__actions">
        <button
          type="button"
          className="is-primary"
          disabled={!ready || phase.at === "uploading"}
          onClick={() => void submit(true)}
        >
          Upload and process
        </button>
        <button
          type="button"
          disabled={!ready || phase.at === "uploading"}
          onClick={() => void submit(false)}
        >
          Upload only
        </button>
      </div>
      <p className="up__note">
        Processing is queued, not immediate: a run is minutes of GPU work and
        happens on the machine holding the models. The recording reads
        &ldquo;not run&rdquo; until a job finishes — which is the honest state,
        since an empty dashboard would read as &ldquo;nothing was found&rdquo;.
      </p>
    </section>
  );
}
