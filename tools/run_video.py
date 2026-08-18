"""End-to-end run over one video, with overlays, into a self-contained folder.

    python tools/run_video.py "C:/DrishtiAI/03.CCTV Mobile Usage.mkv" --name mobile03
    python tools/run_video.py VIDEO --skip verify        # everything but Gemma
    python tools/run_video.py VIDEO --residency combined # both CV models at once

Model residency
---------------
By default exactly one model is loaded at a time. The detector runs over every
event and is released; then the pose model runs over every event and is
released; then, and only then, `llama-server` is started for the verifier and
shut down when it finishes. `--residency combined` keeps the two CV models
co-resident for one shared decode pass, which is faster and is what the 32 GB
target can afford -- it is offered, not assumed.

Sequential residency means the detector pass and the pose pass each write their
own evidence stream independently, which is why `evidence.persist` clears one
stream at a time. Before that it cleared both, and the pose pass silently erased
the detections the detector pass had just written.

Nothing here blocks on anything it also holds. The verifier is an out-of-process
HTTP server: it is started with a bounded health poll, every request carries a
timeout, and it is terminated in a `finally` so a failed verification cannot
leave a GPU-resident process behind.

Output layout
-------------
    runs/<name>/
      run.json                 manifest: inputs, config, stage timings, counts
      log.txt                  the console transcript
      calibration.json         the seat geometry this run used
      motion/windows.csv       every window, every seat, every zone
      motion/timeline.jpg      per-seat deviation strip with event bands
      motion/heatmap.jpg       seat activity heatmap over the recording
      events/events.csv        the ranked event table
      events/<id>/frames/      sampled stills, marked
      events/<id>/sheet.jpg    the verifier contact sheet
      events/<id>/clip.mp4     the evidence clip
      overlay/marked.mp4       the recording with seats, zones and events drawn
      verify/<id>.json         verifier output, written last
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import (  # noqa: E402
    calibration as cal_mod, clips, db, detect, evidence, ingest, motion,
    pipeline, pose, scoring, track, verify,
)
from classroom.calibration import Calibration, MaskRegion, Seat  # noqa: E402
from classroom.config import Config  # noqa: E402

# Seat geometry per camera, shared with tools/run_corpus.py. Traced from frames
# of each view; the burned-in date/time sits top-left in every corpus file.
OVERLAY_720 = (8, 30, 520, 92)
OVERLAY_480 = (6, 20, 300, 60)

LAYOUTS = {
    "Seat No. 12": {"seats": {"L-front": (30, 300, 330, 620),
                              "C-front": (600, 230, 790, 560),
                              "R-cluster": (800, 60, 1010, 420)},
                    "overlay": OVERLAY_720},
    # Camera 12. The R-desk row is backed by an interior serving window into an
    # adjacent room, and people move through it constantly. Without the mask the
    # empty desk row reports activity driven entirely by a different room --
    # observed directly in the first run of this file, where R-desk went ACTIVE
    # at 174.8 s with nobody seated in it. This is the second measured instance
    # of the same class of error; the first was a person detected inside a glass
    # partition reflection (PRD 8.2.1).
    "03.CCTV": {"seats": {"S-61": (60, 280, 380, 700),
                          "R-desk": (700, 120, 1180, 480)},
                "overlay": OVERLAY_720,
                "environment": [(655, 0, 965, 232)]},
    "01.Candidate": {"seats": {"B-18": (60, 380, 380, 700),
                               "B-20": (380, 300, 640, 660),
                               "B-mid": (700, 300, 1080, 700)},
                     "overlay": OVERLAY_720},
    "02.Candidate": {"seats": {"B-18": (60, 380, 380, 700),
                               "B-20": (380, 300, 640, 660),
                               "B-mid": (700, 300, 1080, 700)},
                     "overlay": OVERLAY_720},
    "05.Crowd": {"seats": {"Left": (20, 180, 420, 700),
                           "Centre": (460, 160, 840, 700),
                           "Right": (880, 140, 1260, 700)},
                 "overlay": OVERLAY_720},
    "04.CCTV": {"seats": {"Whole": (10, 60, 630, 470)}, "overlay": OVERLAY_480},
}

SEAT_COLOURS = [(255, 210, 60), (80, 220, 120), (240, 150, 40),
                (200, 60, 220), (60, 200, 240)]
ACTIVE_COLOUR = (60, 60, 255)


class Tee:
    """Console output duplicated into the run folder."""

    def __init__(self, path: Path):
        self.stream = path.open("w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.stream.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def rect(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def layout_for(name: str):
    for key, layout in LAYOUTS.items():
        if name.startswith(key):
            return layout
    return None


def build_calibration(video_name: str, width: int, height: int) -> Calibration:
    layout = layout_for(video_name)
    if layout is None:
        raise SystemExit(
            f"no seat layout for {video_name!r}. Add one to LAYOUTS, or author "
            f"it with tools/calibrate.py and load it with `classroom.cli "
            f"calib-import`."
        )
    seats = []
    for label, box in layout["seats"].items():
        x1, y1, x2, y2 = box
        mid = y1 + (y2 - y1) * 0.55
        seats.append(Seat(
            label=label, polygon=rect(box),
            zones={
                # The screen band is excluded from scoring entirely: a live
                # monitor is an aperiodic motion source that has nothing to do
                # with the candidate (PRD 4.2).
                "screen": rect((x1, y1, x2, y1 + (y2 - y1) * 0.30)),
                "desk": rect((x1, mid, x2, y2)),
                "torso": rect((x1, y1 + (y2 - y1) * 0.30, x2, mid)),
            },
            desk_line_y=mid,
        ))
    masks = [MaskRegion(kind="overlay", polygon=rect(layout["overlay"]),
                        note="burned-in date/time, changes every second")]
    for box in layout.get("environment", []):
        masks.append(MaskRegion(
            kind="environment", polygon=rect(box),
            note="motion from outside this room"))
    for box in layout.get("reflection", []):
        masks.append(MaskRegion(kind="reflection", polygon=rect(box),
                                note="glass or mirror"))
    cal = Calibration(camera_id=0, version=1, frame_w=width, frame_h=height,
                      seats=seats, masks=masks)
    cal.validate()
    return cal


# ---------------------------------------------------------------- overlays --

def draw_calibration(frame: np.ndarray, cal: Calibration,
                     active: set[str] | None = None) -> np.ndarray:
    """Draw seat footprints, zones and masks onto a frame."""
    view = frame.copy()
    overlay = view.copy()
    for index, seat in enumerate(cal.seats):
        colour = SEAT_COLOURS[index % len(SEAT_COLOURS)]
        hot = active is not None and seat.label in active
        pts = np.array(seat.polygon, dtype=np.int32)
        cv2.polylines(view, [pts], True, ACTIVE_COLOUR if hot else colour,
                      3 if hot else 1, cv2.LINE_AA)
        for kind, polygon in seat.zones.items():
            zpts = np.array(polygon, dtype=np.int32)
            if kind == "screen":
                # Shaded rather than outlined, because "this region is not
                # scored" is the thing a reviewer needs to see at a glance.
                cv2.fillPoly(overlay, [zpts], (40, 40, 40))
            else:
                cv2.polylines(view, [zpts], True, colour, 1, cv2.LINE_AA)
        if seat.desk_line_y is not None:
            x1 = int(min(p[0] for p in seat.polygon))
            x2 = int(max(p[0] for p in seat.polygon))
            y = int(seat.desk_line_y)
            cv2.line(view, (x1, y), (x2, y), colour, 1, cv2.LINE_AA)
        label = seat.label + ("  ACTIVE" if hot else "")
        cv2.putText(view, label, (pts[:, 0].min() + 4, pts[:, 1].min() + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    ACTIVE_COLOUR if hot else colour, 2, cv2.LINE_AA)

    for mask in cal.masks:
        mpts = np.array(mask.polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [mpts], (0, 0, 120))
        cv2.putText(view, f"masked: {mask.kind}", (mpts[:, 0].min() + 4,
                    mpts[:, 1].max() - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (120, 120, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.35, view, 0.65, 0)


def render_marked_video(
    video_path: str, out_path: Path, cal: Calibration,
    events: list[dict], duration_s: float, fps: float = 8.0,
) -> Path:
    """Write the recording back out with the analysis drawn on it.

    Re-encoded at a fixed low frame rate rather than frame-for-frame. Three of
    the six corpus files are variable frame rate, so a frame-for-frame copy
    would need the original timing reconstructed to stay in sync; sampling on a
    fixed grid by timestamp sidesteps that entirely and produces a file that
    plays anywhere. The banner carries the true source timestamp, so nothing
    about the timing is lost to the reviewer.
    """
    import av

    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(0.0, duration_s, 1.0 / fps)

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    time_base = float(stream.time_base)

    writer = None
    try:
        wanted = iter(times)
        target = next(wanted, None)
        for frame in container.decode(stream):
            if target is None:
                break
            t = float(frame.pts * time_base) if frame.pts is not None else 0.0
            if t + 1e-9 < target:
                continue
            image = frame.to_ndarray(format="bgr24")
            active = {e["seat_label"] for e in events
                      if e["t_start"] <= t <= e["t_end"]}
            view = draw_calibration(image, cal, active)
            banner(view, t, duration_s, events, active)

            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (view.shape[1], view.shape[0]))
                if not writer.isOpened():
                    raise SystemExit(f"could not open {out_path} for writing")
            writer.write(view)

            while target is not None and target <= t:
                target = next(wanted, None)
    finally:
        if writer is not None:
            writer.release()
        container.close()
    return out_path


def banner(view: np.ndarray, t: float, duration_s: float,
           events: list[dict], active: set[str]) -> None:
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, h - 46), (w, h), (0, 0, 0), -1)
    text = f"t={t:7.2f}s / {duration_s:.0f}s"
    if active:
        text += f"   ACTIVE: {', '.join(sorted(active))}"
    cv2.putText(view, text, (10, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255) if not active else (60, 60, 255), 2, cv2.LINE_AA)
    # Event ribbon: the whole recording at a glance, with the playhead.
    y = h - 8
    cv2.line(view, (10, y), (w - 10, y), (70, 70, 70), 3)
    for event in events:
        x1 = 10 + int((w - 20) * event["t_start"] / max(duration_s, 1e-9))
        x2 = 10 + int((w - 20) * event["t_end"] / max(duration_s, 1e-9))
        cv2.line(view, (x1, y), (max(x2, x1 + 2), y), (60, 60, 255), 3)
    px = 10 + int((w - 20) * t / max(duration_s, 1e-9))
    cv2.line(view, (px, y - 6), (px, y + 6), (255, 255, 255), 2)


def render_timeline(scored, cal: Calibration, events: list[dict],
                    out_path: Path, duration_s: float) -> Path:
    """One deviation strip per seat, with event bands drawn over it."""
    seats = [s.label for s in cal.seats]
    row_h, width, pad = 90, 1400, 120
    canvas = np.full((row_h * len(seats) + 40, width + pad, 3), 18, dtype=np.uint8)

    for index, label in enumerate(seats):
        top = index * row_h + 20
        colour = SEAT_COLOURS[index % len(SEAT_COLOURS)]
        cv2.putText(canvas, label, (8, top + row_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        series = [
            (w.window.t_start,
             max((z for (seat, _zone), z in w.deviation.items() if seat == label),
                 default=0.0))
            for w in scored
        ]
        if not series:
            continue
        peak = max(max(v for _, v in series), 1e-6)
        for t, value in series:
            x = pad + int(width * t / max(duration_s, 1e-9))
            height = int((row_h - 26) * min(value / peak, 1.0))
            cv2.line(canvas, (x, top + row_h - 14), (x, top + row_h - 14 - height),
                     colour, 1)

        for event in events:
            if event["seat_label"] != label:
                continue
            x1 = pad + int(width * event["t_start"] / max(duration_s, 1e-9))
            x2 = pad + int(width * event["t_end"] / max(duration_s, 1e-9))
            cv2.rectangle(canvas, (x1, top + 2), (max(x2, x1 + 2), top + row_h - 12),
                          ACTIVE_COLOUR, 1)
            cv2.putText(canvas, str(event["id"]), (x1 + 2, top + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACTIVE_COLOUR, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"peak z {peak:.1f}", (pad + width - 90, top + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def render_heatmap(scored, cal: Calibration, frame: np.ndarray,
                   out_path: Path) -> Path:
    """Seat footprints tinted by how much of the recording they were active for."""
    totals: dict[str, float] = {s.label: 0.0 for s in cal.seats}
    counts: dict[str, int] = {s.label: 0 for s in cal.seats}
    for window in scored:
        for (seat, _zone), z in window.deviation.items():
            if seat in totals:
                totals[seat] += max(z, 0.0)
                counts[seat] += 1
    peak = max(
        [totals[s] / counts[s] for s in totals if counts[s]] or [1.0]) or 1.0

    view = frame.copy()
    layer = view.copy()
    for seat in cal.seats:
        mean = totals[seat.label] / max(counts[seat.label], 1)
        intensity = min(mean / peak, 1.0)
        colour = (int(255 * (1 - intensity)), 40, int(255 * intensity))
        pts = np.array(seat.polygon, dtype=np.int32)
        cv2.fillPoly(layer, [pts], colour)
        cv2.putText(view, f"{seat.label} {mean:.2f}z",
                    (pts[:, 0].min() + 4, pts[:, 1].min() + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    blended = cv2.addWeighted(layer, 0.4, view, 0.6, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)
    return out_path


# ------------------------------------------------------------------ verify --

@contextmanager
def llama_server(cfg, port: int, binary: str, log_path: Path):
    """Start llama-server, wait for health, and always tear it down.

    The teardown is in a `finally` because the alternative -- a verifier crash
    leaving a model resident on the GPU -- is exactly the state that makes the
    next run fail for an unrelated-looking reason.
    """
    # Absolute paths throughout, because the server is launched with its own
    # directory as the working directory (see below) and a relative model path
    # would then resolve against the wrong place.
    model = Path(cfg.model_path).resolve()
    mmproj = Path(cfg.mmproj_path).resolve()
    if not model.exists() or not mmproj.exists():
        raise FileNotFoundError(f"verifier weights missing: {model}, {mmproj}")

    found = shutil.which(binary) or (str(Path(binary).resolve())
                                     if Path(binary).exists() else None)
    if found is None:
        raise FileNotFoundError(
            f"{binary} not found. Point --llama-binary at the llama-server "
            f"executable, or pass --skip verify."
        )
    exe = Path(found).resolve()

    # The shipped `llama-server.exe` is a 9 KB stub that loads
    # `llama-server-impl.dll` and the ggml backend DLLs from its own directory.
    # Launched from anywhere else, CreateProcess cannot resolve those and
    # reports WinError 2 -- "cannot find the file specified" -- naming the
    # executable, which does exist. Checking `Path(binary).exists()` therefore
    # proves nothing about whether it can start, and this run failed at the last
    # stage after 5 minutes of CV work for exactly that reason.
    #
    # So: run it from its own directory, and prove it launches before committing
    # to the slow startup path.
    workdir = exe.parent
    probe = subprocess.run([str(exe), "--version"], capture_output=True,
                           text=True, timeout=120, cwd=str(workdir))
    if probe.returncode != 0:
        raise RuntimeError(
            f"{exe} exists but will not run (exit {probe.returncode}): "
            f"{(probe.stderr or probe.stdout).strip()[:300]}"
        )
    print(f"    {(probe.stdout or probe.stderr).strip().splitlines()[0]}")

    command = [str(exe), "-m", str(model), "--mmproj", str(mmproj),
               "-c", str(cfg.n_ctx), "-ngl", str(cfg.n_gpu_layers),
               "--port", str(port), "--host", "127.0.0.1"]
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               cwd=str(workdir))
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 600
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {process.returncode}; "
                    f"see {log_path}")
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=3) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(2.0)
        else:
            raise TimeoutError(f"llama-server did not become healthy; see {log_path}")
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        log.close()


# --------------------------------------------------------------------- run --

def event_rows(conn, video_id: int) -> list[dict]:
    return [dict(r) for r in db.all_rows(
        conn,
        """
        SELECT e.id, e.t_start, e.t_end, e.peak_t, e.peak_deviation, e.score,
               e.evidence_quality, e.explanation, s.seat_label
        FROM event e JOIN seat s ON s.id = e.seat_id
        WHERE e.source_video_id = ? ORDER BY e.t_start
        """,
        (video_id,),
    )]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--name", default=None, help="run folder name")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--residency", choices=("sequential", "combined"),
                    default="sequential")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["overlay", "detect", "pose", "clips", "verify"])
    ap.add_argument("--events", type=int, default=None,
                    help="cap the events sent to the model stages")
    ap.add_argument("--overlay-fps", type=float, default=8.0)
    ap.add_argument("--llama-binary", default="llama-server")
    ap.add_argument("--llama-port", type=int, default=8081)
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 2

    name = args.name or args.video.stem[:24].strip().replace(" ", "_")
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(run_dir / "log.txt")
    sys.stdout = tee

    manifest: dict = {
        "video": str(args.video),
        "run_dir": str(run_dir),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "residency": args.residency,
        "skipped": args.skip,
        "stages": {},
    }
    stage_started = time.perf_counter()

    def stage(label: str, **facts) -> None:
        elapsed = time.perf_counter() - stage_started
        manifest["stages"][label] = {"seconds": round(elapsed, 2), **facts}
        print(f"  [{label}] {elapsed:.1f}s  " +
              "  ".join(f"{k}={v}" for k, v in facts.items()))

    try:
        cfg = Config(db_path=run_dir / "run.db", media_root=run_dir / "events")
        conn = db.connect(cfg.db_path)

        print(f"Run: {name}\n{'=' * 64}\n{args.video}\n")

        # -- 1. ingest ---------------------------------------------------
        print("1. ingest")
        stage_started = time.perf_counter()
        profile = ingest.probe(args.video)
        print(f"  {profile.summary()}")
        if not profile.has_mv:
            raise SystemExit("no codec motion vectors; the tier-1 scan cannot run")
        session_id = db.insert(conn, "session", name=f"run {name}")
        video_id = ingest.register(conn, profile, session_id, args.video.stem[:24])
        manifest["source"] = {
            "codec": profile.codec, "width": profile.width,
            "height": profile.height, "fps": round(profile.avg_fps, 3),
            "duration_s": round(profile.duration_s, 2), "vfr": profile.is_vfr,
            "mv_per_frame": round(profile.mv_per_frame, 1),
        }
        stage("ingest", video_id=video_id, duration_s=round(profile.duration_s, 1))

        # -- 2. calibration ----------------------------------------------
        print("\n2. calibration")
        stage_started = time.perf_counter()
        cal = build_calibration(args.video.name, profile.width, profile.height)
        cal.camera_id = int(db.one(
            conn, "SELECT camera_id FROM source_video WHERE id = ?",
            (video_id,))["camera_id"])
        calibration_id = cal_mod.save(conn, cal)
        cal.write_json(run_dir / "calibration.json")
        stage("calibration", seats=len(cal.seats), masks=len(cal.masks),
              calibration_id=calibration_id)

        # -- 3. motion + segmentation ------------------------------------
        print("\n3. motion scan and segmentation")
        stage_started = time.perf_counter()
        result = pipeline.run(conn, video_id, calibration_id, cfg)
        print(f"  {result.windows} windows, {result.motion_rows} zone rows, "
              f"{result.events} events "
              f"({result.candidate_fraction * 100:.1f}% of the recording)")
        print(f"  {profile.duration_s / max(result.seconds, 1e-9):.0f}x realtime")
        stage("motion", windows=result.windows, events=result.events,
              candidate_fraction=round(result.candidate_fraction, 4),
              realtime_factor=round(profile.duration_s / max(result.seconds, 1e-9), 1))

        # Re-derive the scored windows for the visualisations. Cheap relative to
        # the scan and keeps the renderers independent of the database schema.
        masks = cal_mod.rasterize(cal)
        windows = list(motion.scan(str(args.video), masks, cfg.motion))
        baselines = motion.learn_baselines(windows, cfg.baseline)
        scored = motion.apply_baselines(windows, baselines, cfg.motion, cfg.baseline)

        (run_dir / "motion").mkdir(exist_ok=True)
        with (run_dir / "motion" / "windows.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_start", "t_end", "seat", "zone", "deviation",
                             "regime", "residual", "area_ratio", "below_desk"])
            for window in scored:
                for (seat, zone), z in window.deviation.items():
                    stats = window.window.zones.get((seat, zone))
                    writer.writerow([
                        f"{window.window.t_start:.3f}", f"{window.window.t_end:.3f}",
                        seat, zone, f"{z:.4f}",
                        window.regime.get((seat, zone), "unknown"),
                        f"{stats.residual:.6f}" if stats else "",
                        f"{stats.area_ratio:.4f}" if stats else "",
                        f"{stats.below_desk:.4f}" if stats else "",
                    ])

        # -- 4. scoring ---------------------------------------------------
        print("\n4. scoring")
        stage_started = time.perf_counter()
        scoring.score_video(conn, video_id, cfg.score)
        events = event_rows(conn, video_id)
        (run_dir / "events").mkdir(exist_ok=True)
        with (run_dir / "events" / "events.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(events[0]) if events
                                    else ["id"])
            writer.writeheader()
            for event in events:
                writer.writerow(event)
        stage("scoring", events=len(events))
        for event in sorted(events, key=lambda e: -e["score"])[:10]:
            print(f"    [{event['id']:>3}] {event['seat_label']:<10} "
                  f"{event['t_start']:7.2f}-{event['t_end']:7.2f}s  "
                  f"score {event['score']:5.2f}  peak z {event['peak_deviation']:5.1f}")

        # -- 5. overlays --------------------------------------------------
        if "overlay" not in args.skip:
            print("\n5. overlays")
            stage_started = time.perf_counter()
            first = evidence.read_frames(str(args.video), [min(2.0, profile.duration_s / 2)])
            base_frame = first[0][1]
            cv2.imwrite(str(run_dir / "calibration.jpg"),
                        draw_calibration(base_frame, cal))
            render_timeline(scored, cal, events, run_dir / "motion" / "timeline.jpg",
                            profile.duration_s)
            render_heatmap(scored, cal, base_frame, run_dir / "motion" / "heatmap.jpg")
            marked = render_marked_video(
                str(args.video), run_dir / "overlay" / "marked.mp4", cal, events,
                profile.duration_s, args.overlay_fps)
            stage("overlay", marked_mb=round(marked.stat().st_size / 1e6, 1),
                  fps=args.overlay_fps)

        # -- 6. clips -----------------------------------------------------
        if "clips" not in args.skip:
            print("\n6. evidence clips")
            stage_started = time.perf_counter()
            made = clips.materialise(conn, video_id, cfg.media_root, cal,
                                     cfg.segment, limit=args.events)
            stage("clips", written=made)

        # -- 7. CV models -------------------------------------------------
        # One at a time by default. Each block loads, runs, and drops its model
        # before the next is constructed, so peak residency is one network.
        detector_cfg = cfg.detect
        if args.residency == "combined" and not {"detect", "pose"} & set(args.skip):
            print("\n7. detection + pose (co-resident, one decode pass)")
            stage_started = time.perf_counter()
            detector = detect.ONNXDetector("models/dfine/onnx/model.onnx")
            pose_model = pose.PoseEstimator(device="cpu")
            tracker = track.SeatTracker(cal)
            found = evidence.run(conn, video_id, cal, detector_cfg, detector,
                                 pose_model, tracker, limit=args.events)
            del detector, pose_model, tracker
            stage("detect+pose", events=len(found))
        else:
            if "detect" not in args.skip:
                print("\n7a. detection (detector resident alone)")
                stage_started = time.perf_counter()
                detector = detect.ONNXDetector("models/dfine/onnx/model.onnx")
                found = evidence.run(conn, video_id, cal, detector_cfg,
                                     detector=detector, limit=args.events)
                total = sum(len(e.detections) for e in found)
                abstained = sum(1 for e in found for d in e.detections if d.abstained)
                del detector
                stage("detect", events=len(found), detections=total,
                      abstained=abstained)
            if "pose" not in args.skip:
                print("\n7b. pose + tracking (pose resident alone)")
                stage_started = time.perf_counter()
                pose_model = pose.PoseEstimator(device="cpu")
                tracker = track.SeatTracker(cal)
                found = evidence.run(conn, video_id, cal, detector_cfg,
                                     pose_model=pose_model, tracker=tracker,
                                     limit=args.events)
                observations = sum(len(e.poses) for e in found)
                del pose_model, tracker
                stage("pose", events=len(found), observations=observations)

        scoring.score_video(conn, video_id, cfg.score)
        events = event_rows(conn, video_id)

        # -- 8. verifier, last --------------------------------------------
        if "verify" not in args.skip:
            print("\n8. verifier (Gemma, started last and shut down after)")
            stage_started = time.perf_counter()
            (run_dir / "verify").mkdir(exist_ok=True)
            ranked = sorted(events, key=lambda e: -e["score"])[:args.events or 5]

            with llama_server(cfg.verify, args.llama_port, args.llama_binary,
                              run_dir / "verify" / "server.log") as base_url:
                verifier = verify.GemmaVerifier(cfg.verify, base_url=base_url)
                done = failed = 0
                for event in ranked:
                    times = verify.select_frames(
                        event["t_start"], event["t_end"], event["peak_t"],
                        cfg.verify.contact_sheet_frames)
                    frames = evidence.read_frames(str(args.video), times)
                    sheet = verify.build_contact_sheet([f for _, f in frames])
                    sheet_dir = run_dir / "events" / str(event["id"])
                    sheet_dir.mkdir(parents=True, exist_ok=True)
                    sheet_path = sheet_dir / "sheet.jpg"
                    cv2.imwrite(str(sheet_path), sheet,
                                [cv2.IMWRITE_JPEG_QUALITY, 88])

                    metadata = {
                        "seat": event["seat_label"],
                        "event_duration_s": round(
                            event["t_end"] - event["t_start"], 2),
                        "peak_deviation_sd": round(event["peak_deviation"], 2),
                        "source_resolution": f"{profile.width}x{profile.height}",
                    }
                    try:
                        outcome = verifier.verify(
                            sheet_path, metadata, timeout=1800,
                            legend=verify.frame_legend(
                                [t for t, _ in frames], event["t_start"],
                                event["t_end"], event["peak_t"]),
                            frame_count=len(frames),
                        )
                        cleaned, removed = verify.sanitize(outcome)
                        verify.persist(conn, event["id"], "gemma-4-E4B", cleaned)
                        (run_dir / "verify" / f"{event['id']}.json").write_text(
                            json.dumps({**asdict(cleaned),
                                        "removed_for_verdict_language": removed},
                                       indent=2), encoding="utf-8")
                        done += 1
                        print(f"    event {event['id']}: "
                              f"{cleaned.object_assessment} / "
                              f"{cleaned.evidence_quality} "
                              f"(conf {cleaned.confidence:.2f})")
                    except Exception as exc:  # noqa: BLE001
                        # PRD 12.3: a verifier failure must leave the
                        # deterministic evidence intact and reviewable.
                        failed += 1
                        (run_dir / "verify" / f"{event['id']}.error.txt").write_text(
                            f"{type(exc).__name__}: {exc}", encoding="utf-8")
                        print(f"    event {event['id']}: FAILED  "
                              f"{type(exc).__name__}: {exc}")
            stage("verify", verified=done, failed=failed)

        manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["events"] = len(events)
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
        conn.close()
        print(f"\nwritten to {run_dir}")
        return 0

    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
