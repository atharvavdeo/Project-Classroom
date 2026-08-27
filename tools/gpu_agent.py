"""Run pipeline jobs on the GPU box, pulling work from the console.

This is the process that turns "somebody uploaded a recording" into "a run
exists". It runs on the machine with the 3090 and needs **no inbound
connectivity**: it polls the console for work, so there is no port to forward,
no public IP to arrange and no tunnel to keep alive. A box behind a college NAT
can run this unchanged.

The loop, once per job:

    claim  -> download the recording -> run the pipeline stages
           -> build the console bundles and review crops
           -> upload the artifacts -> report complete

Progress is reported as it goes, so the console can show which stage is
executing rather than a spinner. Every report carries the claim token issued at
claim time; an agent whose job was reassigned cannot overwrite the new
holder's status.

A failure is reported as a failure, with the stage and the error text. It is
never reported as a completed run with no findings -- an empty dashboard reads
as "nothing was found", which is a very different statement from "the pipeline
died in stage 8".

Usage on the GPU box:

    set DRISHTI_CONSOLE=https://console.example.workers.dev
    set ROBOFLOW_API_KEY=...
    python tools/gpu_agent.py --once        # take one job and stop
    python tools/gpu_agent.py               # poll forever
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Run as `python tools/gpu_agent.py`, the repository root is not on the path,
# so `from pipeline import artifacts` fails with ModuleNotFoundError. The
# stage tools are launched as subprocesses with cwd=REPO and are unaffected;
# this is only for the one import this file makes itself.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The stages this agent runs, in order, as (label, argv-builder, share of the
# progress bar).
#
# These are the tools that actually exist and actually produced the committed
# runs. An earlier version of this file invoked `python -m pipeline.run --stage
# <name>`, which is a module that does not exist -- it would have failed on the
# first job with a ModuleNotFoundError. There is no single stage-dispatch entry
# point in this repository; the runs are a chain of four tools plus the three
# bundlers, and that chain is written out here rather than inferred.
#
# The weights are rough wall-clock proportions from the 1512 run. Stage 14 does
# person detection, tracking and pose in one pass and dominates everything
# else; SAM 3 is network-bound and comes third. A bar that advanced linearly
# across seven stages would sit near 15% for most of a job.
EXPERIMENT = "configs/experiments/product_zero.yaml"


def _stages(run_dir, source, run_key):
    """The pipeline chain for one recording, as argv lists."""
    py = sys.executable
    return [
        # Stage 0. Preflight is not optional bookkeeping: it creates the run
        # directory, writes RUN_MANIFEST.json, and probes every model for
        # launchability. Every later stage calls `RunManifest.read`, so without
        # it stage 14 dies immediately on a missing manifest -- measured.
        #
        # It also records which models resolved to CUDA and which fell back,
        # which is the only durable evidence of whether a run was actually on
        # the GPU.
        ("00_preflight", 0.02,
         [py, "tools/preflight_rtx3090.py", "--experiment", EXPERIMENT,
          "--run-id", run_key, "--sources", str(source)]),
        # Stage 14: sampling, person detection, IoU tracking, AlphaPose, and
        # D-FINE objects associated to wrists. Everything on the GPU.
        ("14_person_timeline", 0.62,
         [py, "tools/run_person_timeline.py", str(source), "--run", str(run_dir),
          "--pose-device", "cuda"]),
        # 14b: the Roboflow chit detector over the hand crops. Online.
        ("14b_chit_detector", 0.10,
         [py, "tools/attach_chit_detections.py", str(source), "--run", str(run_dir)]),
        # 14c: geometric gates. Cheap, local, and it is what keeps SAM 3 from
        # being asked about every keyboard in the hall.
        ("14c_gates", 0.02,
         [py, "tools/score_chit_evidence.py", "--run", str(run_dir)]),
        # 14d: SAM 3 as referee, episode-sampled. Online, and the slowest thing
        # that is not the GPU.
        ("14d_sam3_adjudication", 0.18,
         [py, "tools/adjudicate_with_sam3.py", str(source), "--run", str(run_dir),
          "--max-per-episode", "3", "--save-annotated", "250"]),
        # 15: evidence records and the fused per-person outcomes.
        ("15_evidence_records", 0.08,
         [py, "tools/build_evidence_records.py", "--run", str(run_dir)]),
    ]


class Console:
    """The console's job API, over plain HTTP."""

    def __init__(self, base: str, agent: str, agent_token: str | None = None):
        self.base = base.rstrip("/")
        self.agent = agent
        # Authorises claiming work. Without it a public console would let a
        # stranger drain the queue -- and on a GPU billed by the hour, spend
        # someone else's money.
        self.agent_token = agent_token
        self.token: str | None = None

    def _call(self, path: str, body: dict | None = None, method="POST"):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as res:
            return json.loads(res.read().decode())

    def claim(self) -> tuple[dict, dict] | None:
        doc = self._call("/api/jobs/claim",
                         {"agent": self.agent, "agent_token": self.agent_token})
        if not doc.get("job"):
            return None
        self.token = doc["claim_token"]
        return doc["job"], doc["video"]

    def report(self, job_id, state="running", stage=None, progress=None,
               run_key=None, error=None):
        return self._call(
            f"/api/jobs/{job_id}/status",
            {
                "claim_token": self.token,
                "state": state,
                "stage": stage,
                "progress": progress,
                "run_key": run_key,
                "error": error,
            },
        )

    def download(self, media_url: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        url = media_url if media_url.startswith("http") else f"{self.base}{media_url}"
        with urllib.request.urlopen(url, timeout=600) as res, \
                target.open("wb") as out:
            shutil.copyfileobj(res, out, length=1 << 20)
        if not target.stat().st_size:
            raise RuntimeError(f"downloaded 0 bytes from {url}")
        return target


def run_stage(cmd: list[str], stage: str, produces: Path | None = None) -> None:
    """Run one stage, and fail loudly if it fails.

    A non-zero exit normally means the stage did not produce what the next one
    reads, and continuing would surface later as an empty result rather than as
    the error it is. So the default is to raise.

    `produces` is the one exception, and it exists for preflight. Preflight
    exits 1 when *any* declared configuration is unavailable, which is right
    for its own purpose -- it is a launchability report -- but on this corpus
    the only missing one is the local Gemma VLM checkpoint, and that stage was
    retired in favour of hosted inference. Treating that as fatal would block
    every run over a model nothing in this chain loads. When `produces` is
    given and the file exists, the stage did its job whatever it thought of the
    model table; if a CV model really is missing, stage 14 fails on it a moment
    later with a specific error, which is a better place to find out.
    """
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode == 0:
        return
    if produces is not None and produces.exists():
        print(f"  ({stage} exited {proc.returncode} but produced "
              f"{produces.name}; continuing)", flush=True)
        return
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
    raise RuntimeError(f"{stage} exited {proc.returncode}:\n" + "\n".join(tail))


def probe_frame_size(video: Path) -> tuple[int, int]:
    """The source frame size, measured from the media itself."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video} to measure its frame size")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not w or not h:
        raise RuntimeError(f"{video} reported a {w}x{h} frame size")
    return w, h


def process(console: Console, job: dict, video: dict, *, keep: bool) -> str:
    """Take one job from claimed to a finished run key."""
    job_id = job["id"]
    run_key = f"job{job_id}"
    runs_root = REPO / "artifacts" / "runs"
    run_dir = runs_root / run_key

    # Run ids are immutable and `create_run` refuses an existing directory, so
    # a retry of the same job has to start from a clean one. Only ever the
    # directory this agent created for this job id.
    if run_dir.exists():
        print(f"  clearing previous attempt at {run_dir}", flush=True)
        shutil.rmtree(run_dir)

    source = console.download(
        video["media_url"],
        REPO / "artifacts" / "incoming" / f"job{job_id}{Path(video['media_url']).suffix}",
    )
    print(f"  downloaded {source} ({source.stat().st_size / 1e6:.1f} MB)", flush=True)

    done = 0.0
    for stage, weight, argv in _stages(run_dir, source, run_key):
        console.report(job_id, "running", stage=stage, progress=round(done, 3))
        print(f"[{done:5.0%}] {stage}", flush=True)
        run_stage(
            argv,
            stage,
            produces=run_dir / "RUN_MANIFEST.json" if stage == "00_preflight" else None,
        )
        done += weight

    # The console reads three derived artifacts, not the run directory itself.
    console.report(job_id, "running", stage="bundling", progress=0.96)
    out = REPO / "console" / "public"
    run_stage([sys.executable, "tools/build_console_bundle.py",
               "--run", str(run_dir), "--out", str(out / "data" / f"{run_key}.json")],
              "bundle")
    # The overlay bundler refuses to guess a frame size, and rightly so -- a
    # bundle built against the wrong one draws every box in the wrong place,
    # silently. It normally reads the size from the ingest inventory, but a run
    # created by preflight has no `source/video_inventory.jsonl`, so the size is
    # measured from the media here and passed explicitly.
    width, height = probe_frame_size(source)
    print(f"  source frame size {width}x{height}", flush=True)
    run_stage([sys.executable, "tools/build_overlay_bundle.py",
               "--run", str(run_dir),
               "--frame-size", f"{width}x{height}",
               "--out", str(out / "data" / f"{run_key}.overlay.json")],
              "overlay")
    run_stage([sys.executable, "tools/build_review_crops.py",
               "--run", str(run_dir), "--video", str(source),
               "--out", str(out / "crops" / run_key)],
              "crops")

    console.report(job_id, "running", stage="uploading", progress=0.99)
    uploader = REPO / "console" / "tools" / "upload-assets.mjs"
    if os.environ.get("DRISHTI_R2_BUCKET") and uploader.exists():
        run_stage(["node", str(uploader), "--only", run_key,
                   "--bucket", os.environ["DRISHTI_R2_BUCKET"]], "upload")

    if not keep:
        source.unlink(missing_ok=True)
    return run_key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--console", default=os.environ.get("DRISHTI_CONSOLE",
                                                        "http://localhost:5179"))
    ap.add_argument("--agent", default=f"{platform.node()}/gpu")
    ap.add_argument("--once", action="store_true",
                    help="take at most one job, then exit")
    ap.add_argument("--poll", type=float, default=10.0,
                    help="seconds between polls when there is no work")
    ap.add_argument("--keep-source", action="store_true",
                    help="do not delete the downloaded recording afterwards")
    args = ap.parse_args()

    agent_token = os.environ.get("DRISHTI_AGENT_TOKEN")
    if not agent_token:
        print("DRISHTI_AGENT_TOKEN is not set; the console will refuse to hand "
              "out work.", file=sys.stderr)
        return 2
    console = Console(args.console, args.agent, agent_token)
    print(f"agent {args.agent} -> {console.base}", flush=True)

    while True:
        try:
            claimed = console.claim()
        except urllib.error.URLError as e:
            # The console being unreachable is not a job failure. Keep polling:
            # a restarting server or a dropped link must not consume the queue.
            print(f"  console unreachable ({e}); retrying", flush=True)
            time.sleep(args.poll)
            continue

        if not claimed:
            if args.once:
                print("no queued jobs", flush=True)
                return 0
            time.sleep(args.poll)
            continue

        job, video = claimed
        print(f"claimed job {job['id']} for {video['name']}", flush=True)
        try:
            run_key = process(console, job, video, keep=args.keep_source)
            console.report(job["id"], "complete", stage="done", progress=1.0,
                           run_key=run_key)
            print(f"job {job['id']} complete -> {run_key}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the agent must never die here
            # Report it. A crashed agent that says nothing leaves the job
            # 'claimed' forever, which looks identical to "still working".
            print(f"job {job['id']} FAILED: {exc}", flush=True)
            try:
                console.report(job["id"], "failed", error=str(exc)[:2000])
            except Exception as report_error:  # noqa: BLE001
                print(f"  could not report failure: {report_error}", flush=True)

        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
