"""Build the reviewer-facing dashboard for one or more runs.

    python tools/build_dashboard.py --run artifacts/runs/1505/03_mobile \
                                    --run artifacts/runs/1501/12_paper \
                                    --out artifacts/dashboard.html

Reads each run's `profiles.json`, `chit_evidence.json` and (if present)
`samples_sam3_adjudicated.jsonl`, applies `pipeline.verdict`, and writes a
single self-contained HTML page plus `review_queue.json`.

### Why a page and not a JSON file

A reviewer opening `chit_evidence.json` sees `kept_detections: 393` and cannot
tell whether one person cheated for a minute or four hundred people rustled
paper. The queue exists to answer, per person: *what did the system see, how
sure is it, and what is it unsure about* -- in sentences, ranked, with the
evidence frame attached.

### What the page will not do

* No single "cheating probability". We cannot compute one honestly, and a
  number invites a decision the evidence does not support.
* No ranking by detection count -- that ranks the busiest tracker.
* No hiding of abstentions. A person whose head was never resolvable says so;
  otherwise "no orientation signal" reads as "behaved normally", which is the
  opposite of what the data says.
* No `confirmed_incident`. Only a human writes that rung, through the
  Confirm / Dismiss controls.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import verdict as vd  # noqa: E402

# (verdict line, LED class). The words are what a reviewer should do, not what
# the system believes -- "review this" is an instruction; "suspicious" is a
# claim we are not entitled to make.
RUNG_LABEL = {
    vd.INCIDENT_CANDIDATE: ("Two independent signals — review", "flag"),
    vd.INDICATOR: ("One signal — worth a look", "watch"),
    vd.OBSERVATION: ("Observed", "watch"),
}


def read_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sam3_by_track(stage: Path) -> dict:
    """Per-track SAM 3 verdict counts, or {} if the run was never adjudicated."""
    path = stage / "samples_sam3_adjudicated.jsonl"
    if not path.is_file():
        return {}
    counts: dict = collections.defaultdict(collections.Counter)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for item in (row.get("near_hand_objects") or []):
                if isinstance(item, dict) and item.get("sam3"):
                    counts[row["track_id"]][item["sam3"]["verdict"]] += 1
    return {k: dict(v) for k, v in counts.items()}


def collect(run_dir: Path) -> tuple:
    stage = run_dir / "14_person_timeline"
    profiles = read_json(stage / "profiles.json", [])
    evidence = {e["track_id"]: e
                for e in read_json(stage / "chit_evidence.json", [])}
    sam3 = sam3_by_track(stage)
    summary = read_json(stage / "summary.json", {})

    verdicts = []
    for profile in profiles:
        track = profile["track_id"]
        verdicts.append(vd.assess(profile, evidence.get(track),
                                  sam3.get(track)))
    return verdicts, summary, bool(sam3)


def timestamp(ms: float) -> str:
    total = int(ms / 1000)
    return f"{total // 60:02d}:{total % 60:02d}"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;450;600;700&display=swap');

:root{
  --paper:#f1f2ef; --raise:#f7f8f5; --ink:#12161a; --ink-2:#4a5560;
  --rule:#d3d7d2; --rule-soft:#e4e7e2;
  --amber:#b4700c; --flag:#9e3529; --settled:#47695b;
  --flagwash:#f6ece9; --amberwash:#f7f0e2;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0d1114; --raise:#141a1f; --ink:#e6e9e6; --ink-2:#93a0a8;
    --rule:#242c33; --rule-soft:#1b2228;
    --amber:#d99a3c; --flag:#c86353; --settled:#7fa593;
    --flagwash:#1e1512; --amberwash:#1c1710;
  }
}
:root[data-theme="dark"]{
  --paper:#0d1114; --raise:#141a1f; --ink:#e6e9e6; --ink-2:#93a0a8;
  --rule:#242c33; --rule-soft:#1b2228;
  --amber:#d99a3c; --flag:#c86353; --settled:#7fa593;
  --flagwash:#1e1512; --amberwash:#1c1710;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  font-size:15px;line-height:1.6;font-weight:400}
.wrap{max-width:960px;margin:0 auto;padding:44px 22px 96px}

.mast{border-bottom:2px solid var(--ink);padding-bottom:18px;
  display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap}
h1{font-size:27px;font-weight:700;letter-spacing:-.015em;margin:0;text-wrap:balance}
.mast .id{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;
  color:var(--ink-2);letter-spacing:.06em;text-transform:uppercase;text-align:right}
.standfirst{max-width:64ch;color:var(--ink-2);margin:16px 0 0;font-size:15px}
.standfirst b{color:var(--ink);font-weight:600}

.tally{display:flex;gap:0;margin:30px 0 6px;border:1px solid var(--rule);
  border-radius:2px;overflow:hidden;flex-wrap:wrap}
.tally div{flex:1 1 118px;padding:13px 16px;border-right:1px solid var(--rule)}
.tally div:last-child{border-right:0}
.tally b{display:block;font-family:"IBM Plex Mono",monospace;font-size:23px;
  font-weight:500;font-variant-numeric:tabular-nums;line-height:1.2}
.tally span{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2)}
.tally .hi b{color:var(--flag)}
.tally .mid b{color:var(--amber)}

h2{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-2);
  margin:44px 0 0;padding-bottom:8px;border-bottom:1px solid var(--rule)}

.entry{display:grid;grid-template-columns:96px 1fr;gap:0 22px;
  padding:20px 0;border-bottom:1px solid var(--rule-soft)}
.gutter{font-family:"IBM Plex Mono",monospace;font-size:12px;
  font-variant-numeric:tabular-nums;color:var(--ink-2);line-height:1.7}
.gutter .tc{color:var(--ink);font-weight:500;font-size:13px}
.gutter .src{display:block;margin-top:3px;font-size:10.5px;letter-spacing:.04em}
.led{width:9px;height:9px;display:inline-block;margin-right:7px;
  vertical-align:1px;border-radius:1px}
.led.flag{background:var(--flag)}
.led.watch{background:var(--amber)}
.subject{font-size:17px;font-weight:600;letter-spacing:-.01em;margin:0}
.verdict{font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;margin:2px 0 0}
.verdict.flag{color:var(--flag)}
.verdict.watch{color:var(--amber)}
.because{margin:12px 0 0;padding:0;list-style:none}
.because li{position:relative;padding-left:15px;margin:0 0 6px;max-width:64ch}
.because li::before{content:"";position:absolute;left:0;top:.62em;
  width:6px;height:1.5px;background:var(--amber)}
.meta{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0 0;
  font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-2);
  letter-spacing:.03em}
.caveat{margin:12px 0 0;padding:10px 13px;background:var(--amberwash);
  border-left:2px solid var(--amber);font-size:13.5px;color:var(--ink-2);max-width:66ch}
.caveat b{color:var(--ink);font-weight:600}
.acts{display:flex;gap:7px;margin:15px 0 0}
button{font:inherit;font-size:12.5px;font-weight:450;padding:5px 13px;
  border:1px solid var(--rule);background:transparent;color:var(--ink);
  border-radius:2px;cursor:pointer;transition:border-color .12s,color .12s}
button:hover{border-color:var(--ink-2)}
button.pri{border-color:var(--flag);color:var(--flag)}
button:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
.entry.done{opacity:.44}
.stamp{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--settled);
  letter-spacing:.05em;margin:13px 0 0}

.ladder{width:100%;border-collapse:collapse;font-size:13.5px;margin:14px 0 0}
.ladder th,.ladder td{text-align:left;padding:9px 12px 9px 0;
  border-bottom:1px solid var(--rule-soft);vertical-align:top}
.ladder th{font-family:"IBM Plex Mono",monospace;font-size:10px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-2);font-weight:600}
.ladder td:first-child{font-family:"IBM Plex Mono",monospace;font-size:12px;
  white-space:nowrap;font-weight:500}
.scroll{overflow-x:auto}
.colophon{margin:40px 0 0;padding:20px 0 0;border-top:2px solid var(--ink);
  font-size:13.5px;color:var(--ink-2);max-width:70ch}
.colophon b{color:var(--ink);font-weight:600}
.empty{padding:34px 0;color:var(--ink-2);max-width:64ch}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media (max-width:620px){
  .entry{grid-template-columns:1fr;gap:9px}
  .gutter{display:flex;gap:12px;align-items:baseline}
  .gutter .src{margin-top:0}
}
"""

JS = """
document.addEventListener('click',e=>{
  const b=e.target.closest('button'); if(!b) return;
  const c=b.closest('.entry'); const a=b.dataset.act;
  c.classList.add('done');
  const m=document.createElement('p'); m.className='stamp';
  m.textContent=(a==='confirm'?'CONFIRMED BY REVIEWER':
    a==='dismiss'?'DISMISSED BY REVIEWER':'FLAGGED FOR A BETTER VIEW')+
    ' — held in this page only, not written back to the run.';
  c.querySelector('.acts').replaceWith(m);
});
"""


def card(item, run_label: str) -> str:
    """One log entry. The timecode is the identifier -- it is what a reviewer
    navigates by -- so it sits in the gutter rather than being decorated with
    an invented sequence number."""
    verdict_line, led = RUNG_LABEL.get(item.rung, ("Observed", "watch"))
    who = (f"Seat {item.seat}" if item.seat != "unattributed"
           else f"Person #{item.track_id}")

    out = ['<article class="entry">',
           '<div class="gutter">',
           f'<span class="tc">{timestamp(item.first_seen_ms)}</span>'
           f'<span class="src">{html.escape(run_label)}</span>',
           "</div>",
           "<div>",
           f'<p class="subject"><span class="led {led}"></span>'
           f'{html.escape(who)}</p>',
           f'<p class="verdict {led}">{html.escape(verdict_line)}</p>']

    if item.indicators:
        out.append('<ul class="because">')
        for indicator in item.indicators:
            out.append(f"<li>{html.escape(indicator.detail)}</li>")
        out.append("</ul>")
        out.append('<div class="meta">')
        for modality in item.modalities:
            out.append(f"<span>{html.escape(modality.replace('_', ' '))}</span>")
        out.append(f"<span>seen {item.coverage:.0%} of recording</span>")
        out.append(f"<span>{timestamp(item.first_seen_ms)}"
                   f"&ndash;{timestamp(item.last_seen_ms)}</span>")
        out.append("</div>")

    for caveat in item.caveats:
        out.append(f'<p class="caveat"><b>Caveat.</b> {html.escape(caveat)}</p>')

    out.append('<div class="acts">'
               '<button class="pri" data-act="confirm">Confirm incident</button>'
               '<button data-act="dismiss">Dismiss</button>'
               '<button data-act="better">Need better view</button></div>')
    out.append("</div></article>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/dashboard.html")
    args = ap.parse_args()

    everything = []
    totals = collections.Counter()
    refereed = {}
    for run_dir in args.run:
        verdicts, summary, has_sam3 = collect(run_dir)
        label = run_dir.name
        refereed[label] = has_sam3
        totals["people"] += summary.get("people", len(verdicts))
        for item in verdicts:
            totals[item.rung] += 1
            everything.append((item, label))

    reportable = [(v, r) for v, r in everything if v.is_reportable]
    reportable.sort(key=lambda p: (-vd.RUNG_ORDER[p[0].rung], -p[0].priority))

    red = sum(1 for v, _ in reportable if v.rung == vd.INCIDENT_CANDIDATE)
    amber = sum(1 for v, _ in reportable if v.rung == vd.INDICATOR)

    runs_label = ", ".join(r.name for r in args.run)
    body = [
        '<div class="wrap">',
        '<header class="mast">',
        "<h1>Invigilation Review Queue</h1>",
        f'<div class="id">Project Classroom · Drishti AI PS2<br>'
        f'{html.escape(runs_label)}</div>',
        "</header>",
        '<p class="standfirst">People are ranked by how much '
        '<b>independent</b> evidence supports a second look — evidence from '
        'different sensing methods, not the same detector firing repeatedly. '
        '<b>Nothing on this page is a finding of cheating.</b> Only a human '
        'watching the footage can confirm one.</p>',
        '<div class="tally">',
        f'<div class="hi"><b>{red}</b><span>Review</span></div>',
        f'<div class="mid"><b>{amber}</b><span>Worth a look</span></div>',
        f'<div><b>{totals["people"]}</b><span>People tracked</span></div>',
        # NOT "Cleared". Most of this number is tracks the system could not
        # see well enough to assess -- fragments, unresolvable heads, objects
        # SAM 3 could not confirm either way. Calling that "cleared" asserts an
        # all-clear the evidence does not support. See docs/OUTPUT_CONTRACT.md
        # and the `needs_better_view` state.
        f'<div><b>{len(everything) - len(reportable)}</b>'
        f'<span>Not assessed</span></div>',
        '<div><b>0</b><span>Confirmed</span></div>',
        "</div>",
    ]

    if not reportable:
        body.append('<p class="empty">Nothing reached the reporting threshold '
                    'in these recordings. That is a result, not an error — but '
                    'read the limits below before treating it as an '
                    'all-clear.</p>')
    else:
        body.append("<h2>Queue</h2>")
    for item, label in reportable:
        body.append(card(item, label))

    body.append("<h2>How an entry reaches this page</h2>")
    body.append('<div class="scroll"><table class="ladder">'
                "<tr><th>Level</th><th>What it means</th><th>Decided by</th></tr>"
                "<tr><td>Review</td><td>Two indicators from <em>different</em> "
                "sensing methods coincide on the same person</td>"
                "<td>Machine, ranked</td></tr>"
                "<tr><td>Worth a look</td><td>One indicator survived every "
                "gate</td><td>Machine</td></tr>"
                "<tr><td>Not assessed</td><td>No indicator survived the "
                "gates — <em>or</em> the recording never showed enough to "
                "judge. This is not an all-clear</td><td>Machine</td></tr>"
                "<tr><td>Confirmed</td><td>A person watched the footage and "
                "agreed</td><td><b>Human only — never the system</b></td>"
                "</tr></table></div>")

    unrefereed = [k for k, v in refereed.items() if not v]
    body.append('<div class="colophon">')
    body.append("<b>What this page does not claim.</b> No ground-truth record "
                "exists for these recordings, so nothing here is a precision "
                "or accuracy measurement — every count is what the system "
                "reported, not what was true. The system cannot re-identify a "
                "person across a tracking break, so one person may appear as "
                "several entries, and a person who leaves and returns is not "
                "verified to be the same person. ")
    if unrefereed:
        body.append("Object evidence in "
                    f"<b>{html.escape(', '.join(unrefereed))}</b> was not "
                    "reviewed by the second model and should be read as "
                    "unconfirmed. ")
    body.append("Object detection runs against a hosted service, which "
                "conflicts with the offline requirement in Problem "
                "Statement 2. ")
    # Stated because the buttons look like they record something and do not.
    # A reviewer who believes their decision was saved, and it was not, is a
    # worse outcome than having no buttons at all.
    body.append("<b>Reviewer decisions on this page are not saved.</b> The "
                "buttons change only what this browser shows; nothing is "
                "written back. Persisted, append-only review is specified in "
                "docs/OUTPUT_CONTRACT.md and is not yet built. This page also "
                "still renders the older <code>pipeline/verdict.py</code> "
                "ladder rather than the frozen output contract.")
    body.append("</div></div>")

    page = (f"<!doctype html><html lang=en><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Invigilation Review Queue</title>"
            f"<style>{CSS}</style>{''.join(body)}<script>{JS}</script></html>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")

    queue = [{"run": label, "track_id": v.track_id, "seat": v.seat,
              "rung": v.rung, "priority": round(v.priority, 3),
              "first_seen_ms": v.first_seen_ms, "last_seen_ms": v.last_seen_ms,
              "coverage": round(v.coverage, 4),
              "modalities": v.modalities,
              "indicators": [{"modality": i.modality, "kind": i.kind,
                              "detail": i.detail,
                              "strength": round(i.strength, 3)}
                             for i in v.indicators],
              "caveats": v.caveats}
             for v, label in reportable]
    json_path = args.out.with_name("review_queue.json")
    json_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")

    print(f"{len(everything)} tracks -> {len(reportable)} reportable "
          f"({red} review, {amber} worth a look)")
    for item, label in reportable[:10]:
        print(f"  [{item.rung:18s}] {label} #{item.track_id} "
              f"p={item.priority:.1f} {','.join(item.modalities)}")
    print(f"\n-> {args.out}")
    print(f"-> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
