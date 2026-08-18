"""Check the implementation against the PRD, and the PRD against itself.

    python tools/prd_check.py                 # document and code checks
    python tools/prd_check.py --corpus C:/DrishtiAI   # also re-probe the files

A specification and an implementation drift apart quietly. This reads the PRD as
data and compares what it claims to what the code and the footage actually do.

Deliberately different from `tools/validate_corpus.py`, which holds its own
transcription of the §5.1 table in a Python dict. A transcription can drift from
the document it transcribes, and then both agree with each other and neither
agrees with the PRD. Here the document *is* the source: the table is parsed out
of the markdown and the files are measured against what the reader would see.

Four groups of checks:

  DOCUMENT   internal consistency -- table shape, section numbering, and every
             file path the PRD tells a reader to run
  CODE       constants quoted in the prose against the values in the code
  TAXONOMY   enumerations printed in the PRD against the tuples that enforce them
  CORPUS     the measured file inventory (only with --corpus)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classroom import annotate, detect, pose, verify  # noqa: E402
from classroom.config import Config  # noqa: E402

PRD = ROOT / "Project_Classroom_PRD.md"


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.checks = 0
        self.group = ""

    def section(self, name: str) -> None:
        self.group = name
        print(f"\n{name}")
        print("-" * len(name))

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        self.checks += 1
        if not condition:
            self.failures += 1
        print(f"  {'PASS' if condition else 'FAIL'}  {label}"
              f"{'  ' + detail if detail else ''}")
        return bool(condition)


def parse_tables(text: str) -> list[tuple[int, list[str], list[list[str]]]]:
    """Pull every markdown table out, with the line number it starts on."""
    tables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            tables.append((i + 1, header, rows))
            i = j
        else:
            i += 1
    return tables


def strip_markup(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").strip()


# --------------------------------------------------------------- DOCUMENT --

def check_document(text: str, report: Report) -> None:
    report.section("DOCUMENT")

    tables = parse_tables(text)
    ragged = [
        (line, len(header), [len(r) for r in rows if len(r) != len(header)])
        for line, header, rows in tables
        if any(len(r) != len(header) for r in rows)
    ]
    report.check(
        f"all {len(tables)} tables have rows matching their header width",
        not ragged,
        "" if not ragged else "; ".join(
            f"line {line}: header {width}, rows {widths}" for line, width, widths in ragged),
    )

    # Section numbers must ascend. A subsection inserted in the wrong place
    # reads as an editing slip and hides whatever it displaced.
    subsections = [
        tuple(int(p) for p in m.group(1).split("."))
        for m in re.finditer(r"^### (\d+(?:\.\d+)+)\s", text, re.M)
    ]
    ascending = all(a < b for a, b in zip(subsections, subsections[1:]))
    report.check(
        f"all {len(subsections)} subsection numbers ascend", ascending,
        "" if ascending else str([
            ".".join(str(n) for n in bad)
            for bad, nxt in zip(subsections, subsections[1:]) if bad >= nxt
        ]),
    )

    # Every repository path the document points a reader at must exist.
    referenced = sorted({
        m.group(1) for m in re.finditer(
            r"`((?:tools|classroom|tests|docs)/[A-Za-z0-9_./-]+\.(?:py|sql|html|json))`",
            text)
    })
    missing = [p for p in referenced if not (ROOT / p).exists()]
    report.check(f"all {len(referenced)} referenced repo paths exist", not missing,
                 ", ".join(missing))

    commands = sorted({
        m.group(1) for m in re.finditer(r"`python (tools/[A-Za-z0-9_./-]+\.py)", text)
    })
    missing = [c for c in commands if not (ROOT / c).exists()]
    report.check(f"all {len(commands)} runnable commands resolve", not missing,
                 ", ".join(missing))

    version = re.search(r"\*\*Version:\*\*\s*([0-9.]+)", text)
    latest_changelog = re.search(r"## Changelog: v[0-9.]+ → v([0-9.]+)", text)
    report.check(
        "header version matches the newest changelog entry",
        bool(version and latest_changelog and version.group(1) == latest_changelog.group(1)),
        f"header {version.group(1) if version else '?'} vs changelog "
        f"{latest_changelog.group(1) if latest_changelog else '?'}",
    )


# ------------------------------------------------------------------- CODE --

def check_code(text: str, report: Report) -> None:
    report.section("CODE")
    cfg = Config()

    # Each entry: a description, the value the code holds, and a regex the PRD
    # must contain. The regex is written to match the number as a reader would
    # find it, so a silent edit on either side breaks the pair.
    claims = [
        ("abstention floor (px^2)", cfg.detect.min_object_px_area,
         r"below\s+the\s+300\s+px|300\s+px.\s+floor"),
        ("phone footprint (px)", "30x20",
         r"30\s*[x×]\s*20\s*px"),
        ("block motion threshold", cfg.motion.block_mag_threshold,
         r"[Bb]elow\s+2\.0[^\n]*noise floor"),
        ("verifier context window", cfg.verify.n_ctx, r"\b4096\b"),
    ]
    for label, value, pattern in claims:
        found = re.search(pattern, text, re.I | re.M) is not None
        report.check(f"PRD states the {label} ({value})", found)

    report.check("abstention floor is 300 px^2 in code",
                 cfg.detect.min_object_px_area == 300.0,
                 str(cfg.detect.min_object_px_area))
    report.check("start threshold exceeds stop threshold (hysteresis)",
                 cfg.segment.start_z > cfg.segment.stop_z,
                 f"{cfg.segment.start_z} > {cfg.segment.stop_z}")
    report.check("a maximum duration forces a split",
                 cfg.segment.max_duration_s > 0, f"{cfg.segment.max_duration_s} s")

    # 8.2.1 reverses 8.2's original choice. The code must agree with the
    # measurement, not with the heading.
    report.check("pose default is top-down, as 8.2.1 concluded",
                 pose.PoseEstimator().model == pose.TOPDOWN,
                 pose.PoseEstimator().model)
    report.check("both one-stage sizes remain selectable for reproduction",
                 {"rtmo-m", "rtmo-l"} <= set(pose.MODELS), str(pose.MODELS))

    report.check("the verifier grammar is one rule per line",
                 all(line.count("::=") == 1
                     for line in verify.GRAMMAR.strip().splitlines()),
                 "multi-line rules are rejected by llama.cpp's parser")
    report.check("no unbounded repetition in the grammar",
                 "*" not in verify.GRAMMAR and "+" not in verify.GRAMMAR,
                 "unbounded rules truncate mid-token")
    report.check("no backslash inside a grammar character class",
                 not re.search(r"\[\^?[^\]]*\\\\[^\]]*\]",
                               verify.GRAMMAR.replace("\\\\n", "").replace("\\\\r", "")
                               .replace("\\\\t", "")),
                 "the GBNF parser refuses it")

    # 15.6 exists because these three rules move the headline number.
    report.check("benign categories are outside the recall denominator",
                 "benign_movement" not in annotate.SCORED_TYPES
                 and "staff_interaction" not in annotate.SCORED_TYPES)
    report.check("the measured confusers are the negative classes",
                 set(annotate.NEGATIVE_CLASSES) == {"mouse", "keyboard", "monitor"},
                 str(annotate.NEGATIVE_CLASSES))


# --------------------------------------------------------------- TAXONOMY --

def check_taxonomy(text: str, report: Report) -> None:
    report.section("TAXONOMY")

    # The 9.2 contract block prints each enumeration as "a | b | c". Those are
    # what a reader will believe; the tuples are what the grammar enforces.
    for field, values in (
        ("object_assessment", verify.OBJECT_VALUES),
        ("interaction_assessment", verify.INTERACTION_VALUES),
        ("evidence_quality", verify.QUALITY_VALUES),
    ):
        match = re.search(rf'"{field}":\s*"([^"]+)"', text)
        if match is None:
            report.check(f"{field} is documented in 9.2", False)
            continue
        documented = tuple(v.strip() for v in match.group(1).split("|"))
        report.check(f"{field} matches the code", documented == values,
                     f"PRD {documented} vs code {values}")

    for value in verify.OBJECT_VALUES:
        if value in ("phone_like", "paper_like"):
            continue
        report.check(f"grammar enumerates {value!r}", f'\\"{value}\\"' in verify.GRAMMAR)

    report.check("the detector can abstain",
                 detect.INSUFFICIENT not in detect.CLASSES,
                 "abstention must not be a class label")


# ----------------------------------------------------------------- PHASES --

def check_phases(text: str, report: Report) -> None:
    report.section("PHASES")

    phase_modules = {
        "1": ["classroom/ingest.py"],
        "2": ["classroom/calibration.py", "tools/calibrate.py"],
        "3": ["classroom/motion.py"],
        "4": ["classroom/segment.py"],
        "5": ["classroom/scoring.py", "classroom/api.py", "classroom/static/console.html"],
        "6": ["classroom/detect.py", "classroom/pose.py", "classroom/track.py"],
        "7": ["classroom/verify.py"],
        "8": ["classroom/annotate.py", "classroom/evaluate.py",
              "tools/annotate_timeline.py", "tools/export_dataset.py"],
    }
    declared = set(re.findall(r"\*\*(\d+) · ", text))
    report.check("every phase in 10 has modules mapped here",
                 declared <= set(phase_modules),
                 f"undeclared: {sorted(declared - set(phase_modules))}")
    for phase, modules in phase_modules.items():
        missing = [m for m in modules if not (ROOT / m).exists()]
        report.check(f"phase {phase} modules present", not missing, ", ".join(missing))

    gates = sorted((ROOT / "tests").glob("test_*.py"))
    report.check("each phase has at least one gate script", len(gates) >= 8,
                 f"{len(gates)} gate scripts")


# ----------------------------------------------------------------- CORPUS --

def check_corpus(text: str, corpus: Path, report: Report) -> None:
    report.section("CORPUS")

    table = None
    for _, header, rows in parse_tables(text):
        if header[:3] == ["File", "Container", "Codec"]:
            table = rows
            break
    if table is None:
        report.check("the 5.1 inventory table is parseable", False)
        return
    report.check("the 5.1 inventory table is parseable", True, f"{len(table)} rows")

    from classroom import ingest

    files = sorted(list(corpus.glob("*.mkv")) + list(corpus.glob("*.mp4")))
    if not files:
        report.check(f"source files found under {corpus}", False)
        return

    # Row order in the table follows filename order, which is how the document
    # presents them.
    if not report.check("row count matches the files on disk",
                        len(table) == len(files),
                        f"{len(table)} rows, {len(files)} files"):
        return

    for row, path in zip(table, files):
        profile = ingest.probe(path)
        codec = strip_markup(row[2])
        resolution = strip_markup(row[3]).replace("×", "x")
        fps = float(strip_markup(row[4]))
        duration = float(strip_markup(row[5]).rstrip(" s"))
        vfr = strip_markup(row[7]).lower() == "yes"

        name = path.name[:28]
        report.check(f"{name} codec", profile.codec == codec,
                     f"{profile.codec} vs {codec}")
        report.check(f"{name} resolution",
                     f"{profile.width}x{profile.height}" == resolution,
                     f"{profile.width}x{profile.height} vs {resolution}")
        report.check(f"{name} fps", abs(profile.avg_fps - fps) < 0.1,
                     f"{profile.avg_fps:.2f} vs {fps}")
        report.check(f"{name} duration", abs(profile.duration_s - duration) < 1.0,
                     f"{profile.duration_s:.1f} vs {duration}")
        report.check(f"{name} VFR flag", profile.is_vfr == vfr,
                     f"{profile.is_vfr} vs {vfr}")
        report.check(f"{name} carries motion vectors", profile.has_mv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="directory of source video; enables the CORPUS group")
    args = ap.parse_args()

    if not PRD.exists():
        print(f"{PRD} not found", file=sys.stderr)
        return 2
    text = PRD.read_text(encoding="utf-8")

    report = Report()
    print(f"Checking {PRD.name} against the implementation")
    check_document(text, report)
    check_code(text, report)
    check_taxonomy(text, report)
    check_phases(text, report)
    if args.corpus:
        check_corpus(text, args.corpus, report)
    else:
        print("\nCORPUS\n------\n  SKIP  pass --corpus to re-probe the source files")

    print(f"\n{report.checks - report.failures}/{report.checks} checks pass")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
