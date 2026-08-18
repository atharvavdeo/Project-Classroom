"""Generate the final report and the model selection report (PRD 15, 20).

    python tools/generate_final_report.py --run artifacts/runs/<run_id>

Writes `13_reports/final_report.md`, `model_selection_report.md`,
`quality_limitations.md` and `resource_report.json`.

PRD 20 requires the final report to choose one of three measured results:

  1. a configuration has sufficient evidence quality to build on;
  2. motion/ROI triage is useful but object detection is unreliable at this
     source scale;
  3. source quality, calibration or reference data prevents a claim, and the
     exact missing camera/data/training requirements are documented.

The choice is derived from the run's own artifacts, not written by hand. With no
reference manifest, (1) is unreachable by construction -- there is nothing to
measure evidence quality against -- so a run in that state reports (3) and lists
what is missing. That is the honest outcome for this corpus today, and making it
automatic means no future run can drift into claiming (1) without the labels
that would justify it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts  # noqa: E402
from pipeline.run_manifest import COMPLETE, RunManifest, RunStatus  # noqa: E402
from tools.render_pose_comparison import table  # noqa: E402


def load(paths, relative: str):
    path = paths.dir / relative
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def choose_outcome(reference: dict | None, detector: dict | None,
                   ranked: dict | None) -> tuple[int, str, list[str]]:
    """Pick one of PRD 20's three results from the run's own artifacts."""
    missing: list[str] = []
    if not reference:
        missing.append(
            "**A reference manifest.** PRD 14's schema, with independent "
            "phone/chit sightings carrying source hash, exact decoded PTS, "
            "class and source-pixel box. Without it there is nothing to measure "
            "recall or precision against, and no configuration can be selected.")
    if detector and not detector.get("taxonomy_approved"):
        missing.append(
            "**Product-owner approval of the taxonomy.** PRD 3 blocks detector "
            "selection and evaluation until the target and hard-negative class "
            "list is approved.")
    if detector:
        for entry in detector.get("unavailable", []):
            missing.append(
                f"**{entry['config_id']}** — {entry['reason']}")

    configurations = (reference or {}).get("configurations") or {}

    if not reference or not configurations:
        return 3, (
            "**(3) Source quality, calibration or reference data prevents a "
            "claim.** Specifically, reference data: this run completed motion, "
            "ROI, segmentation, tracking, pose and object inference over the "
            "full decodable duration and produced auditable artifacts for each, "
            "but no reference manifest was evaluated, so no accuracy statement "
            "of any kind is supportable. This is a statement about what is "
            "missing, not about what the pipeline found."), missing

    if not reference.get("complete"):
        missing.append(
            "**A complete reference manifest.** Rows in the supplied manifest "
            "failed validation, so every metric is computed over an unknown "
            "subset (PRD 14).")
        return 3, (
            "**(3) Reference data prevents a claim.** The manifest supplied had "
            "rows that failed validation, so the metrics below describe a "
            "subset of unknown composition. No configuration may be selected "
            "from them."), missing

    # The decision is made on the evaluation, not on how many events ranking
    # surfaced. An earlier version keyed (1) off `normal_priority`, and on a run
    # with zero true positives and 449 false positives it duly reported that a
    # configuration had "sufficient evidence quality to build on". Ranking counts
    # how much was surfaced; only the reference match says whether any of it was
    # right.
    best_tp = max((c["overall"].get("TP", 0) for c in configurations.values()),
                  default=0)
    precisions = [c["overall"].get("precision") for c in configurations.values()]
    defined = [p for p in precisions if p is not None]
    routing = max((c["overall"].get("candidate_routing_recall") or 0.0
                   for c in configurations.values()), default=0.0)

    if best_tp == 0:
        detail = (f"No configuration produced a single true positive against "
                  f"{sum(c['overall']['reference_count'] for c in configurations.values()) // max(len(configurations), 1)} "
                  f"reference row(s)")
        if defined:
            detail += f", at a best precision of {max(defined):.3f}"
        if routing > 0:
            return 2, (
                f"**(2) Motion/ROI triage is useful but object detection is "
                f"unreliable at this source scale.** {detail}. Motion routed "
                f"{routing * 100:.0f}% of the supportable references to a "
                f"candidate, so the triage stage is carrying its share; the "
                f"loss is downstream of it."), missing
        return 3, (
            f"**(3) Source quality prevents a claim.** {detail}, and motion "
            f"routed none of them to a candidate, so the loss is not "
            f"attributable to any single stage."), missing

    if not taxonomy_ok(detector):
        return 2, (
            "**(2) Motion/ROI triage is useful but no object claim may be "
            "made.** True positives exist, but PRD 3 blocks detector selection "
            "and evaluation until the product owner approves the taxonomy, so "
            "the precision and recall below are not a basis for a claim."), missing

    return 1, (
        f"**(1) A configuration has sufficient evidence quality to build on.** "
        f"Best true-positive count {best_tp}"
        + (f" at precision {max(defined):.3f}" if defined else "")
        + ". See the selection report for the measured comparison."), missing


def taxonomy_ok(detector: dict | None) -> bool:
    return bool(detector and detector.get("taxonomy_approved"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    paths = artifacts.open_run(args.run.parent, args.run.name)
    manifest = RunManifest.read(paths)
    status = load(paths, "RUN_STATUS.json") or {}

    pose = load(paths, "08_pose/comparison.json")
    detector = load(paths, "09_object/comparison.json")
    crops = load(paths, "09_object/crop_manifest_summary.json")
    confirmation = load(paths, "10_evidence/temporal_confirmation_summary.json")
    packets = load(paths, "10_evidence/packet_summary.json")
    gemma = load(paths, "11_gemma/summary.json")
    ranked = load(paths, "10_evidence/ranked_events.json")
    reference = load(paths, "12_evaluation/metrics.json")

    outcome_n, outcome_text, missing = choose_outcome(reference, detector, ranked)

    # ------------------------------------------------------ final report --
    lines = [f"# Final report — run `{manifest.run_id}`", ""]
    lines += ["## Measured result (PRD 20)", "", outcome_text, ""]

    if ranked:
        lines += ["## What this run surfaced", "", ranked["headline"], "",
                  table([["Events ranked", str(ranked["events_ranked"])],
                         ["Normal priority", str(ranked["normal_priority"])],
                         ["Retained at low priority",
                          str(ranked["low_priority_retained"])],
                         ["Excluded from automatic ordering",
                          str(ranked["suppressed_from_auto_ranking"])],
                         ["Need human review", str(ranked["needs_human_review"])],
                         ["Analysed", f"{ranked['analysed_ms'] / 1000.0:.1f} s"],
                         ["Failed", f"{ranked['failed_ms'] / 1000.0:.1f} s"],
                         ["Quality affected",
                          f"{ranked['quality_affected_ms'] / 1000.0:.1f} s"]],
                        ["Measure", "Value"]), "",
                  "Priority is a reading order over what this configuration "
                  "surfaced. It is not a probability that anything occurred, "
                  "and no row here is a finding about any person.", ""]

    lines += ["## Stage coverage", ""]
    stage_rows = []
    for entry in status.get("stages") or []:
        stage_rows.append([entry.get("stage", "—"), entry.get("state", "—"),
                           str(entry.get("reason") or "")[:70]])
    if stage_rows:
        lines += [table(stage_rows, ["Stage", "State", "Reason"]), ""]

    if crops:
        lines += ["## Source limitations", "",
                  f"- **{crops['insufficient_native_resolution']} of "
                  f"{crops['crops']} crops** "
                  f"({crops['insufficient_fraction'] * 100:.1f}%) fall below the "
                  f"native-resolution floor. No detector can answer for these.",
                  f"- Median crop magnification "
                  f"**{crops['median_magnification']}x**, maximum "
                  f"**{crops['max_magnification']}x**. Magnification is "
                  f"interpolation, not detail.",
                  f"- Crop policy mix: {crops['by_crop_config']}.", ""]
        if crops.get("fallback_reasons"):
            lines += [f"- Crops fell back from `pose_hand_context` for: "
                      f"{crops['fallback_reasons']}.", ""]

    if packets:
        lines += ["## Evidence packets", "",
                  f"{packets['valid']} of {packets['packets']} packets meet PRD "
                  f"11's composition requirements. {packets['note']}", ""]
        if packets.get("invalid_reasons"):
            lines += [table([[k, str(v)] for k, v in
                             packets["invalid_reasons"].items()],
                            ["Why a packet was rejected", "Count"]), ""]

    if gemma:
        lines += ["## Bounded VLM review", "",
                  table([["Reviews", str(gemma["reviews"])],
                         ["Attempted", str(gemma.get("attempted", 0))],
                         ["Not attempted", str(gemma.get("not_attempted", 0))],
                         ["Schema valid", str(gemma["schema_valid"])],
                         ["Schema invalid", str(gemma["schema_invalid"])],
                         ["Repaired", str(gemma["repaired"])],
                         ["Packets blocked before review",
                          str(gemma["blocked_packets"])],
                         ["Need human review", str(gemma["needs_human_review"])]],
                        ["Measure", "Value"]), "", gemma["note"], ""]

    lines += ["## What is missing", ""]
    if missing:
        lines += [f"{i}. {item}" for i, item in enumerate(missing, 1)] + [""]
    else:
        lines += ["Nothing blocking was recorded.", ""]

    lines += ["## Scope of any claim", "",
              "This system produces observational evidence for human review. It "
              "is not a cheating classifier, it does not infer intent, and no "
              "output here is a verdict about any person. The permitted "
              "vocabulary is observational throughout: `phone-like object "
              "detected`, `hand entered lap zone`, `ambiguous_seat`, "
              "`insufficient_visual_evidence`.", "",
              "The validated corpus is 1-17 people. The 30-60 person figure in "
              "the PRD is a capacity and design target, not a result this "
              "corpus supports.", ""]

    paths.write_json("13_reports/resource_report.json", {
        "run_id": manifest.run_id,
        "environment": manifest.to_dict().get("environment", {}),
        "models": manifest.to_dict().get("models", []),
        "stages": status.get("stages", []),
    })

    out = paths.dir / "13_reports/final_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    # -------------------------------------------- model selection report --
    selection = [f"# Model selection report — run `{manifest.run_id}`", "",
                 "PRD 5: no configuration is called best until every runnable "
                 "mandatory configuration has a matched report. PRD 15 places "
                 "selection at stage 12, against the reference manifest.", ""]

    selection += ["## Configuration availability", "",
                  table([[m["config_id"], m.get("family", "—"), m["state"],
                          str(m.get("licence", "—")),
                          str(m.get("reason") or "")[:60]]
                         for m in manifest.to_dict().get("models", [])],
                        ["Configuration", "Family", "State", "Licence",
                         "Reason"]), ""]

    if pose:
        selection += ["## Pose comparison", "",
                      f"Manifest equivalence: **{pose['manifest_equivalence']}** "
                      f"({pose['manifest_rows']} rows, digest "
                      f"`{pose['manifest_digest'][:16]}`)", "",
                      pose["equivalence_note"], "",
                      f"Selection: {pose['selection']}", ""]

    if detector:
        selection += ["## Detector and slicing comparison", "",
                      f"Crop equivalence: **{detector['crop_equivalence']}** "
                      f"({detector['crop_rows']} crops, digest "
                      f"`{detector['crop_digest'][:16]}`)", ""]
        rows = []
        for entry in detector["configurations"]:
            if entry.get("state") != "available":
                rows.append([entry["config_id"], entry["state"]] + ["—"] * 4)
                continue
            rows.append([entry["config_id"], entry["state"],
                         str(entry["merged_detections"]), str(entry["phone"]),
                         str(entry["secondary_paper_chit"]),
                         str(entry["hard_negative_detections"])])
        selection += [table(rows, ["Configuration", "State", "Merged", "Phone",
                                   "Chit", "Hard negatives"]), "",
                      "### Failure clusters", "",
                      "PRD 10 treats a phone called on a mouse, keyboard or "
                      "paper edge as a failure cluster to fix, not a false "
                      "positive to average away. Without reference labels the "
                      "class mix below is the shape of the output, not its "
                      "accuracy.", ""]
        for config_id, clusters in (detector.get("failure_clusters") or {}).items():
            selection += [f"**{config_id}** — {clusters.get('status')}: "
                          f"{clusters.get('by_class', clusters.get('clusters'))}",
                          ""]
        selection += [f"Selection: {detector['selection']}", ""]

    if confirmation and confirmation.get("by_status"):
        selection += ["## Temporal confirmation", "",
                      table([[k, str(v)] for k, v in
                             sorted(confirmation["by_status"].items())],
                            ["Status", "Groups"]), "",
                      f"{confirmation['retained_singletons']} singletons "
                      f"retained, {confirmation['deleted']} deleted. "
                      f"{confirmation['note']}", ""]

    selection += ["## Recommendation", "",
                  "No forced winner. " + (
                      "With no reference manifest, every comparison above is a "
                      "record of what each configuration produced, not of "
                      "which is correct. The next measurable step is the "
                      "reference manifest described in PRD 14."
                      if not reference else
                      "See the evaluation metrics in `12_evaluation/`."), ""]

    (paths.dir / "13_reports/model_selection_report.md").write_text(
        "\n".join(selection), encoding="utf-8")

    # ----------------------------------------------- quality limitations --
    quality = [f"# Quality limitations — run `{manifest.run_id}`", "",
               "Every limitation below is measured from this run's artifacts.", ""]
    if crops:
        quality += [
            f"## Source resolution", "",
            f"{crops['insufficient_native_resolution']} of {crops['crops']} "
            f"crops ({crops['insufficient_fraction'] * 100:.1f}%) are below the "
            f"native-resolution floor for an object call. These are abstentions "
            f"by construction: no detector, slicing configuration or VLM can "
            f"recover detail the source never captured.", "",
            f"Median magnification {crops['median_magnification']}x, maximum "
            f"{crops['max_magnification']}x.", ""]
    quality += ["## Blocking inputs", ""]
    quality += [f"- {item}" for item in missing] or ["- None recorded."]
    quality += ["", "## What cannot be claimed", "",
                "- No accuracy, precision, recall or false-positives-per-hour "
                "figure, in the absence of a reference manifest.",
                "- No statement about 30-60 person halls; the validated corpus "
                "is 1-17 people.",
                "- No long-recording or events-per-hour claim; the corpus is "
                "short pre-cut clips processed end to end.",
                "- No claim about any individual. The output is observational "
                "evidence for human review.", ""]
    (paths.dir / "13_reports/quality_limitations.md").write_text(
        "\n".join(quality), encoding="utf-8")

    RunStatus(paths, manifest.run_id).finish(
        "13_reports", COMPLETE, reason=f"PRD 20 measured result ({outcome_n})",
        reports=4)

    print(f"measured result: ({outcome_n})")
    print(f"written {paths.dir / '13_reports'}")
    for name in ("final_report.md", "model_selection_report.md",
                 "quality_limitations.md", "resource_report.json"):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
