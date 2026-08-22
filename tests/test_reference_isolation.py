"""Phase 7 gate: the reference manifest is unreachable before stage 12 (PRD 15).

PRD 15 asks for this test by name:

> Reference manifest becomes readable only in stage 12. Add an automated test
> that fails if a candidate/crop row includes `reference_id` or reference
> annotations.

The leak this prevents rarely arrives as someone reading the answers on purpose.
It arrives as a `reference_id` attached to a candidate row "just for debugging",
which survives review because it changes nothing, and which three commits later
is the field a threshold is tuned against. By then every metric in the run is
measuring a pipeline that was shown the answers, and nothing in the artifacts
says so.

So the check runs over the **artifacts**, not the code. A field that never
reaches a JSON row cannot influence a later stage; a field that does is caught
whatever the intention behind it.

The test also verifies the manifest loader itself: PRD 14 requires invalid rows
to be recorded in `validation_errors.json` and the evaluation marked incomplete,
never silently skipped. A manifest that drops twelve malformed rows and reports
clean metrics over the remaining forty is the quiet version of the same failure.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.reference_manifest import (  # noqa: E402
    ISOLATED_STAGES, MARKERS, VISIBILITY, assert_isolated, leaked_fields, load,
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


GOOD_ROW = {
    "reference_id": "REF-001",
    "source_video_sha256": "a" * 64,
    "source_pts_ms": 12000.0,
    "class": "phone",
    "annotation_source": "reviewer-a, independent pass",
    "frame_index": 300,
    "bbox_xyxy_source": [200.0, 500.0, 232.0, 522.0],
    "seat_id": "S-61",
    "visibility": "partially_occluded",
    "notes": "",
}


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def main() -> int:
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # ------------------------------------------------- leak detection --
        print("reference-looking fields are found wherever they are nested")
        ok &= check("a top-level reference_id",
                    leaked_fields({"reference_id": "R1"}) == ["reference_id"])
        ok &= check("nested inside an object",
                    "meta.reference_class" in
                    leaked_fields({"meta": {"reference_class": "phone"}}))
        ok &= check("inside a list of objects",
                    bool(leaked_fields({"rows": [{"ground_truth": True}]})))
        ok &= check("ordinary rows are clean",
                    not leaked_fields({"event_id": "EV-1", "seat_id": "S-61",
                                       "score": 0.4, "preference": "x"}),
                    "'preference' contains 'reference' but is not a marker")
        for marker in ("reference_id", "ground_truth", "is_true_positive"):
            ok &= check(f"{marker!r} is a marker", marker in MARKERS)

        # ------------------------------------------------- clean run folder --
        print("\na clean run passes the isolation check")
        clean = root / "clean"
        for stage in ISOLATED_STAGES:
            (clean / stage).mkdir(parents=True, exist_ok=True)
        write_jsonl(clean / "05_motion_roi/candidate_manifest.jsonl",
                    [{"event_id": "EV-1", "seat_id": "S-61",
                      "start_pts_ms": 1000.0, "end_pts_ms": 2000.0}])
        write_jsonl(clean / "09_object/object_crop_manifest.jsonl",
                    [{"crop_id": "c1", "pts_ms": 1500.0, "x1": 0.0}])
        (clean / "08_pose/comparison.json").write_text(
            json.dumps({"manifest_digest": "abc", "configurations": []}),
            encoding="utf-8")
        ok &= check("no violations", assert_isolated(clean) == [],
                    str(assert_isolated(clean)))

        print("\nTHE GUARD: a reference_id on a candidate row fails the check")
        leaky = root / "leaky"
        for stage in ISOLATED_STAGES:
            (leaky / stage).mkdir(parents=True, exist_ok=True)
        write_jsonl(leaky / "05_motion_roi/candidate_manifest.jsonl",
                    [{"event_id": "EV-1", "reference_id": "REF-001"}])
        violations = assert_isolated(leaky)
        ok &= check("the leak is detected", bool(violations))
        ok &= check("the file and line are named",
                    "candidate_manifest.jsonl:1" in violations[0], violations[0])

        print("\nand so does one on a crop row, or nested in a pose comparison")
        write_jsonl(leaky / "09_object/object_crop_manifest.jsonl",
                    [{"crop_id": "c1", "ground_truth_class": "phone"}])
        (leaky / "08_pose/comparison.json").write_text(
            json.dumps({"rows": [{"meta": {"reference_bbox": [1, 2, 3, 4]}}]}),
            encoding="utf-8")
        violations = assert_isolated(leaky)
        ok &= check("all three are reported", len(violations) >= 3,
                    f"{len(violations)} violations")
        ok &= check("every isolated stage is scanned",
                    len(ISOLATED_STAGES) >= 7, str(ISOLATED_STAGES))
        ok &= check("stage 12 is not in the isolated set",
                    "12_evaluation" not in ISOLATED_STAGES,
                    "stage 12 is the one place references may live")

        # -------------------------------------------------- manifest loading --
        print("\na valid manifest loads and is marked complete")
        good = write_jsonl(root / "good.jsonl", [GOOD_ROW])
        manifest = load(good, source_sha256="a" * 64,
                        frame_w=1280, frame_h=720)
        ok &= check("one row accepted", len(manifest.rows) == 1)
        ok &= check("no errors", not manifest.errors, str(manifest.errors))
        ok &= check("marked complete", manifest.complete)
        ok &= check("the box survives", manifest.rows[0].localised)
        ok &= check("the summary says metrics are complete",
                    "complete" in manifest.summary()["note"])

        print("\nevery PRD 14 rejection is recorded, not skipped")
        cases = [
            ({"source_video_sha256": "b" * 64}, "source_video_sha256",
             "a row describing a different file"),
            ({"source_pts_ms": -5.0}, "source_pts_ms", "a negative timestamp"),
            ({"source_pts_ms": "not a number"}, "source_pts_ms",
             "an unparseable timestamp"),
            ({"bbox_xyxy_source": [900.0, 100.0, 200.0, 300.0]},
             "bbox_xyxy_source", "a degenerate box"),
            ({"bbox_xyxy_source": [200.0, 500.0, 5000.0, 522.0]},
             "bbox_xyxy_source", "a box outside the frame"),
            ({"visibility": "sort of visible"}, "visibility",
             "an unknown visibility"),
            ({"annotation_source": ""}, "annotation_source",
             "no annotation provenance"),
        ]
        for override, field, label in cases:
            path = write_jsonl(root / "bad.jsonl", [{**GOOD_ROW, **override}])
            bad = load(path, source_sha256="a" * 64, frame_w=1280, frame_h=720)
            ok &= check(f"rejected: {label}",
                        len(bad.rows) == 0 and len(bad.errors) == 1,
                        f"{len(bad.rows)} accepted, {len(bad.errors)} rejected")
            ok &= check(f"  the failing field is named {field!r}",
                        bad.errors[0].field.startswith(field.split(",")[0]),
                        bad.errors[0].field)

        print("\na duplicate reference_id is rejected (PRD 14: unique, immutable)")
        path = write_jsonl(root / "dup.jsonl", [GOOD_ROW, GOOD_ROW])
        dup = load(path, source_sha256="a" * 64)
        ok &= check("one accepted, one rejected",
                    len(dup.rows) == 1 and len(dup.errors) == 1)

        print("\na mixed manifest makes the whole evaluation incomplete")
        path = write_jsonl(root / "mixed.jsonl", [
            GOOD_ROW,
            {**GOOD_ROW, "reference_id": "REF-002", "source_pts_ms": -1.0},
            {**GOOD_ROW, "reference_id": "REF-003", "source_pts_ms": 20000.0}])
        mixed = load(path, source_sha256="a" * 64)
        ok &= check("valid rows are still loaded", len(mixed.rows) == 2)
        ok &= check("the invalid one is recorded", len(mixed.errors) == 1)
        ok &= check("but the manifest is NOT complete", not mixed.complete,
                    "clean metrics over a silently reduced set is the quiet "
                    "version of the same failure")
        ok &= check("the summary says INCOMPLETE",
                    "INCOMPLETE" in mixed.summary()["note"],
                    mixed.summary()["note"][:60])

        print("\nan unapproved class is rejected when a taxonomy is supplied")
        path = write_jsonl(root / "cls.jsonl",
                           [{**GOOD_ROW, "class": "smartwatch"}])
        unapproved = load(path, source_sha256="a" * 64,
                          approved_classes=("phone", "secondary_paper_chit"))
        ok &= check("rejected", len(unapproved.rows) == 0)
        ok &= check("with the class named",
                    "smartwatch" in unapproved.errors[0].detail,
                    unapproved.errors[0].detail[:60])
        loose = load(path, source_sha256="a" * 64)
        ok &= check("and accepted when no taxonomy is enforced",
                    len(loose.rows) == 1,
                    "PRD 3 blocks selection on an unapproved taxonomy, not loading")

        print("\nCSV and JSONL load identically")
        csv_path = root / "refs.csv"
        csv_path.write_text(
            "reference_id,source_video_sha256,source_pts_ms,class,"
            "annotation_source,visibility\n"
            f"REF-001,{'a' * 64},12000.0,phone,reviewer-a,visible\n",
            encoding="utf-8")
        from_csv = load(csv_path, source_sha256="a" * 64)
        ok &= check("one row from CSV", len(from_csv.rows) == 1,
                    str(from_csv.errors))
        ok &= check("an unlocalised row is allowed",
                    not from_csv.rows[0].localised,
                    "PRD 14 requires a box only when localisation is known")
        ok &= check("every visibility value is enumerated",
                    len(VISIBILITY) == 6, str(VISIBILITY))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
