"""Ground-truth store and evaluation gate.

The metrics are driven by hand-built timelines with known answers rather than by
pipeline output, because the point is to catch an evaluator that flatters the
system. Three failure modes are specifically exercised: crediting one long event
with several separate actions, counting benign motion as an error, and computing
rates over footage nobody reviewed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import annotate, db, evaluate  # noqa: E402
from classroom.annotate import Annotation, Span  # noqa: E402
from classroom.evaluate import SystemEvent, match_events  # noqa: E402

VIDEO = 1


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def ann(t0: float, t1: float, kind: str = "object_handling",
        vis: str = "clear", seat: str | None = "19") -> Annotation:
    return Annotation(VIDEO, t0, t1, kind, vis, seat, "test")


def ev(t0: float, t1: float, score: float = 1.0,
       seat: str | None = "19", event_id: int = 0) -> SystemEvent:
    return SystemEvent(event_id, VIDEO, t0, t1, score, seat)


def main() -> int:
    ok = True
    spans = [Span(VIDEO, 0.0, 600.0, "test")]

    print("coverage arithmetic")
    overlapping = [Span(VIDEO, 0, 100, "a"), Span(VIDEO, 50, 150, "b"),
                   Span(VIDEO, 300, 400, "a")]
    ok &= check("overlapping review passes are not double counted",
                annotate.coverage_s(overlapping) == 250.0,
                f"{annotate.coverage_s(overlapping)}")
    merged = annotate.merge_spans(overlapping)
    ok &= check("merged into disjoint intervals",
                merged == [(0, 150), (300, 400)], f"{merged}")
    ok &= check("an event outside every span is excluded",
                not annotate.within_coverage(200, 210, merged))
    ok &= check("an event inside a span is included",
                annotate.within_coverage(60, 70, merged))

    print("\nexact overlap")
    result = match_events([ev(10, 20)], [ann(10, 20)], spans)
    ok &= check("one match", len(result.matches) == 1)
    ok &= check("recall 1.0", result.recall == 1.0)
    ok &= check("precision 1.0", result.precision == 1.0)
    ok &= check("tIoU 1.0", abs(result.matches[0].tiou - 1.0) < 1e-9)

    print("\none event spanning three actions is credited with one")
    three = [ann(10, 15), ann(20, 25), ann(40, 45)]
    result = match_events([ev(10, 45)], three, spans)
    ok &= check("exactly one match", len(result.matches) == 1,
                f"{len(result.matches)}")
    ok &= check("the other two are misses", len(result.missed) == 2,
                f"{len(result.missed)}")
    ok &= check("recall is 1/3", abs(result.recall - 1 / 3) < 1e-9,
                f"{result.recall:.3f}")

    print("\nan event vastly longer than the action does not localise it")
    # 100 s flagged around a 5 s action is a tIoU of 0.05. Counting that as a
    # hit would let a system that flags everything claim perfect recall.
    result = match_events([ev(0, 100)], [ann(10, 15)], spans)
    ok &= check("no match at 20x the action length", len(result.matches) == 0,
                f"{len(result.matches)}")
    ok &= check("it is a miss, not a silent pass", len(result.missed) == 1)

    print("\nthree events on one action are credited with one")
    result = match_events([ev(10, 12, event_id=1), ev(12, 14, event_id=2),
                           ev(14, 16, event_id=3)], [ann(10, 16)], spans)
    ok &= check("one match", len(result.matches) == 1, f"{len(result.matches)}")
    ok &= check("the surplus events are false positives",
                len(result.false_events) == 2, f"{len(result.false_events)}")

    print("\nbenign motion is dismissible, not a false positive")
    result = match_events([ev(30, 40)], [ann(30, 40, kind="benign_movement")], spans)
    ok &= check("not counted as a false event", len(result.false_events) == 0)
    ok &= check("counted as dismissible", len(result.dismissible) == 1)
    ok &= check("no scored annotation to recall",
                result.scored_annotations == 0)
    ok &= check("precision is undefined rather than zero",
                result.precision is None, f"{result.precision}")

    print("\nstaff interaction is dismissible too")
    result = match_events([ev(30, 40)], [ann(30, 40, kind="staff_interaction")], spans)
    ok &= check("not a false event", len(result.false_events) == 0)

    print("\ninvisible actions are excluded from recall")
    result = match_events([], [ann(10, 20, vis="insufficient")], spans)
    ok &= check("not counted against recall", result.scored_annotations == 0,
                f"{result.scored_annotations}")
    ok &= check("recall undefined, not zero", result.recall is None)
    # An action of real interest that the footage cannot show is outside the
    # recall denominator, but an event firing on it has found something genuine.
    # Calling that a false positive punishes the intended behaviour.
    result = match_events([ev(10, 20)], [ann(10, 20, vis="insufficient")], spans)
    ok &= check("firing on one is not a false positive",
                len(result.false_events) == 0, f"{len(result.false_events)}")
    ok &= check("it is recorded as dismissible", len(result.dismissible) == 1)

    print("\nseat mismatch prevents a match")
    result = match_events([ev(10, 20, seat="19")], [ann(10, 20, seat="24")], spans)
    ok &= check("different seats do not match", len(result.matches) == 0)
    result = match_events([ev(10, 20, seat=None)], [ann(10, 20, seat="24")], spans)
    ok &= check("an unknown seat still matches", len(result.matches) == 1)

    print("\nthresholds behave monotonically")
    partial = [ev(10, 20)]
    labelled = [ann(18, 30)]
    loose = match_events(partial, labelled, spans, 0.1)
    strict = match_events(partial, labelled, spans, 0.5)
    ok &= check("matches at a loose threshold", len(loose.matches) == 1)
    ok &= check("does not match at a strict one", len(strict.matches) == 0)

    print("\nevents outside reviewed footage are ignored entirely")
    narrow = [Span(VIDEO, 0.0, 50.0, "test")]
    result = match_events([ev(10, 20), ev(100, 110)], [ann(10, 20)], narrow)
    ok &= check("only the covered event is considered",
                result.events_considered == 1, f"{result.events_considered}")
    ok &= check("the uncovered one is not a false positive",
                len(result.false_events) == 0)
    ok &= check("coverage drives the rate clock",
                abs(result.coverage_s - 50.0) < 1e-9)

    print("\nfalse events per hour uses reviewed time, not wall clock")
    result = match_events([ev(5, 6, event_id=1), ev(20, 21, event_id=2)],
                          [], [Span(VIDEO, 0.0, 1800.0, "test")])
    ok &= check("2 events in 30 min = 4.0/hour",
                abs(result.false_events_per_hour - 4.0) < 1e-9,
                f"{result.false_events_per_hour}")

    print("\nreview burden is the flagged share of reviewed footage")
    result = match_events([ev(0, 60)], [], [Span(VIDEO, 0.0, 600.0, "test")])
    ok &= check("60 s flagged out of 600 s = 10%",
                abs(result.review_burden - 0.1) < 1e-9, f"{result.review_burden}")

    print("\nprecision@k ranks by score")
    events = [ev(0, 5, score=9.0, event_id=1), ev(100, 105, score=8.0, event_id=2),
              ev(200, 205, score=0.1, event_id=3)]
    labels = [ann(0, 5), ann(100, 105)]
    ok &= check("top 2 are both real",
                evaluate.precision_at_k(events, labels, spans, 2) == 1.0)
    ok &= check("top 3 dilutes to 2/3",
                abs(evaluate.precision_at_k(events, labels, spans, 3) - 2 / 3) < 1e-9)

    print("\nbreakdowns")
    result = match_events(
        [ev(10, 20, event_id=1), ev(50, 60, event_id=2)],
        [ann(10, 20, vis="clear"), ann(50, 60, vis="occluded"),
         ann(90, 100, vis="occluded")],
        spans,
    )
    vis = evaluate.by_visibility(result)
    ok &= check("clear actions fully recalled", vis.get("clear") == (1, 1), f"{vis}")
    ok &= check("occluded actions partly recalled",
                vis.get("occluded") == (1, 2), f"{vis}")

    print("\nvalidation refuses malformed labels")
    for bad, why in (
        (lambda: Annotation(VIDEO, 20, 10, "object_handling", "clear").validate(),
         "reversed interval"),
        (lambda: Annotation(VIDEO, 0, 10, "nonsense", "clear").validate(),
         "unknown event type"),
        (lambda: Annotation(VIDEO, 0, 10, "object_handling", "maybe").validate(),
         "unknown visibility"),
        (lambda: annotate.Box("f.jpg", "phone", 0, 0, 10, 10).validate(),
         "unknown box class"),
        (lambda: annotate.Box("f.jpg", "phone_like", 10, 10, 10, 10).validate(),
         "degenerate box"),
        (lambda: Span(VIDEO, 0, 10, "").validate(), "span with no annotator"),
    ):
        raised = False
        try:
            bad()
        except ValueError:
            raised = True
        ok &= check(f"rejects {why}", raised)

    print("\nround trip through the database")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gt.db"
        conn = db.connect(path)
        conn.execute("INSERT INTO session (id, name) VALUES (1, 'test')")
        conn.execute("INSERT INTO camera (id, session_id, label) "
                     "VALUES (1, 1, 'cam')")
        conn.execute(
            "INSERT INTO source_video (id, camera_id, path, sha256, size_bytes, "
            "container, codec, pix_fmt, width, height, avg_fps, duration_s, "
            "frame_count, is_vfr, has_mv) VALUES "
            "(1, 1, 'x.mkv', 'abc', 1, 'mkv', 'h264', 'yuv420p', 1280, 720, "
            "25.0, 600.0, 15000, 0, 1)"
        )
        annotate.save_span(conn, Span(VIDEO, 0.0, 600.0, "test"))
        annotate.save_annotation(conn, ann(10, 20))
        annotate.save_box(conn, annotate.Box("a.jpg", "mouse", 0, 0, 30, 20,
                                             source_video_id=VIDEO))
        annotate.mark_frame_reviewed(conn, "a.jpg", VIDEO)
        annotate.mark_frame_reviewed(conn, "b.jpg", VIDEO)
        annotate.mark_frame_reviewed(conn, "b.jpg", VIDEO)   # re-review is normal

        ok &= check("span reloads", len(annotate.load_spans(conn)) == 1)
        ok &= check("annotation reloads", len(annotate.load_annotations(conn)) == 1)
        boxes = annotate.load_boxes(conn)
        ok &= check("box reloads", len(boxes) == 1)
        ok &= check("negative flag derived from class", boxes[0].negative)
        ok &= check("re-reviewing a frame does not duplicate it",
                    len(annotate.reviewed_frames(conn)) == 2,
                    f"{annotate.reviewed_frames(conn)}")

        annotate.clear_boxes(conn, "a.jpg")
        ok &= check("clearing a frame removes its boxes",
                    len(annotate.load_boxes(conn)) == 0)

        print("\nCOCO export")
        boxes = [annotate.Box("a.jpg", "phone_like", 10, 10, 40, 30),
                 annotate.Box("a.jpg", "mouse", 100, 100, 140, 130)]
        coco = annotate.to_coco(boxes, ["a.jpg", "b.jpg"],
                                {"a.jpg": (1280, 720), "b.jpg": (1280, 720)})
        ok &= check("empty reviewed frames become background images",
                    len(coco["images"]) == 2, f"{len(coco['images'])}")
        ok &= check("annotations exported", len(coco["annotations"]) == 2)
        ok &= check("bbox is xywh",
                    coco["annotations"][0]["bbox"] == [10, 10, 30, 20],
                    f"{coco['annotations'][0]['bbox']}")
        ok &= check("category ids are 1-based",
                    min(c["id"] for c in coco["categories"]) == 1)
        missing = False
        try:
            annotate.to_coco(boxes, [], {})
        except ValueError:
            missing = True
        ok &= check("refuses to export an image of unknown size", missing)

        print("\nthe report refuses to invent numbers")
        empty = db.connect(Path(tmp) / "empty.db")
        text = evaluate.report(empty)
        ok &= check("says plainly that nothing is measurable",
                    "nothing here can" in text and "PRD 15.4" in text)
        ok &= check("no percentage is printed", "%" not in text)

        text = evaluate.report(conn)
        ok &= check("with labels present, it reports", "recall" in text)

        # Windows will not delete an open SQLite file, and the WAL keeps a
        # handle alive until the connection closes.
        conn.close()
        empty.close()

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
