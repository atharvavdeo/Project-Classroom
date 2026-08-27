"""Parsing and live checks for the `general-segmentation-api-6` workflow.

The offline tests run always and cover the traps that a live call would hide:
the leading space on `class`, the centre-to-corner box conversion, the mask
being dropped, and a 200 whose body does not match the workflow's declared
outputs.

The live test runs only with `ROBOFLOW_API_KEY` set, and asserts against the
workflow's **own output names**, so it fails loudly if the workflow is edited
to rename an output rather than silently returning nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import roboflow_workflow as rw  # noqa: E402


def response(predictions, width=1152, height=864):
    """A response with the shape a real call returned."""
    return {"outputs": [{
        rw.OUTPUT_ANNOTATED: {"type": "base64", "value": "",
                              "video_metadata": None},
        rw.OUTPUT_PREDICTIONS: {"image": {"width": width, "height": height},
                                "predictions": predictions}}],
        "profiler_trace": []}


def prediction(cls=" keyboard", conf=0.766, x=573.0, y=356.5, w=60.0, h=33.0):
    """One prediction as the API actually returns it, leading space included."""
    return {"width": w, "height": h, "x": x, "y": y, "confidence": conf,
            "class_id": 2, "class": cls, "detection_id": "abc",
            "parent_id": "image",
            "rle_mask": {"size": [864, 1152], "counts": "X`Z>3kj04M" * 400}}


def test_class_name_whitespace_is_stripped():
    """The API splits the prompt on commas without trimming, so every class
    after the first arrives with a leading space. Matching on the raw value
    silently finds nothing."""
    out = rw._parse(response([prediction(cls=" keyboard")]), False)
    assert out.segments[0].cls == "keyboard"
    assert out.of_class("keyboard"), "lookup must survive the API's whitespace"
    assert out.of_class(" KEYBOARD "), "lookup is trimmed and case-insensitive"


def test_boxes_convert_from_centre_to_corners():
    out = rw._parse(response([prediction(x=100.0, y=200.0, w=60.0, h=40.0)]),
                    False)
    assert out.segments[0].box == (70.0, 180.0, 130.0, 220.0)
    assert out.segments[0].area == 60.0 * 40.0


def test_mask_is_not_carried():
    """`rle_mask` is several KB and nothing reads it. Carrying it per frame
    across a video is how a run exhausts memory."""
    out = rw._parse(response([prediction()]), False)
    segment = out.segments[0]
    assert not hasattr(segment, "rle_mask")
    assert not hasattr(segment, "points")
    assert "counts" not in repr(segment)


def test_image_size_is_read_from_the_response():
    """Coordinates are in the space the workflow saw, not the space we sent."""
    out = rw._parse(response([], width=640, height=480), False)
    assert out.image_size == (640, 480)


def test_annotated_image_is_only_decoded_when_asked():
    import base64

    payload = response([])
    payload["outputs"][0][rw.OUTPUT_ANNOTATED]["value"] = \
        base64.b64encode(b"not-really-a-png").decode()
    assert rw._parse(payload, False).annotated_png is None
    assert rw._parse(payload, True).annotated_png == b"not-really-a-png"


def test_a_200_with_the_wrong_shape_raises():
    """The failure that matters: a 200 whose body does not match the declared
    outputs must be an error, never an empty result that reads as 'found
    nothing'."""
    for bad in ({}, {"outputs": []}, {"outputs": [[]]},
                {"outputs": [{"something_else": 1}]}):
        try:
            rw._parse(bad, False)
        except rw.WorkflowResponseError:
            continue
        raise AssertionError(f"{bad} should have raised")


def test_malformed_predictions_are_skipped_not_fatal():
    out = rw._parse(response([prediction(), {"class": "x"}, "junk"]), False)
    assert len(out.segments) == 1


def test_missing_key_raises_an_auth_error():
    saved = os.environ.pop("ROBOFLOW_API_KEY", None)
    try:
        rw.api_key()
    except rw.WorkflowAuthError as error:
        assert "app.roboflow.com" in str(error)
    else:
        raise AssertionError("a missing key must raise")
    finally:
        if saved is not None:
            os.environ["ROBOFLOW_API_KEY"] = saved


def test_live_call_returns_the_workflows_declared_outputs():
    """Skipped without a key. Asserts the workflow's own output names are
    present and that a text prompt reaches SAM 3."""
    if not os.environ.get("ROBOFLOW_API_KEY"):
        print("      SKIP  no ROBOFLOW_API_KEY set")
        return
    sample = ROOT.parent / "Chits_Frames" / "3.png"
    if not sample.is_file():
        print(f"      SKIP  no sample image at {sample}")
        return

    client = rw.SegmentationWorkflow()
    out = client.segment(sample, "mobile phone,paper chit",
                         keep_annotated=True)

    assert out.image_size[0] > 0 and out.image_size[1] > 0
    assert out.annotated_png and out.annotated_png[:3] in (b"\xff\xd8\xff",
                                                           b"\x89PN")
    # Frame 3 is a man plainly holding a phone -- the frame where the
    # single-class chit model answered `chit-paper 0.52`.
    phones = out.of_class("mobile phone")
    assert phones, f"expected a phone; got {[s.cls for s in out.segments]}"
    assert phones[0].confidence > 0.5
    print(f"      live: {len(out.segments)} segment(s), "
          f"phone at {phones[0].confidence:.2f}")


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:                                    # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
