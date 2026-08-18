"""Phase 6 validation gate: crop geometry, abstention, pose masking.

None of this needs model weights. The parts that would silently corrupt
evidence -- a box that maps back to the wrong pixel, an object that qualifies
only because its crop was magnified, a wrist coordinate invented for a hand
behind a desk -- are all pure geometry, and that is what is checked here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from classroom import crops, detect, pose  # noqa: E402
from classroom.detect import Detection  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def main() -> int:
    ok = True
    frame = np.random.default_rng(3).integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    print("crop geometry: round-trip through the transform")
    seat = rect(400, 300, 620, 460)
    crop, tf = crops.seat_crop(frame, seat, input_size=640, context_px=0)
    ok &= check("crop is the requested input size", crop.shape[:2] == (640, 640),
                str(crop.shape[:2]))
    ok &= check("aspect ratio preserved via letterbox",
                abs((tf.crop_w / tf.crop_h) - (220 / 160)) < 1e-6)

    # A box drawn in frame coordinates must survive a trip through the crop.
    for box in [(430.0, 330.0, 470.0, 360.0), (400.0, 300.0, 620.0, 460.0),
                (500.5, 380.25, 512.75, 391.5)]:
        back = tf.to_frame(tf.to_input(box))
        err = max(abs(a - b) for a, b in zip(box, back))
        ok &= check(f"box round-trips {box[:2]}", err < 1e-6, f"max err {err:.2e}")

    print("\nsupersampling: the whole point of cropping this way")
    ok &= check("220x160 seat crop is magnified into 640", tf.supersampling > 2.5,
                f"{tf.supersampling:.2f}x")
    big, tf_big = crops.seat_crop(frame, rect(0, 0, 1280, 720), 640, context_px=0)
    ok &= check("full-frame crop is shrunk, not magnified", tf_big.supersampling < 1.0,
                f"{tf_big.supersampling:.2f}x")

    print("\narea measured in source pixels, never magnified pixels")
    # A 30x20 phone: 600 px^2 in the frame, but 3.2x magnified in the input.
    phone_frame = (500.0, 380.0, 530.0, 400.0)
    phone_input = tf.to_input(phone_frame)
    input_area = (phone_input[2] - phone_input[0]) * (phone_input[3] - phone_input[1])
    ok &= check("input area is inflated by magnification", input_area > 600 * 4,
                f"{input_area:.0f} px^2 in input")
    ok &= check("frame_area recovers the true footprint",
                abs(tf.frame_area(input_area) - 600.0) < 1.0,
                f"{tf.frame_area(input_area):.1f} px^2 in frame")

    print("\ntiny crops are widened rather than magnified without limit")
    _, tf_tiny = crops.seat_crop(frame, rect(600, 400, 620, 415), 640, min_crop_px=96)
    ok &= check("crop widened to the floor", tf_tiny.crop_w >= 96 and tf_tiny.crop_h >= 96,
                f"{tf_tiny.crop_w}x{tf_tiny.crop_h}")
    ok &= check("magnification therefore bounded", tf_tiny.supersampling <= 640 / 96 + 1e-6,
                f"{tf_tiny.supersampling:.2f}x")

    print("\ncrops clamp to the frame edge")
    _, tf_edge = crops.seat_crop(frame, rect(1200, 650, 1400, 800), 640, context_px=40)
    ok &= check("stays inside the frame",
                tf_edge.x0 >= 0 and tf_edge.y0 >= 0
                and tf_edge.x0 + tf_edge.crop_w <= 1280
                and tf_edge.y0 + tf_edge.crop_h <= 720,
                f"({tf_edge.x0},{tf_edge.y0}) {tf_edge.crop_w}x{tf_edge.crop_h}")
    try:
        crops.seat_crop(frame, rect(2000, 2000, 2100, 2100), 640)
        ok &= check("off-frame polygon rejected", False)
    except ValueError:
        ok &= check("off-frame polygon rejected", True)

    print("\nabstention at the measured phone footprint")
    dets = [
        Detection("phone_like", 0.91, (500, 380, 530, 400), 600.0, False),   # 30x20
        Detection("phone_like", 0.88, (500, 380, 515, 390), 150.0, False),   # 15x10
        Detection("paper_like", 0.75, (400, 300, 460, 350), 3000.0, False),
    ]
    after = detect.apply_abstention(dets, min_px_area=300.0)
    ok &= check("600 px^2 object survives", after[0].cls == "phone_like")
    ok &= check("150 px^2 object abstains", after[1].cls == detect.INSUFFICIENT
                and after[1].abstained)
    ok &= check("large paper survives", after[2].cls == "paper_like")
    ok &= check("abstained detection is not usable", not after[1].usable)

    print("\nabstention cannot be defeated by magnifying the crop")
    small_in_input = tf.to_input((500.0, 380.0, 515.0, 390.0))
    area_input = ((small_in_input[2] - small_in_input[0])
                  * (small_in_input[3] - small_in_input[1]))
    ok &= check("input area alone would pass the floor", area_input > 300.0,
                f"{area_input:.0f} px^2")
    ok &= check("source area correctly fails the floor",
                tf.frame_area(area_input) < 300.0,
                f"{tf.frame_area(area_input):.0f} px^2")

    print("\nobject evidence weighting")
    ok &= check("abstained objects contribute nothing",
                detect.object_evidence([after[1]]) == 0.0)
    ok &= check("phone outweighs paper",
                detect.object_evidence([after[0]]) > detect.object_evidence([after[2]]),
                f"{detect.object_evidence([after[0]]):.2f} vs "
                f"{detect.object_evidence([after[2]]):.2f}")
    ok &= check("no detections means no evidence", detect.object_evidence([]) == 0.0)

    print("\ntemporal confirmation across adjacent frames")
    steady = [[Detection("phone_like", 0.8, (500, 380, 530, 400), 600.0, False)]
              for _ in range(4)]
    ok &= check("object present in 4 frames confirmed",
                len(detect.temporal_confirm(steady, required=3)) == 1)
    flicker = [
        [Detection("phone_like", 0.8, (500, 380, 530, 400), 600.0, False)],
        [], [], [],
    ]
    ok &= check("single-frame flicker rejected",
                len(detect.temporal_confirm(flicker, required=3)) == 0)
    moved = [
        [Detection("phone_like", 0.8, (500, 380, 530, 400), 600.0, False)],
        [Detection("phone_like", 0.8, (900, 600, 930, 620), 600.0, False)],
        [Detection("phone_like", 0.8, (100, 100, 130, 120), 600.0, False)],
    ]
    ok &= check("detections in unrelated places not confirmed",
                len(detect.temporal_confirm(moved, required=3)) == 0)

    print("\nhead pitch from COCO-17")
    def person(nose_y, shoulder_y=300.0, conf=0.9):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[pose.NOSE] = (500.0, nose_y, conf)
        kp[pose.L_SHOULDER] = (470.0, shoulder_y, conf)
        kp[pose.R_SHOULDER] = (530.0, shoulder_y, conf)
        return kp

    upright = pose.head_pitch(person(nose_y=270.0))
    looking_down = pose.head_pitch(person(nose_y=330.0))
    ok &= check("upright head gives negative pitch", upright < 0, f"{upright:.3f}")
    ok &= check("head down gives positive pitch", looking_down > 0, f"{looking_down:.3f}")
    ok &= check("pitch is scale invariant",
                abs(pose.head_pitch(person(330.0)) -
                    pose.head_pitch(np.array([[c[0] * 2, c[1] * 2, c[2]]
                                              for c in person(330.0)],
                                             dtype=np.float32) *
                                    np.array([1, 1, 1], dtype=np.float32))) < 1e-6)

    hidden = person(330.0)
    hidden[pose.NOSE, 2] = 0.05
    ok &= check("unresolvable pitch returns None, not zero",
                pose.head_pitch(hidden) is None)

    print("\nwrist masking below the desk line")
    kp = person(310.0)
    kp[pose.L_WRIST] = (480.0, 520.0, 0.85)   # below a desk line at 480
    kp[pose.R_WRIST] = (520.0, 400.0, 0.85)   # above it
    obs = pose.observe(kp, desk_line_y=480.0)
    ok &= check("one wrist counted below the line", obs.wrists_below == 1)
    ok &= check("masking flagged", obs.wrist_masked)
    ok &= check("below-line wrist confidence zeroed",
                obs.keypoints[pose.L_WRIST, 2] == 0.0)
    ok &= check("above-line wrist untouched",
                obs.keypoints[pose.R_WRIST, 2] == 0.85)

    occluded = person(310.0)
    occluded[pose.L_WRIST] = (480.0, 520.0, 0.05)
    obs2 = pose.observe(occluded, desk_line_y=480.0)
    ok &= check("occluded wrist is not counted as below", obs2.wrists_below == 0)

    no_line = pose.observe(kp, desk_line_y=None)
    ok &= check("no desk line means no masking", not no_line.wrist_masked)
    ok &= check("no desk line preserves confidence",
                no_line.keypoints[pose.L_WRIST, 2] == 0.85)

    print("\naggregate pose evidence over an event")
    down = [pose.observe(person(340.0), 480.0) for _ in range(7)]
    up = [pose.observe(person(265.0), 480.0) for _ in range(3)]
    ok &= check("sustained looking-down scores high",
                pose.pitch_evidence(down + up) > 0.6,
                f"{pose.pitch_evidence(down + up):.2f}")
    ok &= check("upright throughout scores zero", pose.pitch_evidence(up) == 0.0)
    unresolved = [pose.observe(hidden, 480.0) for _ in range(5)]
    ok &= check("all-unresolvable scores zero, not spurious",
                pose.pitch_evidence(unresolved) == 0.0)

    lap = [pose.observe(kp, 480.0) for _ in range(8)]
    ok &= check("below-desk evidence aggregates",
                pose.below_desk_evidence(lap) == 1.0)
    ok &= check("no observations means no evidence",
                pose.below_desk_evidence([]) == 0.0)

    print("\nmissing weights fail loudly, not silently")
    det = detect.ONNXDetector("models/does-not-exist.onnx", list(detect.CLASSES))
    try:
        det.load()
        ok &= check("missing detector weights raise", False)
    except FileNotFoundError:
        ok &= check("missing detector weights raise", True)

    print("\ntwo separate model passes do not erase each other")
    # The normal shape when only one model is resident at a time: detector
    # first, pose second. Clearing both tables on every persist meant the
    # pose-only pass silently deleted the detections, and the run still
    # reported success.
    import tempfile

    from classroom import db, evidence
    from classroom.evidence import EventEvidence

    with tempfile.TemporaryDirectory() as tmp:
        conn = db.connect(Path(tmp) / "e.db")
        conn.execute("INSERT INTO session (id, name) VALUES (1, 't')")
        conn.execute("INSERT INTO camera (id, session_id, label) VALUES (1, 1, 'c')")
        conn.execute(
            "INSERT INTO source_video (id, camera_id, path, sha256, size_bytes, "
            "container, codec, pix_fmt, width, height, avg_fps, duration_s, "
            "frame_count, is_vfr, has_mv) VALUES (1, 1, 'x', 'a', 1, 'mkv', "
            "'h264', 'yuv420p', 1280, 720, 25.0, 10.0, 250, 0, 1)"
        )
        conn.execute("INSERT INTO calibration (id, camera_id, version, frame_w, "
                     "frame_h) VALUES (1, 1, 1, 1280, 720)")
        conn.execute("INSERT INTO seat (id, calibration_id, seat_label, polygon) "
                     "VALUES (1, 1, 'A', '[]')")
        conn.execute(
            "INSERT INTO event (id, source_video_id, seat_id, t_start, t_end, "
            "peak_t, peak_deviation) VALUES (1, 1, 1, 0.0, 2.0, 1.0, 5.0)")

        detection = Detection("phone_like", 0.8, (10, 10, 40, 30), 900.0, False, 1.0)
        evidence.persist(conn, EventEvidence(
            1, "A", [detection], [], object_evidence=0.7,
            pitch_evidence=None, below_desk_evidence=None))
        n_det = conn.execute("SELECT COUNT(*) FROM detection").fetchone()[0]
        ok &= check("detector pass writes detections", n_det == 1, str(n_det))

        observation = pose.observe(np.zeros((17, 3), dtype=np.float32), None)
        evidence.persist(conn, EventEvidence(
            1, "A", [], [observation], object_evidence=None,
            pitch_evidence=0.3, below_desk_evidence=0.0))
        n_det = conn.execute("SELECT COUNT(*) FROM detection").fetchone()[0]
        n_pose = conn.execute("SELECT COUNT(*) FROM pose_observation").fetchone()[0]
        ok &= check("pose pass preserves the detections", n_det == 1, str(n_det))
        ok &= check("pose pass writes its own rows", n_pose == 1, str(n_pose))

        evidence.persist(conn, EventEvidence(
            1, "A", [detection, detection], [], object_evidence=0.7,
            pitch_evidence=None, below_desk_evidence=None))
        n_det = conn.execute("SELECT COUNT(*) FROM detection").fetchone()[0]
        n_pose = conn.execute("SELECT COUNT(*) FROM pose_observation").fetchone()[0]
        ok &= check("re-running the detector replaces its own rows", n_det == 2,
                    str(n_det))
        ok &= check("and leaves the pose rows alone", n_pose == 1, str(n_pose))
        conn.close()

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
