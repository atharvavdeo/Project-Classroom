"""Head/neck orientation: geometry, abstention, and event formation.

These are the checks that would have caught the two defects the 1 Hz run
exposed -- a merge gap shorter than the sample period, so no event could ever
form, and a summary keyed on `seat_state` when every seat is "unattributed".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import head_pose as hp  # noqa: E402


def joint(x, y, confidence=0.9, observability="visible"):
    return {"x": x, "y": y, "confidence": confidence,
            "observability": observability}


def head(nose_x, *, ear_half=10.0, centre_x=100.0, y=50.0, eye_dy=0.0,
         shoulders=True):
    """A synthetic head. Ears at +/- ear_half around centre, nose at nose_x."""
    out = {
        "nose": joint(nose_x, y),
        "left_ear": joint(centre_x + ear_half, y),
        "right_ear": joint(centre_x - ear_half, y),
        "left_eye": joint(centre_x + 4, y + eye_dy),
        "right_eye": joint(centre_x - 4, y + eye_dy),
    }
    if shoulders:
        out["left_shoulder"] = joint(centre_x + 30, y + 40)
        out["right_shoulder"] = joint(centre_x - 30, y + 40)
    return out


def test_frontal_head_has_near_zero_yaw():
    pose = hp.estimate(head(nose_x=100.0))
    assert pose.yaw is not None
    assert abs(pose.yaw) < 1e-6
    assert pose.facing == hp.FRONTAL


def test_yaw_sign_follows_the_direction_of_turn():
    """+ve yaw is toward the subject's own left ear, by the module's convention."""
    left = hp.estimate(head(nose_x=106.0))     # nose toward the left ear
    right = hp.estimate(head(nose_x=94.0))     # nose toward the right ear
    assert left.yaw > 0 and right.yaw < 0
    assert left.facing == hp.TURNED_LEFT
    assert right.facing == hp.TURNED_RIGHT
    # Symmetric geometry must give symmetric magnitude.
    assert abs(abs(left.yaw) - abs(right.yaw)) < 1e-6


def test_yaw_is_scale_free():
    """The same turn at two camera distances reads the same."""
    near = hp.estimate(head(nose_x=100.0 + 0.6 * 20.0, ear_half=20.0))
    far = hp.estimate(head(nose_x=100.0 + 0.6 * 8.0, ear_half=8.0))
    assert abs(near.yaw - far.yaw) < 1e-6, (near.yaw, far.yaw)


def test_one_visible_ear_saturates_in_that_direction():
    """The occluded ear IS the measurement, not a missing input."""
    joints = head(nose_x=100.0)
    joints["right_ear"]["observability"] = "occluded"
    pose = hp.estimate(joints)
    assert pose.yaw == 1.0
    assert pose.facing == hp.TURNED_LEFT
    assert "occluded by the turn" in pose.evidence


def test_head_below_the_scale_floor_abstains():
    cfg = hp.HeadPoseConfig()
    pose = hp.estimate(head(nose_x=100.0, ear_half=1.0), cfg=cfg)
    assert pose.facing == hp.NOT_RESOLVABLE
    assert pose.yaw is None
    assert "under the" in pose.evidence


def test_low_confidence_keypoints_are_not_used():
    joints = head(nose_x=106.0)
    for name in ("left_ear", "right_ear"):
        joints[name]["confidence"] = 0.05
    pose = hp.estimate(joints)
    # Ears rejected, so head scale falls back to interocular; yaw needs ears.
    assert "left_ear" not in pose.keypoints_used
    assert "right_ear" not in pose.keypoints_used


def test_excursions_form_at_the_actual_sample_rate():
    """The regression: a fixed 800 ms merge gap under 1 Hz sampling split every
    run into singletons, so no event could reach `min_event_ms`."""
    poses = []
    for index in range(20):
        pts = index * 1000.0                      # 1 Hz, as the pose stage samples
        yaw = 0.9 if 5 <= index <= 12 else 0.0    # a sustained 8-second turn
        poses.append(hp.HeadPose(pose_row_id=f"r{index}", config_id="t",
                                 pts_ms=pts, seat_state="unattributed",
                                 track_id=7, yaw=yaw, pitch=0.0,
                                 facing=hp.FRONTAL))
    events = hp.excursions(poses)
    turns = [e for e in events if e.kind == "yaw_left"]
    assert turns, "a sustained 8 s turn at 1 Hz must produce an event"
    assert turns[0].duration_ms >= 7000.0
    assert turns[0].track_id == 7


def test_brief_glance_is_not_an_event():
    poses = []
    for index in range(20):
        yaw = 0.9 if index == 8 else 0.0          # one sample only
        poses.append(hp.HeadPose(pose_row_id=f"r{index}", config_id="t",
                                 pts_ms=index * 1000.0, seat_state="s",
                                 track_id=1, yaw=yaw, pitch=0.0))
    assert not [e for e in hp.excursions(poses) if e.kind.startswith("yaw")]


def test_tracks_are_not_pooled_into_one_baseline():
    """Every seat is 'unattributed' under a draft calibration, so keying the
    summary on seat alone pools every person in the hall."""
    poses = []
    for track, yaw in ((1, 0.0), (2, 0.8)):
        for index in range(10):
            poses.append(hp.HeadPose(
                pose_row_id=f"t{track}r{index}", config_id="t",
                pts_ms=index * 1000.0, seat_state="unattributed",
                track_id=track, yaw=yaw, pitch=0.0, facing=hp.FRONTAL))
    report = hp.summarise(poses, hp.excursions(poses))
    assert report["seats"] == 2, report["per_seat"].keys()
    assert all("track_" in key for key in report["per_seat"])


def test_summary_states_what_is_not_measurable():
    report = hp.summarise([], [], {})
    assert "gaze" in report["not_measurable"].lower()


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
