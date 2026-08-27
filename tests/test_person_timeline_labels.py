"""Regression tests for the additive dashboard orientation outputs."""
from __future__ import annotations

from pipeline import person_timeline as pt


def _sample(index: int, *, pitch: float = 0.0,
            torso_yaw: float | None = 0.0) -> pt.PersonSample:
    return pt.PersonSample(
        track_id=7, pts_ms=index * 1000.0, yaw=1.5,
        pitch=pitch, yaw_relative_to_torso=torso_yaw,
    )


def _profile(rows: list[pt.PersonSample]) -> pt.PersonProfile:
    return pt.profile(pt.PersonTimeline(track_id=7, samples=rows),
                      duration_ms=max((r.pts_ms for r in rows), default=0.0) + 1000,
                      interval_ms=1000.0)


def _labels(profile: pt.PersonProfile) -> set[str]:
    return {row["label"] for row in profile.orientation_labels}


def test_pitch_bands_are_explicit_and_non_escalating():
    assert pt.pitch_state(-0.21) == "looking_up"
    assert pt.pitch_state(0.20) == "shallow_down"
    assert pt.pitch_state(0.45) == "shallow_down"
    assert pt.pitch_state(0.46) == "deep_down"

    profile = _profile([_sample(i, pitch=-0.3) for i in range(10)] +
                       [_sample(i + 10, pitch=0.3) for i in range(10)])
    labels = {row["label"]: row for row in profile.orientation_labels}
    assert {"looking_up", "shallow_down"} <= set(labels)
    assert labels["looking_up"]["dashboard_action"] == "display_only"
    assert labels["shallow_down"]["dashboard_action"] == "display_only"
    assert not profile.orientation_events


def test_deep_down_requires_fifteen_continuous_seconds():
    short = _profile([_sample(i, pitch=0.5) for i in range(14)])
    assert "deep_down_sustained" not in _labels(short)

    sustained = _profile([_sample(i, pitch=0.5) for i in range(15)])
    assert "deep_down_sustained" in _labels(sustained)
    event = next(e for e in sustained.orientation_events
                 if e["label"] == "deep_down_sustained")
    assert event["duration_ms"] == 15000.0
    assert event["dashboard_action"] == "show_review_context"


def test_orientation_labels_are_additive_and_change_nothing_else():
    """The dashboard layer must not move the established review queue.

    Every field the old profile produced has to be identical whether or not
    the orientation labels fire, and the labels may never write a flag.
    """
    plain = _profile([_sample(i, pitch=0.0, torso_yaw=None) for i in range(20)])
    labelled = _profile([_sample(i, pitch=0.5, torso_yaw=None)
                         for i in range(20)])

    assert "deep_down" == pt.pitch_state(0.5)
    assert labelled.orientation_labels          # the layer did fire
    assert labelled.flags == plain.flags        # ...and wrote no flag
    for field in ("samples", "first_seen_ms", "last_seen_ms",
                  "observed_span_ms", "coverage_of_recording",
                  "samples_turned_left", "samples_turned_right",
                  "absences", "seat_state"):
        assert getattr(labelled, field) == getattr(plain, field), field

    # No orientation label may name itself a flag or a machine state.
    for row in labelled.orientation_labels:
        assert row["classification"] in ("observed_only", "review_context",
                                         "context_requires_second_modality")


def test_held_turn_uses_torso_relative_yaw_not_saturated_raw_yaw():
    # Raw yaw is pinned at 1.5 in every frame. Only the torso-relative signal
    # varies, mirroring the saturation seen in the CCTV corpus.
    rows = [_sample(i, torso_yaw=0.05) for i in range(10)]
    rows += [_sample(i, torso_yaw=0.55) for i in range(10, 14)]
    profile = _profile(rows)

    assert profile.baseline_yaw == 1.5
    assert profile.baseline_yaw_relative_to_torso == 0.05
    assert "turned_and_held" in _labels(profile)
    event = next(e for e in profile.orientation_events
                 if e["label"] == "turned_and_held")
    assert event["duration_ms"] == 4000.0
    assert event["dashboard_action"] == "show_only_with_second_modality"
