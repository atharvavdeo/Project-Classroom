"""Tests for the output contract: reason codes, evidence records, fusion.

These are invariant tests, not accuracy tests. They check that the system
cannot say things it is not entitled to say. Nothing here measures whether a
detection was correct -- that requires the labelled evaluation pack described
in docs/OUTPUT_CONTRACT.md, and until it exists no test in this repository is
allowed to imply otherwise.
"""
from __future__ import annotations

import json

import pytest

from pipeline import evidence_record as er
from pipeline import fusion
from pipeline import reason_codes as rc
from pipeline import verdict


# --- the vocabulary -----------------------------------------------------------

def test_every_code_has_a_guardrail():
    """A reason with no guardrail is a reason that can be misread freely."""
    for reason in rc.ALL:
        assert reason.guardrail.strip(), f"{reason.code} has no guardrail"
        assert reason.meaning.strip(), f"{reason.code} has no meaning"


def test_routes_are_known_states():
    for reason in rc.ALL:
        assert reason.route in fusion.MACHINE_STATES


def test_no_code_routes_straight_to_a_human_state():
    """Nothing in the vocabulary may argue for a verdict only a person gives."""
    for reason in rc.ALL:
        assert reason.route not in fusion.HUMAN_STATES


def test_invented_codes_are_refused():
    with pytest.raises(KeyError, match="not in the reason vocabulary"):
        rc.get("OBVIOUSLY_CHEATING")


def test_sam3_not_confirmed_guardrail_says_not_false():
    """The single most dangerous misreading in the system, pinned by a test."""
    guardrail = rc.SAM3_NOT_CONFIRMED.guardrail.lower()
    assert "not false" in guardrail
    assert rc.SAM3_NOT_CONFIRMED.route == rc.R_BETTER_VIEW


def test_dfine_object_context_can_never_route_to_review():
    assert rc.DFINE_OBJECT_CONTEXT.route == rc.R_CONTEXT


def test_declared_only_codes_are_declared_not_hidden():
    """Unimplemented codes must exist and be marked, so no panel silently
    reads zero for a behaviour the pipeline cannot detect."""
    assert rc.DECLARED_ONLY, "the contract claims full coverage of nothing"
    for reason in rc.DECLARED_ONLY:
        assert reason.implemented is False


# --- the record ---------------------------------------------------------------

def _record(**kw):
    # `association` is a default, not a fixture: the phone route is decided on
    # the individual frame's wrist distance, so those tests must be able to
    # supply their own. setdefault keeps every existing call site unchanged.
    kw.setdefault("association", er.Association(track_id=7))
    record = er.EvidenceRecord(
        provenance=er.Provenance(run_id="t", video="v.mkv", pts_ms=250.0),
        detector=er.Detector(name="chit-paper-new", version="2",
                             class_name="paper_like_object", confidence=0.61),
        geometry=er.Geometry(box=(10.0, 10.0, 30.0, 30.0)),
        **kw)
    return record.assign_id()


def test_detector_output_is_immutable():
    """No later stage may rewrite what the detector originally said."""
    record = _record()
    with pytest.raises(Exception):
        record.detector.class_name = "mobile phone"
    with pytest.raises(Exception):
        record.geometry.box = (0.0, 0.0, 1.0, 1.0)


def test_reasons_accumulate_and_do_not_overwrite():
    record = _record()
    record.add_reason(rc.OBJECT_PROPOSAL_PAPER.code)
    record.add_reason(rc.SAM3_EQUIPMENT_CONTEXT.code)
    record.add_reason(rc.OBJECT_PROPOSAL_PAPER.code)     # repeat is a no-op
    assert record.reason_codes == [rc.OBJECT_PROPOSAL_PAPER.code,
                                   rc.SAM3_EQUIPMENT_CONTEXT.code]


def test_record_refuses_an_invented_reason():
    with pytest.raises(KeyError):
        _record().add_reason("DEFINITELY_A_CHIT")


def test_ids_are_stable_and_content_addressed():
    assert _record().record_id == _record().record_id


def test_roundtrip_preserves_the_argument(tmp_path):
    record = _record()
    record.add_reason(rc.OBJECT_PROPOSAL_PAPER.code)
    record.add_reason(rc.SAM3_NOT_CONFIRMED.code)
    record.sam3 = er.Sam3Response(attempted=True, responded=True,
                                  prompt="paper chit,keyboard")
    path = tmp_path / "r.jsonl"
    er.write_jsonl([record], path)
    back = er.read_jsonl(path)[0]
    assert back.reason_codes == record.reason_codes
    assert back.detector.confidence == 0.61
    assert back.sam3.attempted and back.sam3.responded


def test_unknown_contract_version_is_refused(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"record_id": "x", "contract_version": "0.1.0"})
                    + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to guess"):
        er.read_jsonl(path)


# --- per-record classification -------------------------------------------------

def test_sam3_outage_never_becomes_a_negative():
    """The failure mode the whole `responded` field exists to prevent."""
    record = _record(sam3=er.Sam3Response(attempted=True, responded=False,
                                          error="timeout"))
    record.add_reason(rc.OBJECT_AWAY_FROM_WRIST.code)   # would say no_action
    assert fusion.classify_record(record) == fusion.NEEDS_BETTER_VIEW


def test_not_confirmed_is_not_no_action():
    record = _record(sam3=er.Sam3Response(attempted=True, responded=True))
    record.add_reason(rc.SAM3_NOT_CONFIRMED.code)
    assert fusion.classify_record(record) == fusion.NEEDS_BETTER_VIEW


def test_equipment_context_suppresses_the_proposal():
    record = _record(sam3=er.Sam3Response(attempted=True, responded=True))
    record.add_reason(rc.OBJECT_AT_WRIST.code)
    record.add_reason(rc.SAM3_EQUIPMENT_CONTEXT.code)
    assert fusion.classify_record(record) == fusion.CONTEXT_OBSERVATION


# --- fusion -------------------------------------------------------------------

def _profile(**kw):
    base = dict(track_id=7, seat_state="unattributed",
                coverage_of_recording=0.9, resolvable_samples=200,
                fraction_away_from_baseline=0.0, samples_turned_left=0,
                samples_turned_right=0, longest_absence_ms=0.0)
    base.update(kw)
    return base


def _evidence(**kw):
    base = dict(track_id=7, sightings=100, wrist_resolved_sightings=100,
                raw_detections=0, kept_detections=0, rejected_too_large=0,
                rejected_far_from_hands=0, rejected_no_wrist=0,
                episodes=[], longest_episode_ms=0.0, total_handling_ms=0.0)
    base.update(kw)
    return base


def test_fragment_tracks_abstain_rather_than_clearing():
    out = fusion.route(_profile(coverage_of_recording=0.01))
    assert out.state == fusion.NEEDS_BETTER_VIEW
    assert rc.TRACK_FRAGMENT.code in out.reason_codes


def test_unresolvable_head_is_never_reported_as_normal():
    out = fusion.route(_profile(resolvable_samples=0))
    assert rc.ORIENTATION_NOT_ASSESSABLE.code in out.reason_codes
    assert out.state != fusion.NO_ACTION


def test_talking_is_declared_unmeasurable_on_every_track():
    out = fusion.route(_profile())
    assert rc.TALKING_NOT_MEASURABLE.code in out.reason_codes


def test_orientation_alone_does_not_reach_review():
    """Looking away from your own baseline is not, by itself, an allegation."""
    out = fusion.route(_profile(fraction_away_from_baseline=0.95))
    assert rc.ORIENTATION_AWAY_SUSTAINED.code in out.reason_codes
    assert out.state != fusion.REVIEW_CANDIDATE


def test_isolated_burst_does_not_recur_into_review():
    """Four 250 ms sampler hits are one burst, not a pattern."""
    out = fusion.route(
        _profile(),
        _evidence(raw_detections=21, kept_detections=11,
                  episodes=[[0, 250], [1000, 1250], [2000, 2250], [3000, 3250]],
                  longest_episode_ms=250.0, total_handling_ms=1000.0))
    assert out.state != fusion.REVIEW_CANDIDATE
    lasts = [c for c in out.conditions if c.name == "lasts_or_recurs"][0]
    assert lasts.passed is False


def test_sustained_supported_handling_reaches_review():
    records = []
    for _ in range(5):
        record = _record(sam3=er.Sam3Response(attempted=True, responded=True))
        record.add_reason(rc.SAM3_SUPPORTED.code)
        records.append(record)
    out = fusion.route(
        _profile(),
        _evidence(raw_detections=303, kept_detections=218,
                  episodes=[[0, 56500]], longest_episode_ms=56500.0,
                  total_handling_ms=56500.0),
        records)
    assert out.state == fusion.REVIEW_CANDIDATE
    assert all(c.passed for c in out.conditions)


def test_equipment_dominance_blocks_the_object_route():
    """The 04_talking failure mode: a seat whose paper detections are its
    keyboard. 402 of 409 adjudications were equipment there."""
    records = []
    for _ in range(4):
        record = _record(sam3=er.Sam3Response(attempted=True, responded=True))
        record.add_reason(rc.SAM3_SUPPORTED.code)
        records.append(record)
    for _ in range(96):
        record = _record(sam3=er.Sam3Response(attempted=True, responded=True))
        record.add_reason(rc.SAM3_EQUIPMENT_CONTEXT.code)
        records.append(record)
    out = fusion.route(
        _profile(),
        _evidence(raw_detections=445, kept_detections=307,
                  episodes=[[0, 21800]], longest_episode_ms=21800.0,
                  total_handling_ms=21800.0),
        records)
    assert out.state != fusion.REVIEW_CANDIDATE
    blocked = [c for c in out.conditions
               if c.name == "no_dominant_equipment_explanation"][0]
    assert blocked.passed is False


def test_unrefereed_object_evidence_is_marked_as_such():
    out = fusion.route(
        _profile(),
        _evidence(raw_detections=21, kept_detections=11,
                  episodes=[[0, 9000]], longest_episode_ms=9000.0,
                  total_handling_ms=9000.0),
        [_record()])
    assert rc.SAM3_NOT_RUN.code in out.reason_codes


def test_permitted_material_policy_demotes_paper_handling():
    """A hall that issues rough sheets must be able to say so."""
    args = (_profile(),
            _evidence(raw_detections=303, kept_detections=218,
                      episodes=[[0, 56500]], longest_episode_ms=56500.0,
                      total_handling_ms=56500.0),
            [_record()])
    assert fusion.route(*args).state == fusion.REVIEW_CANDIDATE
    permissive = fusion.Policy(loose_paper_prohibited=False)
    demoted = fusion.route(*args, policy=permissive)
    assert demoted.state != fusion.REVIEW_CANDIDATE
    assert rc.POLICY_MATERIAL_PERMITTED.code in demoted.reason_codes


def test_machine_code_cannot_write_a_human_verdict():
    for state in fusion.HUMAN_STATES:
        with pytest.raises(ValueError, match="only be written by a reviewer"):
            fusion.assert_machine_state(state)


def test_funnel_refuses_to_call_itself_accuracy():
    report = fusion.funnel([fusion.route(_profile())])
    assert "not accuracy" in report["note"]
    assert set(report["states"]) == set(fusion.ALL_STATES)


def test_named_phone_still_requires_association():
    """SAM 3 naming a phone says an object exists, not whose it is.

    A phone on a neighbouring desk, or in an invigilator's hand, falls inside
    the crop cut for this track. Routing on the name alone would make it
    evidence against whoever the crop belonged to.
    """
    unassociated_record = _record(
        association=er.Association(track_id=7, wrist_resolved=True,
                                   wrist_distance_norm=0.76),
        sam3=er.Sam3Response(attempted=True, responded=True))
    unassociated_record.add_reason(rc.SAM3_PHONE_NAMED.code)
    unassociated = fusion.route(
        _profile(),
        _evidence(sightings=100, wrist_resolved_sightings=5,
                  raw_detections=40, kept_detections=12,
                  episodes=[[0, 9000]], longest_episode_ms=9000.0,
                  total_handling_ms=9000.0),
        [unassociated_record])
    assert unassociated.state == fusion.NEEDS_BETTER_VIEW
    assert rc.SAM3_PHONE_NAMED.code in unassociated.reason_codes

    associated_record = _record(
        association=er.Association(track_id=7, wrist_resolved=True,
                                   wrist_distance_norm=0.20,
                                   nearest_wrist="right"),
        sam3=er.Sam3Response(attempted=True, responded=True))
    associated_record.add_reason(rc.SAM3_PHONE_NAMED.code)
    associated = fusion.route(
        _profile(),
        _evidence(sightings=100, wrist_resolved_sightings=100,
                  raw_detections=40, kept_detections=12,
                  episodes=[[0, 9000]], longest_episode_ms=9000.0,
                  total_handling_ms=9000.0),
        [associated_record])
    assert associated.state == fusion.REVIEW_CANDIDATE


def test_phone_association_cannot_be_inferred_from_track_coverage():
    """A wrist elsewhere in the track does not own this phone frame."""
    record = _record(
        association=er.Association(track_id=7, wrist_resolved=True,
                                   wrist_distance_norm=-1.0),
        sam3=er.Sam3Response(attempted=True, responded=True))
    record.add_reason(rc.SAM3_PHONE_NAMED.code)
    out = fusion.route(
        _profile(),
        _evidence(sightings=100, wrist_resolved_sightings=100,
                  raw_detections=4, kept_detections=1,
                  episodes=[[0, 9000]], longest_episode_ms=9000.0,
                  total_handling_ms=9000.0),
        [record])
    assert out.state == fusion.NEEDS_BETTER_VIEW
    association = next(c for c in out.conditions
                       if c.name == "associated_with_this_person")
    assert association.passed is False


def test_corroboration_uses_rate_after_episode_sampling():
    records = []
    for code in (rc.SAM3_SUPPORTED.code, rc.SAM3_SUPPORTED.code,
                 rc.SAM3_NOT_CONFIRMED.code):
        record = _record(sam3=er.Sam3Response(attempted=True, responded=True))
        record.add_reason(code)
        records.append(record)
    out = fusion.route(
        _profile(),
        _evidence(raw_detections=9, kept_detections=5,
                  episodes=[[0, 9000]], longest_episode_ms=9000.0,
                  total_handling_ms=9000.0), records)
    assert out.state == fusion.REVIEW_CANDIDATE
    condition = next(c for c in out.conditions
                     if c.name == "sam3_supports_or_cannot_exclude")
    assert "2 of 3" in condition.detail


def test_corroboration_rate_supplements_the_count_and_never_replaces_it():
    """The rate is an extra way to pass, never a stricter one.

    Replacing the count floor with a 2/3 rate dropped 12_paper tracks 118
    (5 of 19 supported) and 53 (5 of 29) out of review, and both are real
    paper handling. SAM 3 is a conservative referee on this corpus; a
    rate-only test deletes true positives.
    """
    guard = verdict.Guard()

    # The two measured true positives: low rate, but the count clears it.
    assert verdict.corroborated_enough(5, 19, guard) is True
    assert verdict.corroborated_enough(5, 29, guard) is True

    # The small-denominator rescue the rate exists for.
    assert verdict.corroborated_enough(2, 2, guard) is True
    assert verdict.corroborated_enough(2, 3, guard) is True

    # Genuinely unsupported stays unsupported.
    assert verdict.corroborated_enough(0, 42, guard) is False
    assert verdict.corroborated_enough(1, 14, guard) is False
    assert verdict.corroborated_enough(0, 0, guard) is False

    # The invariant: adding the rate may only ever ADD passes.
    for supported in range(0, 8):
        for adjudicated in range(supported, 40):
            if supported >= guard.min_corroborations_indicator:
                assert verdict.corroborated_enough(supported, adjudicated,
                                                   guard) is True, \
                    f"count floor regressed at {supported}/{adjudicated}"


# --- the referee's match test -------------------------------------------------

def test_overlap_is_symmetric_so_loose_proposals_still_match():
    """A loose proposal box must not throw away a correct SAM 3 answer.

    D-FINE's COCO boxes are often a hand plus a stretch of desk (~100x70 px);
    a real phone segment is ~26x21. Dividing only by the proposal's area caps
    the overlap at 0.078, under any threshold — so on run 1506, 1,438 of 1,982
    phone proposals were recorded `unsupported`, including frames holding the
    same phone confirmed at 85.0s.
    """
    from tools.adjudicate_with_sam3 import MIN_OVERLAP, overlap_fraction

    loose_proposal = (0.0, 0.0, 100.0, 70.0)
    tight_phone = (30.0, 20.0, 56.0, 41.0)
    assert overlap_fraction(loose_proposal, tight_phone) >= MIN_OVERLAP

    # The direction that already worked must keep working: a small proposal
    # sitting inside a large equipment segment is still suppressed.
    small_proposal = (30.0, 20.0, 56.0, 41.0)
    monitor = (0.0, 0.0, 300.0, 200.0)
    assert overlap_fraction(small_proposal, monitor) >= MIN_OVERLAP

    # Genuinely disjoint boxes still score zero.
    assert overlap_fraction((0.0, 0.0, 10.0, 10.0),
                            (50.0, 50.0, 60.0, 60.0)) == 0.0


def test_suppression_requires_the_equipment_to_cover_the_proposal():
    """Confirmation and suppression must not share a match test.

    Using the symmetric test for suppression on 12_paper collapsed
    corroborations 136 -> 20, drove suppressions 48 -> 314, and pushed the one
    man who actually took the paper (tracks 118 and 53) out of the queue.
    """
    from tools.adjudicate_with_sam3 import MIN_OVERLAP, covers, overlap_fraction

    big_paper_proposal = (0.0, 0.0, 100.0, 70.0)
    small_mouse = (40.0, 30.0, 60.0, 45.0)

    # A mouse sitting inside a large paper proposal does not make it a mouse.
    assert covers(big_paper_proposal, small_mouse) < MIN_OVERLAP
    # ...while the symmetric test, used for confirmation, would have matched.
    assert overlap_fraction(big_paper_proposal, small_mouse) >= MIN_OVERLAP

    # Equipment that genuinely covers the proposal still suppresses it.
    small_proposal = (40.0, 30.0, 60.0, 45.0)
    monitor = (0.0, 0.0, 300.0, 200.0)
    assert covers(small_proposal, monitor) >= MIN_OVERLAP


# --- episode sampling ---------------------------------------------------------

def test_stationary_object_becomes_one_episode():
    """A phone lying still for a minute is one question, not 240."""
    from pipeline import episodes as ep

    proposals = [(t * 250.0, 7, {"cls": "phone", "confidence": 0.5,
                                 "box": [100.0, 100.0, 126.0, 121.0]})
                 for t in range(240)]
    built = ep.build(proposals)
    assert len(built) == 1
    assert len(built[0]) == 240
    assert ep.representatives(built[0]) != built[0].members  # sampled, not all


def test_objects_in_different_places_stay_separate():
    from pipeline import episodes as ep

    proposals = [(0.0, 7, {"cls": "phone", "confidence": 0.5,
                           "box": [100.0, 100.0, 126.0, 121.0]}),
                 (250.0, 7, {"cls": "phone", "confidence": 0.5,
                             "box": [600.0, 400.0, 626.0, 421.0]})]
    assert len(ep.build(proposals)) == 2


def test_a_long_gap_starts_a_new_episode():
    from pipeline import episodes as ep

    box = [100.0, 100.0, 126.0, 121.0]
    proposals = [(0.0, 7, {"cls": "phone", "confidence": 0.5, "box": box}),
                 (30000.0, 7, {"cls": "phone", "confidence": 0.5, "box": box})]
    assert len(ep.build(proposals)) == 2


def test_representatives_include_first_strongest_and_last():
    """Sampling must not be by confidence alone — the edges carry persistence."""
    from pipeline import episodes as ep

    box = [100.0, 100.0, 126.0, 121.0]
    members = []
    for i in range(20):
        conf = 0.9 if i == 11 else 0.4
        members.append((i * 250.0, 7,
                        {"cls": "phone", "confidence": conf, "box": box}))
    episode = ep.build(members)[0]
    picked = ep.representatives(episode, ep.Config(max_per_episode=3))
    stamps = [p[0] for p in picked]
    assert episode.start_ms in stamps
    assert episode.end_ms in stamps
    assert 11 * 250.0 in stamps          # the detector's strongest look


def test_single_frame_episode_yields_one_call():
    from pipeline import episodes as ep

    one = [(0.0, 7, {"cls": "phone", "confidence": 0.5,
                     "box": [100.0, 100.0, 126.0, 121.0]})]
    assert len(ep.representatives(ep.build(one)[0])) == 1
