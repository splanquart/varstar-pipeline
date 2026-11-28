"""A* checks: what the WCS solutions say about the acquisition."""

from builders import make_settings
from varstar.domain.qa.astrometry import astrometry_checks

A = make_settings().astrometry


def row(raw, ok=True, *, orientation=180.6, scale=3.4877, center=(60.44, 36.11), segment=1, time="2021-09-06T23:00:00",
        reason=None):
    return {"raw": raw, "ok": ok, "reason": reason, "scale_arcsec": scale if ok else None,
            "orientation_deg": orientation if ok else None, "center": list(center) if ok else None,
            "segment": segment if ok else None, "time": time, "field_deg": [4.0, 2.7]}


def group(key, session, images, *, label=None, filt="H", flat="f1", expected=3.4855, borrowed=False):
    return {"key": key, "session": session, "label": label or session, "filter": filt, "flat_id": flat,
            "flat_borrowed": borrowed,
            "expected_scale": expected, "focallen": 274.0, "xpixsz": 4.63, "images": images}


def checks(groups):
    return {(i.check, i.severity): i for i in astrometry_checks(groups, A, 0.10)}


def test_clean_run_has_no_finding():
    g = group("g1", "2021-09-06", [row("a"), row("b")])
    assert checks([g]) == {}


def test_unsolved_images():
    part = group("g1", "2021-09-06", [row("a"), row("b", ok=False, reason="nuages")])
    every = group("g2", "2021-09-07", [row("c", ok=False, reason="nuages"), row("d", ok=False, reason="nuages")])
    found = checks([part, every])
    assert "1/2" in found[("A1", "warning")].message and found[("A1", "warning")].files == ["b"]
    assert "nuages (2)" in found[("A1", "error")].message


def test_focal_length_from_the_scale():
    g = group("g1", "2021-09-06", [row("a", scale=3.60), row("b", scale=3.60)])
    a2 = checks([g])[("A2", "info")]
    assert "focale effective ≈ 265 mm" in a2.message


def test_framing_shift_between_sessions():
    g1 = group("g1", "2021-09-06", [row("a"), row("b")])
    g2 = group("g2", "2021-09-02", [row("c"), row("d")])
    g3 = group("g3", "2021-09-12", [row("e", center=(60.44, 36.60))])
    a3 = checks([g1, g2, g3])[("A3", "warning")]
    assert a3.data["group"] == "g3" and 25 < a3.data["offset_arcmin"] < 35


def test_meridian_flip_segments():
    g = group("g1", "2021-09-06", [row("a", time="2021-09-06T23:00:00"),
                                   row("b", orientation=0.6, segment=2, time="2021-09-07T02:00:00")])
    a4 = checks([g])[("A4", "info")]
    assert "2 segments" in a4.message and "23:00–23:00" in a4.message


def rotation_case(**answers):
    """sept06 owns flat f1 (angle 0.6°); sept02 borrows it at 2.0° (camera turned by 1.4°)."""
    from builders import make_settings
    a = make_settings(astrometry=answers).astrometry if answers else A
    g1 = group("g1", "2021-09-06", [row("a"), row("b", orientation=0.7, segment=2)])   # flip: same camera angle
    g2 = group("g2", "2021-09-02", [row("c", orientation=182.0)], borrowed=True)
    return {(i.check, i.severity): i for i in astrometry_checks([g1, g2], a, 0.10)}


def test_rotation_with_a_borrowed_flat_asks_the_question():
    found = rotation_case()
    assert "2021-09-02 2.0°" in found[("A5", "info")].message          # angles per night, information
    q = found[("A5", "warning")]
    assert "QUESTION" in q.message and q.data["question"] and q.data["night"] == "2021-09-02"
    assert q.data["flat_nights"] == ["2021-09-06"] and 1.2 < q.data["spread_deg"] < 1.6
    assert "whole_train_rotation_nights: {2021-09-02: true}" in q.advice


def test_rotation_answers():
    # Whole optical train turned (D18): the flat is still valid.
    ok = rotation_case(whole_train_rotation_nights={"2021-09-02": True})
    assert ("A5", "warning") not in ok and ("A5", "error") not in ok
    assert any("flat valable" in i.message for k, i in ok.items() if k[0] == "A5")
    # Camera alone turned: the flat is wrong — an error, with what to do.
    bad = rotation_case(whole_train_rotation=False)
    e = bad[("A5", "error")]
    assert "flat inadapté" in e.message and "same_optical_setup" in e.advice and e.data["answer"] is False
    # The answer for the night wins over the default one.
    assert ("A5", "error") not in rotation_case(whole_train_rotation=False,
                                                whole_train_rotation_nights={"2021-09-02": True})


def test_rotation_without_borrowed_flat_is_only_information():
    g1 = group("g1", "2021-09-06", [row("a")])
    g3 = group("g3", "2021-09-02", [row("c", orientation=182.0)], flat="f2")
    assert {(i.check, i.severity) for i in astrometry_checks([g1, g3], A, 0.10)} == {("A5", "info")}
