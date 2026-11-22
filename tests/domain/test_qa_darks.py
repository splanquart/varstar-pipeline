"""Dark levels per series and the darks actually combined (D16), on frames and statistics in memory."""

import math

from hypothesis import example, given
from hypothesis import strategies as st

from builders import inventory_of, make_frame_stats, make_settings, series_frames, two_nights_frames
from varstar.domain.model import DARK, LIGHT
from varstar.domain.qa.darks import (classify_excess, consistent_dark_sets, dark_candidates, effective_dark_sets,
                                     level_clusters, set_summary)

LEVELS = {"dark/a": 516.0, "dark/b": 541.0, "dark/c": 516.0}


def stats_by_folder(frames, levels=LEVELS):
    return {str(f.path): make_frame_stats(next((v for k, v in levels.items() if f.rel_path.startswith(k)), 600.0))
            for f in frames}


def group(inv, label):
    return next(g for g in inv.light_groups if g.label == label)


def test_classify_excess():
    assert classify_excess(25, 12, 12, 0.25) == "offset"
    assert classify_excess(25, 12, math.sqrt(144 + 100), 0.25) == "lumière"
    assert classify_excess(25, 12, math.sqrt(144 + 50), 0.25) == "indéterminé"
    assert classify_excess(25, 12, 12, None) == "indéterminé"
    assert classify_excess(-3, 12, 12, 0.25) == "indéterminé"


def test_set_summary_is_the_median_of_the_frames():
    frames = series_frames("dark/a", DARK, 300, "2021-09-02T20:00:00", 3, step_s=310)
    inv = inventory_of(frames)
    (s,) = inv.calibration_sets[DARK]
    stats = {str(f.path): make_frame_stats(lv, sg, corners=c) for f, lv, sg, c in
             zip(s.frames, (510, 516, 530), (11, 12, 20), (0, 1, 5))}
    summ = set_summary(s, stats)
    assert (summ["level"], summ["sigma"]) == (516, 12)
    assert summ["corners"] == {k: 1.0 for k in summ["corners"]}


@given(st.lists(st.floats(0, 1000, allow_nan=False), max_size=30), st.floats(0.1, 50))
@example([0.0, 5.0, 10.0], 5.0)          # gaps exactly equal to the jump stay together
def test_level_clusters_partition_sorted_levels(levels, jump):
    items = list(enumerate(levels))
    clusters = level_clusters(items, jump)
    assert sorted(i for c in clusters for i in c) == list(range(len(levels)))
    value = dict(items)
    for c in clusters:
        vals = [value[i] for i in c]
        assert vals == sorted(vals)
        assert all(b - a <= jump for a, b in zip(vals, vals[1:]))
    for c1, c2 in zip(clusters, clusters[1:]):
        assert value[c2[0]] - value[c1[-1]] > jump


def test_dark_candidates_share_exposure_and_temperature():
    frames = two_nights_frames() + series_frames("dark/hot", DARK, 300, "2021-09-04T02:00:00", 5, step_s=310,
                                                 ccd_temp=-5.0)
    frames += series_frames("dark/180", DARK, 180, "2021-09-04T03:00:00", 5, step_s=190)
    inv = inventory_of(frames, make_settings(matching={"dark_same_night": False}))
    ids = [s.id for s in dark_candidates(inv, group(inv, "sept06"), inv.config)]
    assert ids == ["DARK-300s-20210902", "DARK-300s-20210907"]


def test_consistent_dark_sets_keeps_the_nearest_level():
    inv = inventory_of(two_nights_frames(), make_settings(matching={"dark_same_night": False}))
    g = group(inv, "sept06")
    stats = stats_by_folder(inv.frames)
    kept, note = consistent_dark_sets(g.dark.sets, stats, 5.0, g.start)
    assert [s.id for s in kept] == ["DARK-300s-20210907"] and "niveaux différents" in note
    # Same level: everything is combined.
    same = stats_by_folder(inv.frames, {"dark/": 516.0})
    assert consistent_dark_sets(g.dark.sets, same, 5.0, g.start) == (g.dark.sets, None)


def test_borrowed_darks_avoid_an_anomalous_night():
    frames = two_nights_frames() + [
        *series_frames("dark/c", DARK, 300, "2021-09-01T01:00:00", 5, step_s=310),
        *series_frames("light/H/SESSION_sept12", LIGHT, 300, "2021-09-12T23:00:00", 2, step_s=600, filter="H"),
    ]
    inv = inventory_of(frames)
    stats = stats_by_folder(inv.frames)
    g12 = group(inv, "sept12")
    assert [s.id for s in g12.dark.sets] == ["DARK-300s-20210907"]      # nearest night (headers only)
    sets, notes = effective_dark_sets(inv, g12, stats, inv.config)
    assert [s.id for s in sets] == ["DARK-300s-20210902"] and "offset anormal" in notes[0]
    # The anomalous night keeps its own darks, a normal night too.
    assert [s.id for s in effective_dark_sets(inv, group(inv, "sept06"), stats, inv.config)[0]] == ["DARK-300s-20210907"]
    assert effective_dark_sets(inv, group(inv, "sept02"), stats, inv.config)[1] == []


def test_borrowed_darks_kept_when_levels_agree():
    frames = two_nights_frames() + series_frames("light/H/SESSION_sept12", LIGHT, 300, "2021-09-12T23:00:00", 2,
                                                 step_s=600, filter="H")
    inv = inventory_of(frames)
    same = stats_by_folder(inv.frames, {"dark/": 516.0})
    sets, notes = effective_dark_sets(inv, group(inv, "sept12"), same, inv.config)
    assert [s.id for s in sets] == ["DARK-300s-20210907"] and notes == []
