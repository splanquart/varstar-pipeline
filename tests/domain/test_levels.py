"""Calibration levels A/B/C: `worst` is the only way levels are combined."""

from hypothesis import given
from hypothesis import strategies as st

from varstar.domain.model import LEVELS, worst

level = st.sampled_from(LEVELS)
maybe_level = st.one_of(st.none(), level)


def test_worst_examples():
    assert worst("A", "B") == "B"
    assert worst("A", None, "A") == "A"
    assert worst() == "C"                 # nothing known → degraded, never silently A
    assert worst(None, None) == "C"


@given(st.lists(maybe_level, max_size=6))
def test_worst_ignores_order(levels):
    assert worst(*levels) == worst(*reversed(levels)) == worst(*sorted(levels, key=str))


@given(st.lists(maybe_level, max_size=6), maybe_level)
def test_worst_is_idempotent_and_associative(levels, extra):
    assert worst(worst(*levels), worst(*levels)) == worst(*levels)
    if any(levels):
        assert worst(worst(*levels), extra) == worst(*levels, extra)


@given(st.lists(level, min_size=1, max_size=6))
def test_worst_is_never_better_than_any_component(levels):
    result = worst(*levels)
    assert result in LEVELS
    assert all(LEVELS.index(result) >= LEVELS.index(lv) for lv in levels)
    assert result in levels
