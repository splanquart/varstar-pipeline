"""ADU scale detection (14-bit samples stored ×4 by N.I.N.A.): measured, never hard-coded."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from varstar.domain.units import MAX_SCALE, detect_adu_scale


def test_adu_scale_detection():
    rng = np.random.default_rng(0)
    assert detect_adu_scale((rng.integers(1, 16000, 10000) * 4).astype(np.uint16)) == 4
    assert detect_adu_scale(rng.integers(1, 65000, 10000).astype(np.uint16) | 1) == 1
    assert detect_adu_scale(np.zeros(10, dtype=np.uint16)) == 1
    assert detect_adu_scale(np.array([0.5, 4.0])) == 1


@given(arrays(np.int64, st.integers(1, 200), elements=st.integers(0, 4000)),
       st.sampled_from([1, 2, 4, 8, 16, 32, 64]))
def test_scale_is_the_largest_power_of_two_dividing_the_data(base, k):
    base[0] = 1                                   # at least one odd sample: base scale is exactly 1
    data = (base * k).astype(np.uint16 if k <= 16 else np.int64)
    assert detect_adu_scale(data) == min(k, MAX_SCALE)


@given(arrays(np.float64, st.integers(1, 50), elements=st.floats(1, 1000)).filter(
    lambda a: not np.all(a == np.round(a))))
def test_non_integer_data_has_scale_one(data):
    assert detect_adu_scale(data) == 1
