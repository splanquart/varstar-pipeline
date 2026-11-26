"""``PixInsightMasters``: finding and reading the PixInsight masters (cross-validation)."""

from pathlib import Path

import numpy as np

from fitsgen import write_xisf
from varstar.adapters.images import NORM_SCALE
from varstar.adapters.pixinsight import PixInsightMasters
from varstar.domain.calibration import MasterRecord


def rec(kind, **meta):
    return MasterRecord("id", kind, Path("/lib/x"), meta)


def test_find_by_exposure_and_by_filter_and_session(tmp_path):
    d = tmp_path / "output" / "master"
    d.mkdir(parents=True)
    write_xisf(d / "masterDark_BINNING-2_EXPOSURE-300.00s.xisf", np.full((4, 4), 0.01, np.float32))
    write_xisf(d / "masterFlat_BINNING-2_FILTER-H_mono_SESSION-sept06.xisf", np.ones((4, 4), np.float32))
    pi = PixInsightMasters.from_config({"paths": {"pixinsight_masters": "auto"}}, tmp_path)
    assert pi.find(rec("dark", exptime=300.0), []).endswith("EXPOSURE-300.00s.xisf")
    assert pi.find(rec("darkflat", exptime=1.13), []) is None
    assert pi.find(rec("flat", filter="H"), ["autre", "sept06"]).endswith("SESSION-sept06.xisf")
    assert pi.find(rec("flat", filter="H"), ["sept02"]) is None
    assert np.allclose(pi.load_adu(pi.find(rec("dark", exptime=300.0), [])), 0.01 * NORM_SCALE)


def test_from_config_disabled_or_missing(tmp_path):
    assert PixInsightMasters.from_config({"paths": {"pixinsight_masters": None}}, tmp_path) is None
    assert PixInsightMasters.from_config({"paths": {"pixinsight_masters": "auto"}}, tmp_path) is None
    assert PixInsightMasters.from_config({"paths": {"pixinsight_masters": str(tmp_path)}}, tmp_path).dir == tmp_path
