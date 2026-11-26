"""The masters plan: typed in memory, historical JSON layout on disk."""

import json

from hypothesis import given
from hypothesis import strategies as st

from varstar.domain.plan import DarkflatRef, DarkRef, FlatRef, GroupPlan, MastersPlan

text = st.text("abcdefgh_-/0123456789", min_size=1, max_size=12)
floats = st.floats(0.01, 1000, allow_nan=False)


@st.composite
def dark_refs(draw):
    mode = draw(st.sampled_from(["direct", "scaled"]))
    scaled = mode == "scaled"
    return DarkRef(draw(text), draw(text), mode, draw(st.integers(1, 50)), draw(floats),
                   draw(floats) if scaled else None, draw(text) if scaled else None)


groups = st.builds(GroupPlan, text, text, st.none() | text, text, floats, st.sampled_from("ABC"),
                   st.lists(text, max_size=3), st.lists(text, max_size=3), st.none() | dark_refs(),
                   st.none() | st.builds(FlatRef, text, text, st.integers(1, 50)),
                   st.none() | st.builds(DarkflatRef, text, text, floats), st.none() | text, st.none() | text)
plans = st.builds(MastersPlan, text, text, text, st.dictionaries(text, groups, max_size=3),
                  st.dictionaries(text, st.fixed_dictionaries({"kind": text, "n": st.integers(1, 9)}), max_size=2),
                  st.lists(text, max_size=3))


@given(plans)
def test_plan_round_trips_through_json(plan):
    data = json.loads(json.dumps(plan.to_dict()))
    assert MastersPlan.from_dict(data) == plan
    assert MastersPlan.from_dict(data).to_dict() == data


def test_historical_layout():
    direct = DarkRef("d", "/lib/d/master.fits", "direct", 8, 300.0, offset_adu=12.0, offset_master="o")
    assert direct.to_dict() == {"id": "d", "path": "/lib/d/master.fits", "mode": "direct", "n": 8, "exptime": 300.0}
    scaled = DarkRef("d", "p", "scaled", 8, 180.0, 512.0, "o").to_dict()
    assert scaled["offset_adu"] == 512.0 and scaled["offset_master"] == "o"
    g = GroupPlan("2021-09-06_H_300s", "2021-09-06", None, "H", 300.0, "C", [], ["/raw/a.fits"])
    assert list(g.to_dict()) == ["key", "session", "label", "filter", "exptime", "level", "reasons", "files",
                                 "dark", "flat", "darkflat", "bpm", "ampglow"]
    assert g.display == "2021-09-06 / H"
    assert list(MastersPlan("t", "/lib", "1.4.5").to_dict()) == ["created", "library", "siril_version", "groups",
                                                                 "masters", "warnings"]
