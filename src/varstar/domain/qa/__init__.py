"""Early quality checks (docs/QA_CHECKS.md), pure part: every rule works on frames, sets and
pixel statistics already measured; no file is opened here.

- ``model``: ``QAItem``, severities, labels shared by the rules;
- ``headers``: header checks (H*), from the inventory only;
- ``darks``: dark levels per series (P1 groundwork) and the darks actually combined (D16);
- ``pixels``: statistics computed on pixel arrays (quick stats, binned flats, calibrated frames);
- ``raw``: raw pixel checks (P*), from per-frame quick statistics;
- ``calibrated``: post-calibration checks (C*).

Nothing is ever corrected: every finding is reported with numbers, the files concerned, its
likely photometric impact and an advice for the next observing session.
"""
