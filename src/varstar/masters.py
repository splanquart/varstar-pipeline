"""Legacy location of the masters (moved to ``domain.calibration``, ``adapters.library`` and
``adapters.siril_masters``)."""

from .adapters.library import DiskMasterRepository as Library  # noqa: F401
from .adapters.siril_masters import frame_stats  # noqa: F401
from .domain.calibration import (MASTER_ALGO_VERSION, SATURATION_ADU, MasterRecord, Rejection,  # noqa: F401
                                 choose_rejection)
