"""Legacy location of the light calibration (moved to ``application.calibrate`` and
``adapters.siril_calibrator``)."""

from .adapters.siril_calibrator import SirilCalibrator, provenance_header, thermal_dark  # noqa: F401
from .domain.calibration import CALIB_VERSION, safe_stem  # noqa: F401
