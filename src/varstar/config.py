"""Legacy location of the configuration loading (moved to ``varstar.adapters.config_loader``)."""

from .adapters.config_loader import deep_merge, load_defaults, load_raw, resolve_workdir  # noqa: F401
from .adapters.config_loader import load_settings as load_config  # noqa: F401
from .domain.settings import optical_setup_periods  # noqa: F401
