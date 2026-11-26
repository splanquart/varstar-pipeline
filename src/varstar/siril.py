"""Legacy location of the siril-cli wrapper (moved to ``varstar.adapters.siril``)."""

from .adapters.siril import (GUARD_MIN_VERSION, GUARD_MODES, PYTHON_INIT_GUARD, REQUIRED, Siril,  # noqa: F401
                             SirilError, link_sequence, parse_version)
