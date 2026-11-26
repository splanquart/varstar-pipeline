"""Legacy location of the image I/O helpers (moved to ``varstar.adapters.images``)."""

from .adapters.images import (NORM_SCALE, XisfError, read_adu, read_siril_output, read_xisf,  # noqa: F401
                              write_float32, write_mask)
