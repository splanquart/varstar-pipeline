"""Compatibility shim (refactoring step 3): JSON in :mod:`varstar.adapters.serialization`, Markdown in
:mod:`varstar.adapters.reporting.markdown`. Removed at step 8.
"""

from .adapters.reporting.markdown import SEVERITY_FR, inventory_markdown, todo_markdown  # noqa: F401
from .adapters.serialization import inventory_to_dict  # noqa: F401
