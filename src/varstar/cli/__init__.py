"""Command line (``varstar …``): the composition root of the application.

- ``app``: the Typer commands (French messages);
- ``bootstrap``: the only place where adapters are instantiated and given to the use cases.
"""

from .app import app, main  # noqa: F401
