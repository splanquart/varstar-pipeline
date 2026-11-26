"""Application layer: use cases (inventory, check, masters, calibrate, pipeline).

Use cases orchestrate the domain through ports (``application.ports``, typing.Protocol) and
never import ``varstar.adapters`` or ``varstar.cli``. Enforced by tests/architecture/test_layers.py.
"""
