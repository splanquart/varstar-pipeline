"""``InventoryReport`` adapter: inventory.json content and inventory.md text."""

from __future__ import annotations

from typing import Any

from ...domain.model import Inventory
from ..serialization import inventory_to_dict
from .markdown import inventory_markdown
from .qa_markdown import items_markdown


class InventoryReportAdapter:
    def to_dict(self, inv: Inventory) -> dict[str, Any]:
        return inventory_to_dict(inv)

    def markdown(self, inv: Inventory, qa_items: list[Any]) -> str:
        return inventory_markdown(inv, items_markdown(qa_items, "## Contrôles qualité (en-têtes)"))
