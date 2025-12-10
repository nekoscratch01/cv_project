from __future__ import annotations

from typing import Protocol, List, Tuple


class VectorStorePort(Protocol):
    """Vector store abstraction for similarity search."""

    async def upsert(self, item_id: str, embedding: List[float], metadata: dict | None = None) -> None:
        ...

    async def search(self, query: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        ...
