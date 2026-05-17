from __future__ import annotations

from typing import Any

import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class LlamaCppProvider(EmbeddingProvider):
    """
    OpenAI-compatible llama.cpp HTTP embedding provider.
    """

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._embedding_url = f"{self.base_url}/v1/embeddings"
        self._vector_size: int | None = None

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        return await self._request_embeddings(documents)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        embeddings = await self._request_embeddings([query])
        return embeddings[0]

    def get_vector_name(self) -> str:
        """Get the name of the vector for the Qdrant collection."""
        model_slug = self.model_name.split("/")[-1].lower().replace(".", "-")
        return f"llama-{model_slug}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        if self._vector_size is not None:
            return self._vector_size

        payload = self._build_payload(["dimension probe"])
        with httpx.Client(timeout=30.0) as client:
            response = client.post(self._embedding_url, json=payload)
            response.raise_for_status()
            data = response.json()

        embedding = self._extract_embeddings(data)[0]
        self._vector_size = len(embedding)
        return self._vector_size

    async def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        payload = self._build_payload(inputs)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self._embedding_url, json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = self._extract_embeddings(data)
        if self._vector_size is None and embeddings:
            self._vector_size = len(embeddings[0])
        return embeddings

    def _build_payload(self, inputs: list[str]) -> dict[str, Any]:
        return {"input": inputs, "model": self.model_name}

    @staticmethod
    def _extract_embeddings(data: dict[str, Any]) -> list[list[float]]:
        items = data.get("data", [])
        if not items:
            raise ValueError("No embeddings returned from llama.cpp server")
        items = sorted(items, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]
