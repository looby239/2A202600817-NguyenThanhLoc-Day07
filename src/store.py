from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import os
            use_chroma_env = os.getenv("USE_CHROMA", "false").lower() in ("true", "1")
            if use_chroma_env:
                import chromadb

                # Ephemeral in-process client; the in-memory self._store stays the
                # source of truth so every operation behaves identically with or
                # without ChromaDB installed (Chroma is a bonus mirror here).
                self._client = chromadb.Client()
                self._collection = self._client.get_or_create_collection(name=collection_name)
                self._use_chroma = True
            else:
                self._use_chroma = False
                self._collection = None
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document, embedding: list[float] | None = None) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)
        if embedding is None:
            embedding = self._embedding_fn(doc.content)
        record = {
            "id": f"{doc.id}#{self._next_index}",
            "doc_id": doc.id,
            "content": doc.content,
            "embedding": list(embedding),
            "metadata": metadata,
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        results = [
            {
                "id": rec["id"],
                "doc_id": rec["doc_id"],
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": compute_similarity(query_embedding, rec["embedding"]),
            }
            for rec in records
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        # Batch-embed in one shot when the backend supports it (cheaper API use).
        embed_batch = getattr(self._embedding_fn, "embed_batch", None)
        if embed_batch is not None:
            embeddings = embed_batch([doc.content for doc in docs])
        else:
            embeddings = [self._embedding_fn(doc.content) for doc in docs]

        for doc, embedding in zip(docs, embeddings):
            record = self._make_record(doc, embedding)
            self._store.append(record)
            if self._use_chroma and self._collection is not None:
                try:
                    self._collection.add(
                        ids=[record["id"]],
                        documents=[record["content"]],
                        embeddings=[record["embedding"]],
                        metadatas=[record["metadata"]],
                    )
                except Exception:
                    pass

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter:
            records = [
                rec
                for rec in self._store
                if all(rec["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            records = self._store
        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        to_remove = [rec for rec in self._store if rec["metadata"].get("doc_id") == doc_id]
        if not to_remove:
            return False

        self._store = [rec for rec in self._store if rec["metadata"].get("doc_id") != doc_id]
        if self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(ids=[rec["id"] for rec in to_remove])
            except Exception:
                pass
        return True
