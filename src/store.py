from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
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

        import os

        if os.getenv("USE_CHROMA") == "true":
            try:
                import chromadb

                self._client = chromadb.Client()
                self._collection = self._client.get_or_create_collection(name=collection_name)
                self._use_chroma = True
            except Exception:
                self._use_chroma = False
                self._collection = None
        else:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        meta = dict(doc.metadata) if doc.metadata is not None else {}
        meta["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": meta,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        from .chunking import compute_similarity
        query_vector = self._embedding_fn(query)
        scored = []
        for rec in records:
            score = compute_similarity(query_vector, rec["embedding"])
            scored.append({
                "id": rec["id"],
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma:
            ids = [doc.id for doc in docs]
            documents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            metadatas = []
            for doc in docs:
                meta = dict(doc.metadata) if doc.metadata is not None else {}
                meta["doc_id"] = doc.id
                metadatas.append(meta)
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self.search_with_filter(query, top_k=top_k, metadata_filter=None)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            from .chunking import compute_similarity
            query_vector = self._embedding_fn(query)
            where_clause = metadata_filter if metadata_filter else None
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "embeddings"]
            )
            formatted_results = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                embeddings = results["embeddings"][0]
                for i in range(len(ids)):
                    score = compute_similarity(query_vector, embeddings[i])
                    formatted_results.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i],
                        "score": score
                    })
                formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results
        else:
            # In-memory
            filtered_records = []
            for rec in self._store:
                match = True
                if metadata_filter:
                    for k, v in metadata_filter.items():
                        if rec["metadata"].get(k) != v:
                            match = False
                            break
                if match:
                    filtered_records.append(rec)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            count_before = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            count_after = self._collection.count()
            return count_after < count_before
        else:
            initial_size = len(self._store)
            self._store = [rec for rec in self._store if rec["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_size
