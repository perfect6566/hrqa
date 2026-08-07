"""Vector store for storing and retrieving document embeddings."""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import numpy.typing as npt


class VectorStore:
    """FAISS-based vector store for document retrieval."""

    def __init__(
        self,
        embedder,
        store_path: str = "./data/vector_store",
        dimension: Optional[int] = None
    ):
        """
        Initialize the vector store.

        Args:
            embedder: Embedder instance for encoding queries
            store_path: Path to persist the vector store
            dimension: Embedding dimension (auto-detected if not provided)
        """
        self.embedder = embedder
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension or embedder.get_embedding_dimension()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_ids: Dict[str, int] = {}

        self._load_existing()

    def _load_existing(self):
        """Load existing index and chunks if available."""
        index_file = self.store_path / "index.faiss"
        chunks_file = self.store_path / "chunks.json"

        if index_file.exists() and chunks_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(chunks_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)

                # Rebuild ID mapping
                self.chunk_ids = {chunk["id"]: i for i, chunk in enumerate(self.chunks)}
                print(f"Loaded existing index with {len(self.chunks)} chunks")
            except Exception as e:
                print(f"Error loading existing index: {e}")
                self._initialize_new_index()

    def _initialize_new_index(self):
        """Initialize a new FAISS index."""
        self.index = faiss.IndexIDMap(
            faiss.IndexFlatIP(self.dimension)
        )
        self.chunks = []
        self.chunk_ids = {}

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of document chunks with 'id' and 'content' fields

        Returns:
            Number of chunks added
        """
        if self.index is None:
            self._initialize_new_index()

        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Normalize embeddings for cosine similarity
        embeddings_matrix = np.array(embeddings).astype("float32")
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_matrix = embeddings_matrix / norms

        # Add to index with IDs
        start_idx = len(self.chunks)
        ids = list(range(start_idx, start_idx + len(chunks)))

        self.index.add_with_ids(embeddings_matrix, np.array(ids))

        # Store chunks and update mapping
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            if "embedding" in chunk_copy:
                del chunk_copy["embedding"]
            self.chunks.append(chunk_copy)
            self.chunk_ids[chunk["id"]] = start_idx + i

        return len(chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.

        Args:
            query: Query string
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of relevant chunks with scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Embed query
        query_embedding = np.array(self.embedder.embed_query(query)).astype("float32").reshape(1, -1)

        # Normalize query
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_norm[query_norm == 0] = 1
        query_embedding = query_embedding / query_norm

        # Search
        k_search = min(k * 3, self.index.ntotal)  # Over-fetch for filtering
        scores, indices = self.index.search(query_embedding, k_search)

        # Get chunks with scores
        results = []
        seen_ids = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx].copy()
            chunk_id = chunk["id"]

            # Skip duplicates
            if chunk_id in seen_ids:
                continue

            # Apply metadata filter
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if chunk.get("metadata", {}).get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            chunk["score"] = float(score)
            chunk["chunk_index"] = int(idx)
            results.append(chunk)
            seen_ids.add(chunk_id)

            if len(results) >= k:
                break

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        if chunk_id not in self.chunk_ids:
            return None

        idx = self.chunk_ids[chunk_id]
        return self.chunks[idx].copy()

    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific document."""
        return [
            chunk.copy() for chunk in self.chunks
            if chunk.get("document_id") == document_id
        ]

    def save(self):
        """Persist the vector store to disk."""
        if self.index is not None:
            index_file = self.store_path / "index.faiss"
            chunks_file = self.store_path / "chunks.json"

            faiss.write_index(self.index, str(index_file))

            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)

            print(f"Saved vector store with {len(self.chunks)} chunks")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None,
            "documents": len(set(c.get("document_id") for c in self.chunks)),
        }

    def clear(self):
        """Clear all data from the vector store."""
        self._initialize_new_index()
        self.save()
