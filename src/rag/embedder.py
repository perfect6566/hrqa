"""Embedder for creating vector representations of text chunks.

Uses ``fastembed`` (Qdrant) under the hood: ONNX Runtime + the same
HuggingFace weights that ``sentence-transformers`` would have downloaded,
but without the PyTorch / Transformers / tokenizers dependency chain.

This keeps the deployed image tiny (~50 MB for fastembed vs ~990 MB for
sentence-transformers) and the cold-start time short (~3-5 s vs ~30 s),
which is what lets the service run on Render's free 512 MB tier.

The public API of this module is intentionally unchanged from the
sentence-transformers version so the rest of the RAG pipeline (the
``VectorStore``, retriever, and CLI build script) continues to work
without modification.
"""

import os
from typing import List, Optional

import numpy as np


class Embedder:
    """Create embeddings for text chunks using local fastembed models."""

    # Model name → embedding dimension.
    #
    # We keep the same human-friendly names that the previous
    # sentence-transformers version exposed. Each one maps to a
    # fastembed-compatible identifier (the ``sentence-transformers/``
    # prefix tells fastembed to use the SBERT-exported ONNX weights).
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
    }

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedder.

        Args:
            model: Model name. Accepts either the bare SBERT name
                (e.g. ``"all-MiniLM-L6-v2"``) or the fully-qualified
                fastembed identifier
                (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``).
            batch_size: Number of texts to embed in each batch.
            device: Unused. Accepted only for backwards compatibility
                with the previous sentence-transformers API; fastembed
                auto-selects CPU/CUDA via ONNX Runtime.
        """
        # Accept the bare SBERT name and remap it to the fastembed-style
        # identifier so users do not need to know the prefix convention.
        if model in self.MODEL_DIMENSIONS and "/" not in model:
            # Prefer the fastembed-qualified name when available; fall
            # back to the bare name (fastembed resolves both).
            qualified = f"sentence-transformers/{model}"
            if qualified in self.MODEL_DIMENSIONS:
                model = qualified

        self.model_name = model
        self.batch_size = batch_size
        # ``device`` is accepted but ignored — kept only so callers that
        # pass ``device="cpu"`` (e.g. from environment plumbing) still work.
        self.device = device or "cpu"

        # Lazy: do NOT construct the TextEmbedding object here.
        # ``fastembed.TextEmbedding(...)`` triggers ONNX model download
        # and weights loading on construction. Doing it eagerly at
        # startup would defeat the purpose of lazy loading in the API
        # lifespan. Instead we build the model on first use.
        self._model = None

    @property
    def model(self):
        """Return the underlying ``TextEmbedding`` instance, building it lazily."""
        if self._model is None:
            # Imported lazily so that simply importing ``src.rag.embedder``
            # (which happens at module load time of the FastAPI app) does
            # not pull in ONNX Runtime or download model weights.
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (Python ``float`` lists so the
            rest of the pipeline can JSON-serialize them if needed).
        """
        if not texts:
            return []

        # fastembed returns a generator of np.ndarray (one per text).
        # ``parallel`` is left at its default of None (1 worker) which
        # is the safe choice on a 512 MB instance — parallel workers
        # would each need their own ONNX Runtime session.
        embeddings = list(
            self.model.embed(texts, batch_size=self.batch_size)
        )
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Create embedding for a query string.

        Args:
            query: Query text to embed.

        Returns:
            1-D numpy array embedding. The previous sentence-transformers
            implementation returned either a 1-D array or a 0-D scalar
            depending on input shape; we always return a 1-D array here
            so ``VectorStore.search`` (which does ``.reshape(1, -1)``)
            continues to work without surprises.
        """
        # ``embed`` accepts an iterable, so we wrap the single string
        # in a list. This avoids the scalar-vs-1D ambiguity of the old
        # ``model.encode(query)`` call.
        embeddings = list(self.model.embed([query], batch_size=1))
        # ``embeddings[0]`` is shape (dim,) — exactly what we want.
        return np.asarray(embeddings[0], dtype="float32")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.MODEL_DIMENSIONS.get(self.model_name, 384)

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(dot_product / (norm_v1 * norm_v2))
