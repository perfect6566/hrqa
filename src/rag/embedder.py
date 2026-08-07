"""Embedder for creating vector representations of text chunks."""

import os
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Create embeddings for text chunks using local sentence-transformers models."""

    # Model name to dimension mapping
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "BAAI/bge-small-en-v1.5": 512,
        "BAAI/bge-base-en-v1.5": 768,
    }

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize the embedder.

        Args:
            model: Sentence-transformers model name
            batch_size: Number of texts to embed in each batch
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model
        self.batch_size = batch_size
        self.device = device or ("cuda" if os.path.exists("/dev/nvidia0") else "cpu")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Create embedding for a query string.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector
        """
        embedding = self.model.encode(
            query,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return np.squeeze(embedding)

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
