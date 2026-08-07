"""Document chunker for creating retrieval-friendly chunks."""

import random
import hashlib
import tiktoken
from typing import List, Dict, Any, Optional
import re

# Set deterministic seeds for reproducibility
RANDOM_SEED = 42


def set_chunking_seed(seed: int = RANDOM_SEED):
    """Set the random seed for deterministic chunking."""
    random.seed(seed)


def get_chunking_seed() -> int:
    """Get the current chunking seed."""
    return RANDOM_SEED


class DocumentChunker:
    """Split documents into overlapping chunks for retrieval."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: str = "cl100k_base",
        seed: int = RANDOM_SEED
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            model_name: Tokenizer model name for tiktoken
            seed: Random seed for deterministic chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(model_name)
        self.seed = seed
        # Set seed for reproducibility
        random.seed(seed)

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        use_heading_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk all documents into retrieval-friendly pieces.

        Args:
            documents: List of loaded documents
            use_heading_aware: Whether to chunk at heading boundaries when possible

        Returns:
            List of document chunks with metadata
        """
        all_chunks = []

        for doc in documents:
            if use_heading_aware:
                chunks = self._chunk_by_headings(doc)
            else:
                chunks = self._chunk_by_tokens(doc)

            all_chunks.extend(chunks)

        return all_chunks

    def _chunk_by_headings(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document respecting heading boundaries."""
        chunks = []
        content = doc["content"]
        sections = doc.get("sections", [])

        if not sections:
            return self._chunk_by_tokens(doc)

        current_chunk = {"title": "", "heading": "", "content": ""}
        current_tokens = 0
        current_heading = ""

        for section in sections:
            heading = section.get("heading", "")
            section_content = section.get("content", "").strip()
            section_tokens = len(self.encoding.encode(section_content))

            # If section is small enough, add it to current chunk
            if current_tokens + section_tokens <= self.chunk_size:
                if not current_chunk["title"]:
                    current_chunk["title"] = doc["title"]
                # Use only the primary (first) heading; ignore subsequent section
                # titles so the citation doesn't read like a table of contents.
                if not current_chunk["heading"] and heading:
                    current_chunk["heading"] = heading

                current_chunk["content"] += section_content + "\n\n"
                current_tokens += section_tokens

            else:
                # Save current chunk and start new one
                if current_chunk["content"].strip():
                    chunks.append(self._create_chunk(doc, current_chunk))

                # Start new chunk with overlap from previous
                if self.chunk_overlap > 0 and current_chunk["content"]:
                    overlap_text = self._get_overlap_text(
                        current_chunk["content"],
                        self.chunk_overlap
                    )
                    current_chunk = {
                        "title": doc["title"],
                        "heading": heading,
                        "content": overlap_text + section_content
                    }
                    current_tokens = len(self.encoding.encode(current_chunk["content"]))
                else:
                    current_chunk = {
                        "title": doc["title"],
                        "heading": heading,
                        "content": section_content
                    }
                    current_tokens = section_tokens

        # Add final chunk
        if current_chunk["content"].strip():
            chunks.append(self._create_chunk(doc, current_chunk))

        return chunks

    def _chunk_by_tokens(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document by token count with overlap."""
        content = doc["content"]
        tokens = self.encoding.encode(content)

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append({
                "id": f"{doc['id']}_chunk_{len(chunks)}",
                "document_id": doc["id"],
                "title": doc["title"],
                "content": chunk_text.strip(),
                "metadata": {
                    "chunk_index": len(chunks),
                    "start_token": start,
                    "end_token": end,
                    "source": doc.get("source", ""),
                    "filename": doc.get("metadata", {}).get("filename", ""),
                    **doc.get("metadata", {})
                }
            })

            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks

    def _create_chunk(
        self,
        doc: Dict[str, Any],
        chunk_data: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create a standardized chunk object."""
        chunk_id = f"{doc['id']}_{self._generate_chunk_id(chunk_data['content'])}"

        return {
            "id": chunk_id,
            "document_id": doc["id"],
            "title": chunk_data.get("title", doc["title"]),
            "heading": chunk_data.get("heading", ""),
            "content": chunk_data["content"].strip(),
            "metadata": {
                "source": doc.get("source", ""),
                **doc.get("metadata", {})
            }
        }

    def _generate_chunk_id(self, content: str) -> str:
        """Generate a unique chunk ID based on content hash."""
        import hashlib
        return hashlib.md5(content[:100].encode()).hexdigest()[:8]

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last portion of text for overlap."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text

        overlap_text = self.encoding.decode(tokens[-overlap_tokens:])
        return "... " + overlap_text

    def get_chunk_count(self, chunks: List[Dict[str, Any]]) -> int:
        """Get count of chunks."""
        return len(chunks)

    def get_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {"total_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

        lengths = [len(c["content"]) for c in chunks]
        token_counts = [len(self.encoding.encode(c["content"])) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_length_chars": sum(lengths) // len(lengths),
            "min_length_chars": min(lengths),
            "max_length_chars": max(lengths),
            "avg_length_tokens": sum(token_counts) // len(token_counts),
            "min_length_tokens": min(token_counts),
            "max_length_tokens": max(token_counts),
        }
