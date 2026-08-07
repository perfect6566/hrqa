"""Retriever for finding relevant document chunks."""

from typing import List, Dict, Any, Optional


class Retriever:
    """Retrieve relevant chunks from the vector store."""

    def __init__(self, vector_store, top_k: int = 5):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance
            top_k: Default number of results to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query string
            k: Number of results (uses default if not provided)
            filters: Optional metadata filters
            rerank: Whether to rerank results (placeholder for future)

        Returns:
            List of relevant chunks with scores
        """
        k = k or self.top_k

        results = self.vector_store.search(
            query=query,
            k=k,
            filter_metadata=filters
        )

        if rerank:
            results = self._rerank_results(query, results)

        return results

    def retrieve_for_policy_query(
        self,
        query: str,
        document_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks specifically for policy-related queries.

        Args:
            query: User query
            document_types: Optional list of document IDs to filter by

        Returns:
            List of relevant policy chunks
        """
        filters = None
        if document_types:
            filters = {"document_id": document_types}

        return self.retrieve(query, k=self.top_k, filters=filters)

    def retrieve_multiple_documents(
        self,
        query: str,
        document_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks from multiple specific documents.

        Args:
            query: User query
            document_ids: List of document IDs

        Returns:
            Dictionary mapping document IDs to their relevant chunks
        """
        all_results = self.retrieve(query, k=10)

        grouped = {doc_id: [] for doc_id in document_ids}

        for chunk in all_results:
            doc_id = chunk.get("document_id")
            if doc_id in grouped:
                grouped[doc_id].append(chunk)

        return grouped

    def get_context_for_query(
        self,
        query: str,
        max_chunks: int = 5,
        include_metadata: bool = True
    ) -> str:
        """
        Get formatted context string for a query.

        Args:
            query: User query
            max_chunks: Maximum number of chunks to include
            include_metadata: Whether to include source metadata

        Returns:
            Formatted context string
        """
        chunks = self.retrieve(query, k=max_chunks)

        if not chunks:
            return "No relevant policy documents found."

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            part = f"[Source {i}: {chunk.get('title', 'Unknown')}]"
            if chunk.get("heading"):
                part += f" - {chunk['heading']}"
            part += f"\n{chunk['content']}"
            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)

    def get_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citations from retrieved chunks.

        Args:
            chunks: List of retrieved chunks

        Returns:
            List of citation objects
        """
        citations = []

        for chunk in chunks:
            # Extract filename from metadata, with fallback
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "")

            citation = {
                "document_id": chunk.get("document_id"),
                "title": chunk.get("title"),
                "heading": chunk.get("heading"),
                "filename": filename,
                "source": metadata.get("source", ""),
                "snippet": chunk.get("content", "")[:200] + "...",
                "score": chunk.get("score", 0),
            }
            citations.append(citation)

        return citations

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results based on additional criteria."""
        # Simple reranking: boost results with query terms in title/heading
        query_terms = set(query.lower().split())

        def boost_score(chunk: Dict[str, Any]) -> float:
            score = chunk.get("score", 0)

            title = chunk.get("title", "").lower()
            heading = chunk.get("heading", "").lower()

            for term in query_terms:
                if term in title:
                    score *= 1.2
                if term in heading:
                    score *= 1.1

            return score

        reranked = [(boost_score(r), r) for r in results]
        reranked.sort(key=lambda x: x[0], reverse=True)

        return [r for _, r in reranked]
