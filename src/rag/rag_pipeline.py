"""RAG pipeline that combines all components."""

from typing import List, Dict, Any, Optional
from pathlib import Path

from .document_loader import DocumentLoader
from .chunker import DocumentChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    """Complete RAG pipeline for HR policy Q&A."""

    @classmethod
    def build_index(
        cls,
        policies_dir: str = "policies",
        vector_store_path: str = "./data/vector_store",
        embedder_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> Dict[str, Any]:
        """
        Rebuild the RAG index from scratch without requiring an API key.
        Use this after adding or modifying policy documents.

        Args:
            policies_dir: Directory containing policy documents
            vector_store_path: Path to persist vector store
            embedder_model: Embedding model name
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks

        Returns:
            Indexing statistics
        """
        loader = DocumentLoader(policies_dir)
        docs = loader.load_documents()
        formats = set(d.get("metadata", {}).get("format", "") for d in docs)
        print(f"[RAG] Loaded {len(docs)} documents in formats: {formats}")

        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, seed=42)
        chunks = chunker.chunk_documents(docs)
        print(f"[RAG] Created {len(chunks)} chunks")

        embedder = Embedder(model=embedder_model)
        vs = VectorStore(
            embedder=embedder,
            store_path=vector_store_path,
            dimension=embedder.get_embedding_dimension(),
        )
        vs.clear()
        vs.add_chunks(chunks)
        vs.save()
        print(f"[RAG] Index saved with {vs.index.ntotal} vectors")
        return {"status": "success", "documents": len(docs), "chunks": vs.index.ntotal}

    def __init__(
        self,
        policies_dir: str = "policies",
        vector_store_path: str = "./data/vector_store",
        embedder_model: str = "all-MiniLM-L6-v2",
        generator_model: str = "deepseek-chat",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the complete RAG pipeline.

        Args:
            policies_dir: Directory containing policy documents
            vector_store_path: Path to persist vector store
            embedder_model: Embedding model name
            generator_model: Generator model name
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            top_k: Default number of retrieved chunks
            api_key: OpenAI API key
            base_url: OpenAI base URL
        """
        self.policies_dir = policies_dir
        self.vector_store_path = vector_store_path
        self.api_key = api_key

        # Initialize components with deterministic seeding
        self.document_loader = DocumentLoader(policies_dir)
        self.chunker = DocumentChunker(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            seed=42  # Fixed seed for deterministic chunking
        )
        self.embedder = Embedder(model=embedder_model)
        self.vector_store = VectorStore(
            embedder=self.embedder,
            store_path=vector_store_path,
            dimension=self.embedder.get_embedding_dimension()
        )
        self.retriever = Retriever(vector_store=self.vector_store, top_k=top_k)
        self.generator = Generator(model=generator_model, api_key=api_key, base_url=base_url)

        self._is_indexed = self.vector_store.index is not None and self.vector_store.index.ntotal > 0

    def index_documents(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Index all policy documents.

        Args:
            force_rebuild: Whether to rebuild the index even if it exists

        Returns:
            Indexing statistics
        """
        if self._is_indexed and not force_rebuild:
            return {
                "status": "already_indexed",
                "chunks": self.vector_store.get_stats()["total_chunks"],
            }

        # Load documents
        documents = self.document_loader.load_documents()

        if not documents:
            return {"status": "no_documents", "error": "No documents found"}

        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)

        # Clear existing index if rebuilding
        if force_rebuild:
            self.vector_store.clear()

        # Add chunks to vector store
        chunks_added = self.vector_store.add_chunks(chunks)

        # Persist
        self.vector_store.save()

        self._is_indexed = True

        return {
            "status": "success",
            "documents": len(documents),
            "chunks": chunks_added,
            "stats": self.chunker.get_stats(chunks),
        }

    def query(
        self,
        question: str,
        use_guardrails: bool = True,
        k: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            question: User's question
            use_guardrails: Whether to apply safety guardrails
            k: Number of chunks to retrieve
            conversation_history: Optional conversation history

        Returns:
            Response with answer, citations, and metadata
        """
        if not self._is_indexed:
            return {
                "answer": "The policy index is not ready. Please wait for indexing to complete.",
                "citations": [],
                "error": "not_indexed",
            }

        # Retrieve relevant chunks
        k = k or self.retriever.top_k
        retrieved_chunks = self.retriever.retrieve(question, k=k)

        # Generate response
        if use_guardrails:
            result = self.generator.generate_with_guardrails(
                query=question,
                context=retrieved_chunks,
                conversation_history=conversation_history
            )
        else:
            result = self.generator.generate(
                query=question,
                context=retrieved_chunks,
                conversation_history=conversation_history
            )

        # Add retrieval info
        result["retrieved_chunks"] = retrieved_chunks
        result["retrieval_scores"] = [c.get("score", 0) for c in retrieved_chunks]

        return result

    def query_with_sources(
        self,
        question: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Query and return detailed source information.

        Args:
            question: User's question
            k: Number of chunks to retrieve

        Returns:
            Response with detailed source citations
        """
        result = self.query(question, k=k)

        # Enhance citations
        enhanced_citations = []
        for citation in result.get("citations", []):
            chunk = None
            for c in result.get("retrieved_chunks", []):
                if c.get("document_id") == citation.get("document_id"):
                    chunk = c
                    break

            enhanced_citations.append({
                **citation,
                "full_content": chunk.get("content", "")[:500] if chunk else "",
                "source_file": chunk.get("metadata", {}).get("source", "") if chunk else "",
            })

        result["citations"] = enhanced_citations

        return result

    def get_policy_section(
        self,
        document_id: str,
        section_heading: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific section from a policy document.

        Args:
            document_id: Document identifier
            section_heading: Optional section heading to find

        Returns:
            Document section if found
        """
        chunks = self.vector_store.get_chunks_by_document(document_id)

        if not chunks:
            return None

        if not section_heading:
            # Return first chunk
            return chunks[0]

        # Find chunk with matching heading
        for chunk in chunks:
            if chunk.get("heading") and section_heading.lower() in chunk["heading"].lower():
                return chunk

        return chunks[0]

    def search_policies(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant policy content.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of matching chunks
        """
        return self.retriever.retrieve(query, k=k)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if not self._is_indexed:
            return {"status": "not_indexed"}

        doc_stats = self.document_loader.get_document_count()
        vector_stats = self.vector_store.get_stats()

        return {
            "status": "indexed",
            "documents": doc_stats,
            "chunks": vector_stats["total_chunks"],
            "dimension": vector_stats["dimension"],
        }

    def reload_index(self) -> bool:
        """Reload the index from disk."""
        self.vector_store._load_existing()
        self._is_indexed = self.vector_store.index is not None and self.vector_store.index.ntotal > 0
        return self._is_indexed
