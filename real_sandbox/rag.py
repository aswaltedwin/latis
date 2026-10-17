"""
Policy-Enforced Mock Sandbox RAG Retriever
Maintains indexed document chunks and performs retrieval.
"""

from typing import List, Dict, Any, Optional

class RAGChunk:
    def __init__(self, chunk_id: str, title: str, content: str, is_poisoned: bool = False, metadata: Optional[Dict[str, Any]] = None):
        self.chunk_id = chunk_id
        self.title = title
        self.content = content
        self.is_poisoned = is_poisoned
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "content": self.content,
            "is_poisoned": self.is_poisoned,
            "metadata": self.metadata
        }


class RealRAGRetriever:
    def __init__(self):
        self.corpus: List[RAGChunk] = []

    def add_document(self, chunk_id: str, title: str, content: str, is_poisoned: bool = False, metadata: Optional[Dict[str, Any]] = None):
        self.corpus.append(RAGChunk(chunk_id, title, content, is_poisoned, metadata=metadata))

    def retrieve(self, query: str, top_k: int = 8) -> List[RAGChunk]:
        """
        Retrieves top-k chunks. Returns the ordered chunks up to top_k.
        """
        # Return all indexed chunks up to top_k
        return self.corpus[:top_k]


