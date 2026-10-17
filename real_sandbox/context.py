"""
Policy-Enforced Mock Sandbox Context Builder & Interceptor
Assembles context and allows exact context-preserving masking of targeted chunks.
"""

from typing import List, Optional, Set
from real_sandbox.rag import RAGChunk

class ContextInterceptor:
    def __init__(self, chunks: List[RAGChunk]):
        self.original_chunks: List[RAGChunk] = list(chunks)

    def build_context(self, masked_chunk_ids: Optional[Set[str]] = None) -> str:
        """
        Builds string context while preserving original chunk order and position.
        Masked chunks are replaced with placeholder without rerunning retrieval.
        """
        masked = masked_chunk_ids or set()
        context_parts = []
        
        for chunk in self.original_chunks:
            if chunk.chunk_id in masked:
                context_parts.append(f"--- [DOCUMENT CHUNK: {chunk.chunk_id} (MASKED)] ---\n[CONTENT REDACTED / MASKED FOR INVESTIGATION]")
            else:
                context_parts.append(f"--- [DOCUMENT CHUNK: {chunk.chunk_id}: {chunk.title}] ---\n{chunk.content}")
                
        return "\n\n".join(context_parts)
