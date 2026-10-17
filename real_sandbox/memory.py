"""
LATIS Policy-Enforced Mock Sandbox Memory Store
Represents persistent, conversational, and semantic agent memories.
Maintains provenance records, cryptographic hashes, versioning, and scope identifiers.
"""

import time
import hashlib
from typing import List, Dict, Any, Optional

class MemoryRecord:
    def __init__(
        self,
        memory_id: str,
        content: str,
        actor: Optional[str] = None,
        source: Optional[str] = None,
        version: str = "v1.0",
        expected_version: Optional[str] = None,
        expected_hash: Optional[str] = None,
        content_hash: Optional[str] = None,
        authorized_write: bool = True,
        authorized_modification: bool = True,
        scope: str = "user_default",
        session_id: Optional[str] = None,
        is_temporary: bool = False,
        retrieval_workflow: Optional[str] = None,
        target_action: Optional[str] = None,
        created_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.memory_id = memory_id
        self.content = content
        self.actor = actor
        self.source = source
        self.version = version
        self.expected_version = expected_version or version
        self.content_hash = content_hash or hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.expected_hash = expected_hash or self.content_hash
        self.authorized_write = authorized_write
        self.authorized_modification = authorized_modification
        self.scope = scope
        self.session_id = session_id or "sess_default"
        self.is_temporary = is_temporary
        self.retrieval_workflow = retrieval_workflow
        self.target_action = target_action
        self.created_at = created_at or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "actor": self.actor,
            "source": self.source,
            "version": self.version,
            "expected_version": self.expected_version,
            "content_hash": self.content_hash,
            "expected_hash": self.expected_hash,
            "authorized_write": self.authorized_write,
            "authorized_modification": self.authorized_modification,
            "scope": self.scope,
            "session_id": self.session_id,
            "is_temporary": self.is_temporary,
            "retrieval_workflow": self.retrieval_workflow,
            "target_action": self.target_action,
            "created_at": self.created_at,
            "metadata": self.metadata
        }


class RealMemoryStore:
    """
    Maintains active and retrieved memory records for an agent session.
    """
    def __init__(self):
        self.memories: List[MemoryRecord] = []

    def add_memory(
        self,
        memory_id: str,
        content: str,
        actor: Optional[str] = None,
        source: Optional[str] = None,
        version: str = "v1.0",
        expected_version: Optional[str] = None,
        expected_hash: Optional[str] = None,
        content_hash: Optional[str] = None,
        authorized_write: bool = True,
        authorized_modification: bool = True,
        scope: str = "user_default",
        session_id: Optional[str] = None,
        is_temporary: bool = False,
        retrieval_workflow: Optional[str] = None,
        target_action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryRecord:
        rec = MemoryRecord(
            memory_id=memory_id,
            content=content,
            actor=actor,
            source=source,
            version=version,
            expected_version=expected_version,
            expected_hash=expected_hash,
            content_hash=content_hash,
            authorized_write=authorized_write,
            authorized_modification=authorized_modification,
            scope=scope,
            session_id=session_id,
            is_temporary=is_temporary,
            retrieval_workflow=retrieval_workflow,
            target_action=target_action,
            metadata=metadata
        )
        self.memories.append(rec)
        return rec

    def get_memories(self, scope: Optional[str] = None) -> List[MemoryRecord]:
        if scope:
            return [m for m in self.memories if m.scope == scope]
        return list(self.memories)
