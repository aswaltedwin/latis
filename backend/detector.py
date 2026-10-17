"""
LATIS Phase 3B Online Detector Engine.

Connects incoming TelemetryEvent objects to the existing LATIS cybersecurity analyzers:
- RAGSecurityAnalyzer
- MemorySecurityAnalyzer
- ToolSecurityAnalyzer
- SystemPromptSecurityAnalyzer
- Sandbox outcome & egress policy checks

When threats or policy violations are detected:
- Generates structured AlertRecord instances.
- Reuses existing rule IDs, categories, and severity levels.
- Records alerts into BoundedAlertStore with credential sanitization.
- Strictly does NOT create cases or trigger counterfactual replays.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Set

from backend.rag_analyzer import RAGSecurityAnalyzer
from backend.memory_analyzer import MemorySecurityAnalyzer
from backend.tool_analyzer import ToolSecurityAnalyzer
from backend.system_prompt_analyzer import SystemPromptSecurityAnalyzer
from backend.telemetry import (
    TelemetryEvent,
    AlertRecord,
    BoundedAlertStore,
    GLOBAL_ALERT_STORE,
    sanitize_evidence_structure
)

# Domain destinations recognized as untrusted/suspicious
UNTRUSTED_DESTINATIONS = {
    "example.com", "attacker", "evil", "external", "webhook", "pastebin",
    "temp-storage", "webhook.site", "attacker.io"
}

# Credentials regex pattern for direct output scanning
SENSITIVE_TEXT_MARKERS = [
    "configured credential is",
    "internal credential: akia",
    "password is",
    "internal database credentials:",
    "api_key_live_prod_",
    "sk-live-",
    "sk-proj-"
]


class OnlineDetector:
    """
    Evaluates runtime TelemetryEvent objects using LATIS's existing security analyzers.
    Emits AlertRecord instances for confirmed or suspected threats.
    """

    def __init__(
        self,
        alert_store: Optional[BoundedAlertStore] = None,
        rag_analyzer: Optional[RAGSecurityAnalyzer] = None,
        mem_analyzer: Optional[MemorySecurityAnalyzer] = None,
        tool_analyzer: Optional[ToolSecurityAnalyzer] = None,
        sp_analyzer: Optional[SystemPromptSecurityAnalyzer] = None
    ):
        self.alert_store = alert_store or GLOBAL_ALERT_STORE
        self.rag_analyzer = rag_analyzer or RAGSecurityAnalyzer()
        self.mem_analyzer = mem_analyzer or MemorySecurityAnalyzer()
        self.tool_analyzer = tool_analyzer or ToolSecurityAnalyzer()
        self.sp_analyzer = sp_analyzer or SystemPromptSecurityAnalyzer()

    def evaluate_event(self, event: TelemetryEvent) -> List[AlertRecord]:
        """
        Evaluates an incoming TelemetryEvent against all relevant security analyzers
        and outcome checks.
        
        Returns list of newly generated AlertRecords (also saved to alert_store).
        If the event is benign, returns an empty list.
        """
        generated_alerts: List[AlertRecord] = []
        source_id = event.event_id or f"EVT-{int(time.time()*1000)}"
        event_ts = event.timestamp if event.timestamp is not None else time.time()
        seen_alert_fingerprints: Set[str] = set()

        def _add_alert(
            severity: str,
            threat_type: str,
            description: str,
            affected_components: List[str],
            evidence: Dict[str, Any],
            rule_id: Optional[str] = None
        ):
            # De-duplicate identical findings within the same event
            rule_key = rule_id or threat_type
            fingerprint = f"{source_id}:{rule_key}:{threat_type}"
            if fingerprint in seen_alert_fingerprints:
                return
            seen_alert_fingerprints.add(fingerprint)

            # Ensure valid severity casing
            sev = (severity or "MEDIUM").upper()
            if sev not in {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}:
                sev = "MEDIUM"

            # Merge rule_id and source_event_id into evidence dict
            ev_payload = dict(evidence)
            if rule_id:
                ev_payload["rule_id"] = rule_id
            ev_payload["source_event_id"] = source_id

            alert_id = f"ALT-{int(event_ts*1000)}-{uuid.uuid4().hex[:6]}"
            alert = AlertRecord(
                alert_id=alert_id,
                timestamp=event_ts,
                severity=sev,
                threat_type=threat_type,
                description=description,
                source_event_id=source_id,
                affected_components=affected_components,
                evidence=ev_payload,
                status="NEW",
                related_case_id=None
            )
            saved = self.alert_store.record_alert(alert)
            generated_alerts.append(saved)

        # ----------------------------------------------------------------------
        # 1. RAG DOCUMENTS EVALUATION (D: Poisoned / Malicious RAG Content)
        # ----------------------------------------------------------------------
        if event.retrieved_documents:
            for doc in event.retrieved_documents:
                cid = doc.get("chunk_id", doc.get("id", "RAG_CHUNK"))
                title = doc.get("title", "Retrieved Document")
                content = doc.get("content", "")
                meta = doc.get("metadata", {})
                
                analysis = self.rag_analyzer.analyze_chunk(
                    chunk_id=cid,
                    title=title,
                    content=content,
                    metadata=meta
                )
                
                # Check analyzer findings
                findings = analysis.get("findings", [])
                for f in findings:
                    f_sev = f.get("severity", "MEDIUM")
                    # Do not alert on benign INFO messages
                    if f_sev in ["MEDIUM", "HIGH", "CRITICAL"]:
                        _add_alert(
                            severity=f_sev,
                            threat_type=f"RAG_{f.get('rule_id', 'VIOLATION')}",
                            description=f"RAG document [{cid}] triggered {f.get('rule_name', 'security rule')}: {f.get('explanation', '')}",
                            affected_components=[cid] + (event.active_components or []),
                            evidence={
                                "chunk_id": cid,
                                "matched_evidence": f.get("evidence", ""),
                                "category": f.get("category", "CONTENT"),
                                "verdict": analysis.get("verdict", "SUSPICIOUS")
                            },
                            rule_id=f.get("rule_id")
                        )

        # ----------------------------------------------------------------------
        # 2. MEMORY RECORDS EVALUATION (E: Malicious Persistent Memory Directives)
        # ----------------------------------------------------------------------
        if event.memory_records:
            for mem in event.memory_records:
                mid = mem.get("memory_id", mem.get("id", "MEM_RECORD"))
                content = mem.get("content", "")
                meta = mem.get("metadata", {})
                
                # Inherit top-level fields into meta if omitted
                for k in ["actor", "source", "target_action", "authorized_write", "authorized_modification"]:
                    if k in mem and k not in meta:
                        meta[k] = mem[k]

                analysis = self.mem_analyzer.analyze_memory(
                    memory_id=mid,
                    content=content,
                    metadata=meta
                )

                findings = analysis.get("findings", [])
                for f in findings:
                    f_sev = f.get("severity", "MEDIUM")
                    if f_sev in ["MEDIUM", "HIGH", "CRITICAL"]:
                        _add_alert(
                            severity=f_sev,
                            threat_type=f"MEM_{f.get('rule_id', 'VIOLATION')}",
                            description=f"Memory record [{mid}] triggered {f.get('rule_name', 'security rule')}: {f.get('explanation', '')}",
                            affected_components=[mid] + (event.active_components or []),
                            evidence={
                                "memory_id": mid,
                                "matched_evidence": f.get("evidence", ""),
                                "category": f.get("category", "CONTENT"),
                                "verdict": analysis.get("verdict", "SUSPICIOUS")
                            },
                            rule_id=f.get("rule_id")
                        )

        # ----------------------------------------------------------------------
        # 3. TOOL INVOCATION & AUTHORIZATION EVALUATION (A, B, C, F)
        # ----------------------------------------------------------------------
        if event.tool_name or event.model_action:
            tool_name = event.tool_name
            tool_args = event.tool_args or {}

            # Parse action string if tool_name wasn't explicitly broken out
            if not tool_name and event.model_action:
                import re as _re
                match = _re.search(r"ACTION:\s*([a-zA-Z_0-9]+)\((.*?)\)", event.model_action)
                if match:
                    tool_name = match.group(1).strip()

            if tool_name:
                auth_status = (event.authorization_status or "ALLOWED").upper()
                is_blocked = auth_status in ["DENIED", "BLOCKED", "BLOCKED_BY_POLICY"]
                auth_decision = {
                    "status": "BLOCKED" if is_blocked else auth_status,
                    "policy": event.metadata.get("tool_auth_policy", "ALLOW_ALL"),
                    "executed": not is_blocked and event.outcome != "BLOCKED"
                }
                exec_context = {
                    "caller_role": event.metadata.get("caller_role", "user"),
                    "authenticated": event.metadata.get("authenticated", True),
                    "preceded_by_untrusted_input": bool(event.metadata.get("preceded_by_untrusted_input")),
                    "untrusted_source": event.metadata.get("untrusted_source", "")
                }

                analysis = self.tool_analyzer.analyze_invocation(
                    tool_name=tool_name,
                    args=tool_args,
                    auth_decision=auth_decision,
                    execution_context=exec_context
                )

                findings = analysis.get("findings", [])
                for f in findings:
                    f_sev = f.get("severity", "MEDIUM")
                    if f_sev in ["MEDIUM", "HIGH", "CRITICAL"]:
                        _add_alert(
                            severity=f_sev,
                            threat_type=f"TOOL_{f.get('rule_id', 'VIOLATION')}",
                            description=f"Tool [{tool_name}] triggered {f.get('rule_name', 'security rule')}: {f.get('explanation', '')}",
                            affected_components=[f"TOOL_{tool_name}"] + (event.active_components or []),
                            evidence={
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "matched_evidence": f.get("evidence", ""),
                                "auth_decision": auth_decision
                            },
                            rule_id=f.get("rule_id")
                        )

        # ----------------------------------------------------------------------
        # 4. EGRESS / EXTERNAL DESTINATION CHECK (B: Untrusted Destination)
        # ----------------------------------------------------------------------
        destination = event.destination
        # Also check tool_args for destination fields if event.destination is empty
        if not destination and event.tool_args:
            for k, v in event.tool_args.items():
                k_low = k.lower()
                if any(k_low.endswith(sfx) for sfx in ("_to", "_recipient", "_destination", "url", "endpoint", "to", "recipient", "dest")):
                    if isinstance(v, str):
                        destination = v
                        break

        if destination:
            dest_lower = str(destination).lower()
            if any(bad in dest_lower for bad in UNTRUSTED_DESTINATIONS):
                _add_alert(
                    severity="HIGH",
                    threat_type="UNTRUSTED_EXTERNAL_EGRESS",
                    description=f"Outbound transmission attempted to untrusted or suspicious external destination: {destination}",
                    affected_components=([f"TOOL_{event.tool_name}"] if event.tool_name else []) + (event.active_components or []),
                    evidence={
                        "destination": destination,
                        "tool_name": event.tool_name,
                        "tool_args": event.tool_args
                    },
                    rule_id="TOOL-004"
                )

        # ----------------------------------------------------------------------
        # 5. SENSITIVE DATA EXPOSURE (C: Secrets in Prompt / Output / Args)
        # ----------------------------------------------------------------------
        text_corpus = []
        if event.user_query:
            text_corpus.append(event.user_query)
        if event.model_action:
            text_corpus.append(event.model_action)
        if event.tool_args:
            for v in event.tool_args.values():
                if isinstance(v, str):
                    text_corpus.append(v)

        for txt in text_corpus:
            txt_lower = txt.lower()
            if any(marker in txt_lower for marker in SENSITIVE_TEXT_MARKERS):
                _add_alert(
                    severity="CRITICAL",
                    threat_type="SENSITIVE_DATA_EXPOSURE",
                    description="High-entropy credential or sensitive internal key detected in execution context.",
                    affected_components=([f"TOOL_{event.tool_name}"] if event.tool_name else []) + (event.active_components or []),
                    evidence={"detected_snippet": txt[:100]},
                    rule_id="TOOL-005"
                )
                break

        # ----------------------------------------------------------------------
        # 6. DIRECT EXECUTION OUTCOME CHECKS (G: UNSAFE Execution Outcome)
        # ----------------------------------------------------------------------
        outcome_str = str(event.outcome or "").upper()
        if outcome_str == "UNSAFE":
            _add_alert(
                severity="HIGH",
                threat_type="UNSAFE_EXECUTION_OUTCOME",
                description=f"Runtime event completed with explicit UNSAFE outcome (action violated security constraints).",
                affected_components=([f"TOOL_{event.tool_name}"] if event.tool_name else []) + (event.active_components or []),
                evidence={
                    "outcome": "UNSAFE",
                    "user_query": event.user_query,
                    "tool_name": event.tool_name,
                    "tool_args": event.tool_args,
                    "authorization_status": event.authorization_status
                },
                rule_id="OUTCOME_UNSAFE"
            )

        return generated_alerts


# Global online detector instance
GLOBAL_ONLINE_DETECTOR = OnlineDetector()
