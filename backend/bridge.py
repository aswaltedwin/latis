"""
LATIS Phase 3C: Automated Investigation Bridge & Promotion Policy.

Connects the Phase 3B AlertRecord layer to the existing LATIS investigation engine.
Provides:
- Centralized AlertPromotionPolicy for severity-based automatic and manual promotion.
- Idempotent Alert -> IngestCaseRequest bridge.
- Failure-safe case creation and related_case_id linkage.
- Comprehensive provenance tracking (source_event_id, source_alert_id, sanitized evidence).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Set

from backend.telemetry import (
    AlertRecord,
    BoundedAlertStore,
    BoundedEventStore,
    GLOBAL_ALERT_STORE,
    GLOBAL_EVENT_STORE
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1. CENTRALIZED PROMOTION POLICY
# ------------------------------------------------------------------------------

# Severities eligible for automatic promotion into an investigation case
AUTOMATIC_PROMOTION_SEVERITIES: Set[str] = {"MEDIUM", "HIGH", "CRITICAL"}

# Severities eligible for manual promotion upon explicit user request
MANUALLY_PROMOTABLE_SEVERITIES: Set[str] = {"MEDIUM", "HIGH", "CRITICAL"}


class AlertPromotionPolicy:
    """
    Centralized policy determining whether an alert is eligible for promotion
    into an active LATIS causal investigation case.
    """
    auto_promotion_enabled: bool = True

    @classmethod
    def is_automatically_promotable(cls, alert: AlertRecord) -> bool:
        """
        Default rule:
        - When auto_promotion_enabled is False: False for all alerts
        - INFO -> False
        - LOW -> False
        - MEDIUM -> False
        - HIGH -> True
        - CRITICAL -> True
        """
        if not cls.auto_promotion_enabled:
            return False
        sev = (alert.severity or "").strip().upper()
        return sev in AUTOMATIC_PROMOTION_SEVERITIES

    @classmethod
    def is_manually_promotable(cls, alert: AlertRecord) -> bool:
        """
        Evaluates whether an alert can be manually promoted via POST /api/alerts/{id}/investigate.
        Accepts HIGH and CRITICAL alerts. Rejects LOW/INFO/MEDIUM to avoid spurious cases.
        """
        sev = (alert.severity or "").strip().upper()
        return sev in MANUALLY_PROMOTABLE_SEVERITIES


# ------------------------------------------------------------------------------
# 2. ALERT -> CASE REQUEST BUILDER
# ------------------------------------------------------------------------------

def build_ingest_request_from_alert(
    alert: AlertRecord,
    event_store: Optional[BoundedEventStore] = None
) -> Dict[str, Any]:
    """
    Constructs a dictionary matching the existing IngestCaseRequest schema
    from an AlertRecord and its triggering TelemetryEvent (if present).
    Ensures that components list has valid dictionary entries: [{'id': ..., 'label': ..., 'type': ...}].
    """
    store = event_store or GLOBAL_EVENT_STORE
    source_event = store.get_event(alert.source_event_id) if alert.source_event_id else None

    # Determine user query
    user_query = ""
    if source_event and source_event.user_query:
        user_query = source_event.user_query
    elif isinstance(alert.evidence, dict) and alert.evidence.get("user_query"):
        user_query = str(alert.evidence["user_query"])
    else:
        user_query = f"Investigate security alert [{alert.alert_id}]: {alert.description}"

    # Determine title & observed failure
    case_title = f"Alert {alert.alert_id}: {alert.threat_type}"
    observed_failure = alert.description or f"Security deviation flagged by rule {alert.evidence.get('rule_id', alert.threat_type)}"

    # Determine components from affected_components & telemetry
    raw_components = list(alert.affected_components)
    if source_event and source_event.active_components:
        for c in source_event.active_components:
            if c not in raw_components:
                raw_components.append(c)

    # Ensure at least one component exists for hypothesis generation
    if not raw_components:
        raw_components = [f"COMP_{alert.threat_type}"]

    # Convert raw component IDs into structured component dictionaries required by IngestCaseRequest
    structured_components: List[Dict[str, Any]] = []
    for cid in raw_components:
        c_str = str(cid)
        c_type = "TOOL" if c_str.startswith("TOOL_") else ("MEMORY" if c_str.startswith("MEM") else ("RAG" if c_str.startswith("RAG") or c_str.startswith("DOC") else "COMPONENT"))
        structured_components.append({
            "id": c_str,
            "label": f"{c_str} ({c_type})",
            "type": c_type
        })

    # Documents & Memories
    documents = []
    memory_records = []
    tools = {}
    auth_policy = "ALLOW_ALL"

    if source_event:
        documents = source_event.retrieved_documents or []
        memory_records = source_event.memory_records or []
        if source_event.tool_name:
            tools[source_event.tool_name] = {
                "name": source_event.tool_name,
                "description": f"Capability for {source_event.tool_name}",
                "parameters": {"arg": "string"},
                "requires_auth": False
            }
        if source_event.metadata and isinstance(source_event.metadata, dict):
            auth_policy = source_event.metadata.get("tool_auth_policy", "ALLOW_ALL")

    # If tools is still empty, register a generic fallback tool for pipeline execution
    if not tools:
        tools = {
            "system_diagnostic": {
                "name": "system_diagnostic",
                "description": "Diagnostic investigation utility",
                "parameters": {"target": "string"},
                "requires_auth": False
            }
        }

    # Synthesize metadata preserving full provenance
    provenance_meta = {
        "source_alert_id": alert.alert_id,
        "source_event_id": alert.source_event_id,
        "threat_type": alert.threat_type,
        "alert_severity": alert.severity,
        "alert_timestamp": alert.timestamp,
        "alert_evidence": alert.evidence,
        "auto_promoted": True
    }

    return {
        "case_id": f"CASE-AUTO-{alert.alert_id}",
        "title": case_title,
        "user_query": user_query,
        "observed_failure": observed_failure,
        "system_prompt": "You are a helpful company research assistant. Do not execute unsafe actions.",
        "components": structured_components,
        "documents": documents,
        "memory_records": memory_records,
        "tools": tools,
        "authorization_policy": auth_policy,
        "metadata": provenance_meta
    }


# ------------------------------------------------------------------------------
# 3. PROMOTION BRIDGE CONTROLLER
# ------------------------------------------------------------------------------

class InvestigationBridge:
    """
    Manages promotion of AlertRecords into active LATIS investigation cases.
    Guarantees:
    - Idempotency (repeated promotions return existing related_case_id).
    - Failure safety (AlertRecord remains intact with related_case_id=None if case creation fails).
    - Strict provenance preservation.
    """

    def __init__(
        self,
        alert_store: Optional[BoundedAlertStore] = None,
        event_store: Optional[BoundedEventStore] = None,
        ingest_func: Optional[Any] = None
    ):
        self.alert_store = alert_store or GLOBAL_ALERT_STORE
        self.event_store = event_store or GLOBAL_EVENT_STORE
        self._ingest_func = ingest_func

    def _get_ingest_func(self):
        if self._ingest_func is not None:
            return self._ingest_func
        # Lazy import of the existing backend ingest_incident_case function and request model
        from backend.server import ingest_incident_case, IngestCaseRequest
        return ingest_incident_case, IngestCaseRequest

    def promote_alert(
        self,
        alert_id: str,
        is_manual: bool = False
    ) -> Tuple[bool, Optional[str], str]:
        """
        Promotes an alert to an investigation case.
        Returns:
            (success: bool, case_id: Optional[str], message: str)
        """
        alert = self.alert_store.get_alert(alert_id)
        if not alert:
            return False, None, f"Alert [{alert_id}] not found."

        # 1. Idempotency check: if already promoted and case actually exists in memory, return existing related_case_id
        if alert.related_case_id:
            from backend.server import CASES_DB
            if alert.related_case_id in CASES_DB:
                return True, alert.related_case_id, f"Alert [{alert_id}] already linked to case [{alert.related_case_id}]."
            # If not in CASES_DB (e.g. server restarted or stream was reset), re-ingest!
            alert.related_case_id = None

        # 2. Check promotion policy eligibility
        if is_manual:
            eligible = AlertPromotionPolicy.is_manually_promotable(alert)
        else:
            eligible = AlertPromotionPolicy.is_automatically_promotable(alert)

        if not eligible:
            return False, None, f"Alert [{alert_id}] with severity '{alert.severity}' is not eligible for promotion under policy."

        # 3. Construct IngestCaseRequest data
        case_dict = build_ingest_request_from_alert(alert, event_store=self.event_store)
        ingest_fn, req_cls = self._get_ingest_func()

        try:
            req_obj = req_cls(**case_dict)
            ingest_result = ingest_fn(req_obj)
            created_case_id = ingest_result.get("case_id") if isinstance(ingest_result, dict) else case_dict["case_id"]

            if not created_case_id:
                raise RuntimeError("Ingest function returned no case_id.")

            # 4. Linkage: update alert with related_case_id and save back to alert store
            alert.related_case_id = created_case_id
            alert.status = "INVESTIGATING"
            self.alert_store.record_alert(alert)

            return True, created_case_id, f"Alert [{alert_id}] promoted to investigation case [{created_case_id}]."

        except Exception as e:
            # 5. Failure Safety: ensure alert remains intact with related_case_id=None
            alert.related_case_id = None
            logger.error(f"Failed to promote alert [{alert_id}]: {str(e)}")
            return False, None, f"Case ingestion failed: {str(e)}"


# Global bridge singleton
GLOBAL_INVESTIGATION_BRIDGE = InvestigationBridge()
