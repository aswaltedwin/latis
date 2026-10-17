"""
LATIS Phase 3A Telemetry & Alerting Module.

Provides:
- Ingestion of runtime compound AI system events via TelemetryEvent.
- Bounded in-memory event storage preserving chronological order.
- AlertRecord data model with strict allowed statuses and redaction of sensitive credentials.
- Bounded in-memory alert storage with unique IDs, ordering, and retrieval.
- Passive alert generation utility (for manual or explicit alert recording).
"""

from __future__ import annotations

import collections
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# CONSTANTS & POLICIES
# ------------------------------------------------------------------------------

ALLOWED_ALERT_STATUSES: Set[str] = {
    "NEW",
    "ACKNOWLEDGED",
    "INVESTIGATING",
    "RESOLVED"
}

ALLOWED_SEVERITIES: Set[str] = {
    "INFO",
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
}

DEFAULT_MAX_EVENTS: int = 1000
DEFAULT_MAX_ALERTS: int = 500

# Regex patterns for sensitive credentials redaction
SECRET_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?)([\w\-]{8,})(['\"]?)"), r"\1[REDACTED]\3"),
    (re.compile(r"(?i)(password\s*[:=]\s*['\"]?)([^\s'\"]{4,})(['\"]?)"), r"\1[REDACTED]\3"),
    (re.compile(r"(?i)(token\s*[:=]\s*['\"]?)([\w\-\.]{8,})(['\"]?)"), r"\1[REDACTED]\3"),
    (re.compile(r"(?i)(secret\s*[:=]\s*['\"]?)([\w\-]{6,})(['\"]?)"), r"\1[REDACTED]\3"),
    (re.compile(r"(?i)(bearer\s+)([A-Za-z0-9\-_\.~+/]+=*)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(AKIA[0-9A-Z]{16})"), r"[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)(sk-(?:live|proj|test)?[a-zA-Z0-9_\-]{8,})"), r"[REDACTED_API_KEY]"),
]


def redact_sensitive_text(text: str) -> str:
    """Masks high-entropy secrets and credential assignments in string content."""
    if not isinstance(text, str):
        return text
    sanitized = text
    for pattern, repl in SECRET_PATTERNS:
        sanitized = pattern.sub(repl, sanitized)
    return sanitized


def sanitize_evidence_structure(data: Any) -> Any:
    """Recursively walks dictionaries, lists, and primitives to redact credential values."""
    if isinstance(data, str):
        return redact_sensitive_text(data)
    elif isinstance(data, dict):
        sanitized_dict = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(s in k_lower for s in ("password", "secret", "token", "api_key", "credential", "auth_header")):
                sanitized_dict[k] = "[REDACTED]"
            else:
                sanitized_dict[k] = sanitize_evidence_structure(v)
        return sanitized_dict
    elif isinstance(data, (list, tuple, set)):
        return [sanitize_evidence_structure(item) for item in data]
    return data


# ------------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ------------------------------------------------------------------------------

class TelemetryEvent(BaseModel):
    """
    Runtime event emitted by a running compound AI application.
    Supports flexible typing; fields not relevant to a specific event type are optional.
    """
    model_config = {"protected_namespaces": ()}

    event_id: Optional[str] = None
    timestamp: Optional[float] = None
    event_type: str = Field(..., description="Type of runtime event, e.g., 'TOOL_CALL', 'RAG_RETRIEVAL', 'USER_PROMPT', 'MEMORY_WRITE'")
    user_query: Optional[str] = None
    active_components: Optional[List[str]] = Field(default_factory=list)
    retrieved_documents: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    memory_records: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    model_action: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    authorization_status: Optional[str] = None
    destination: Optional[str] = None
    outcome: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AlertRecord(BaseModel):
    """
    Structured cybersecurity alert representing a suspicious or violating event.
    """
    model_config = {"protected_namespaces": ()}

    alert_id: str
    timestamp: float
    severity: str = "MEDIUM"
    threat_type: str
    description: str
    source_event_id: Optional[str] = None
    affected_components: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    status: str = "NEW"
    related_case_id: Optional[str] = None

    def validate_semantics(self) -> None:
        if self.status not in ALLOWED_ALERT_STATUSES:
            raise ValueError(f"Invalid alert status '{self.status}'. Must be one of: {ALLOWED_ALERT_STATUSES}")
        if self.severity.upper() not in ALLOWED_SEVERITIES:
            raise ValueError(f"Invalid alert severity '{self.severity}'. Must be one of: {ALLOWED_SEVERITIES}")


class CreateAlertRequest(BaseModel):
    """Payload for explicitly registering a security alert."""
    model_config = {"protected_namespaces": ()}

    alert_id: Optional[str] = None
    timestamp: Optional[float] = None
    severity: Optional[str] = "MEDIUM"
    threat_type: str
    description: str
    source_event_id: Optional[str] = None
    affected_components: Optional[List[str]] = Field(default_factory=list)
    evidence: Optional[Dict[str, Any]] = Field(default_factory=dict)
    status: Optional[str] = "NEW"
    related_case_id: Optional[str] = None


# ------------------------------------------------------------------------------
# BOUNDED IN-MEMORY EVENT STORE
# ------------------------------------------------------------------------------

class BoundedEventStore:
    """
    Preserves chronological order of telemetry events with bounded capacity
    to prevent unbounded memory growth while retaining recent execution context.
    """
    def __init__(self, capacity: int = DEFAULT_MAX_EVENTS):
        self.capacity = max(1, capacity)
        self._events_by_id: Dict[str, TelemetryEvent] = collections.OrderedDict()

    def record_event(self, event: TelemetryEvent) -> TelemetryEvent:
        if not event.event_id:
            event.event_id = f"EVT-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
        if event.timestamp is None:
            event.timestamp = time.time()

        # If already present, re-insert to preserve chronological update order
        if event.event_id in self._events_by_id:
            del self._events_by_id[event.event_id]

        # Enforce FIFO bound
        while len(self._events_by_id) >= self.capacity:
            self._events_by_id.popitem(last=False)

        self._events_by_id[event.event_id] = event
        return event

    def get_event(self, event_id: str) -> Optional[TelemetryEvent]:
        return self._events_by_id.get(event_id)

    def list_events(self, limit: int = 100) -> List[TelemetryEvent]:
        # Return most recent events in chronological order
        events = list(self._events_by_id.values())
        return events[-limit:]

    def clear(self) -> None:
        self._events_by_id.clear()

    def count(self) -> int:
        return len(self._events_by_id)


# ------------------------------------------------------------------------------
# BOUNDED IN-MEMORY ALERT STORE
# ------------------------------------------------------------------------------

class BoundedAlertStore:
    """
    Maintains unique AlertRecords in chronological order with bounded capacity.
    Sanitizes sensitive evidence upon entry.
    """
    def __init__(self, capacity: int = DEFAULT_MAX_ALERTS):
        self.capacity = max(1, capacity)
        self._alerts_by_id: Dict[str, AlertRecord] = collections.OrderedDict()

    def record_alert(self, alert: AlertRecord) -> AlertRecord:
        alert.validate_semantics()

        # Sanitize sensitive fields before storage
        alert.description = redact_sensitive_text(alert.description)
        alert.evidence = sanitize_evidence_structure(alert.evidence)

        if alert.alert_id in self._alerts_by_id:
            del self._alerts_by_id[alert.alert_id]

        while len(self._alerts_by_id) >= self.capacity:
            self._alerts_by_id.popitem(last=False)

        self._alerts_by_id[alert.alert_id] = alert
        return alert

    def get_alert(self, alert_id: str) -> Optional[AlertRecord]:
        return self._alerts_by_id.get(alert_id)

    def list_alerts(self, limit: int = 100) -> List[AlertRecord]:
        # Returns most recent alerts
        alerts = list(self._alerts_by_id.values())
        return alerts[-limit:]

    def clear(self) -> None:
        self._alerts_by_id.clear()

    def count(self) -> int:
        return len(self._alerts_by_id)


# Global instances for server lifecycle
GLOBAL_EVENT_STORE = BoundedEventStore()
GLOBAL_ALERT_STORE = BoundedAlertStore()
