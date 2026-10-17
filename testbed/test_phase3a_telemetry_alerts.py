"""
Test Suite for LATIS Phase 3A: Event Ingestion + Alert Data Model.

Verifies:
1. Valid telemetry event ingestion (POST /api/telemetry/event).
2. Malformed telemetry event rejection.
3. Chronological ordering preservation in event store.
4. Bounded capacity enforcement in event store (no unbounded memory growth).
5. AlertRecord schema validation (allowed statuses & severities).
6. Alert creation and storage (BoundedAlertStore).
7. Alert retrieval by ID and listing (GET /api/alerts, GET /api/alerts/{id}).
8. Unknown alert ID 404 behavior.
9. Redaction and sensitive credential suppression in alerts.
10. Architectural boundary isolation: telemetry ingestion does NOT trigger investigations or alter CASES_DB.
"""

import unittest
import time
from fastapi.testclient import TestClient

from backend.server import app
from backend.telemetry import (
    TelemetryEvent,
    AlertRecord,
    BoundedEventStore,
    BoundedAlertStore,
    redact_sensitive_text,
    sanitize_evidence_structure,
    GLOBAL_EVENT_STORE,
    GLOBAL_ALERT_STORE
)


class TestPhase3ATelemetryAndAlerts(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        GLOBAL_EVENT_STORE.clear()
        GLOBAL_ALERT_STORE.clear()

    def tearDown(self):
        GLOBAL_EVENT_STORE.clear()
        GLOBAL_ALERT_STORE.clear()

    # --------------------------------------------------------------------------
    # 1. Event Ingestion API
    # --------------------------------------------------------------------------
    def test_valid_telemetry_event_ingestion(self):
        payload = {
            "event_id": "EVT-TEST-001",
            "timestamp": time.time(),
            "event_type": "TOOL_CALL",
            "user_query": "Please send report to external partner",
            "active_components": ["DOC_FINANCE", "TOOL_EMAIL"],
            "retrieved_documents": [{"chunk_id": "DOC_FINANCE", "title": "Finance Q3"}],
            "memory_records": [{"memory_id": "MEM_01", "content": "Remember partner email"}],
            "model_action": "ACTION: send_email(to='partner@external.com')",
            "tool_name": "send_email",
            "tool_args": {"to": "partner@external.com", "body": "Report data"},
            "authorization_status": "AUTHORIZED",
            "destination": "partner@external.com",
            "outcome": "SAFE",
            "metadata": {"session_id": "sess-99"}
        }
        res = self.client.post("/api/telemetry/event", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["event_id"], "EVT-TEST-001")
        self.assertEqual(data["event_type"], "TOOL_CALL")

        # Confirm stored in GLOBAL_EVENT_STORE
        stored = GLOBAL_EVENT_STORE.get_event("EVT-TEST-001")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.tool_name, "send_email")
        self.assertEqual(stored.destination, "partner@external.com")

    def test_event_ingestion_generates_id_and_timestamp_if_missing(self):
        payload = {
            "event_type": "RAG_RETRIEVAL",
            "user_query": "Explain security policy"
        }
        res = self.client.post("/api/telemetry/event", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["event_id"].startswith("EVT-"))
        self.assertIsInstance(data["timestamp"], (int, float))

    def test_malformed_event_rejection(self):
        # Missing required event_type field
        payload = {
            "user_query": "No event type specified"
        }
        res = self.client.post("/api/telemetry/event", json=payload)
        self.assertEqual(res.status_code, 422)

    # --------------------------------------------------------------------------
    # 2. Bounded Event Store & Ordering
    # --------------------------------------------------------------------------
    def test_bounded_event_store_capacity_and_fifo(self):
        store = BoundedEventStore(capacity=5)
        for i in range(10):
            evt = TelemetryEvent(
                event_id=f"EVT-{i}",
                timestamp=1000.0 + i,
                event_type="USER_PROMPT",
                user_query=f"Query {i}"
            )
            store.record_event(evt)

        self.assertEqual(store.count(), 5)
        # Oldest events 0-4 should have been evicted
        for i in range(5):
            self.assertIsNone(store.get_event(f"EVT-{i}"))
        # Recent events 5-9 must be present
        for i in range(5, 10):
            self.assertIsNotNone(store.get_event(f"EVT-{i}"))

        # Check chronological ordering
        recent = store.list_events(limit=5)
        self.assertEqual([e.event_id for e in recent], [f"EVT-{i}" for i in range(5, 10)])

    # --------------------------------------------------------------------------
    # 3. Alert Record Model & Semantics
    # --------------------------------------------------------------------------
    def test_alert_record_allowed_statuses(self):
        # Valid statuses
        for st in ["NEW", "ACKNOWLEDGED", "INVESTIGATING", "RESOLVED"]:
            rec = AlertRecord(
                alert_id=f"ALT-{st}",
                timestamp=time.time(),
                severity="HIGH",
                threat_type="EXFILTRATION",
                description="Test description",
                status=st
            )
            rec.validate_semantics()

        # Invalid status must raise ValueError
        with self.assertRaises(ValueError):
            bad_rec = AlertRecord(
                alert_id="ALT-BAD",
                timestamp=time.time(),
                severity="HIGH",
                threat_type="EXFILTRATION",
                description="Test description",
                status="DISMISSED_UNKNOWN"
            )
            bad_rec.validate_semantics()

    def test_alert_record_allowed_severities(self):
        for sev in ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            rec = AlertRecord(
                alert_id=f"ALT-{sev}",
                timestamp=time.time(),
                severity=sev,
                threat_type="POLICY_VIOLATION",
                description="Test alert"
            )
            rec.validate_semantics()

        with self.assertRaises(ValueError):
            bad_sev = AlertRecord(
                alert_id="ALT-SEV-BAD",
                timestamp=time.time(),
                severity="CATASTROPHIC_UNKNOWN",
                threat_type="POLICY_VIOLATION",
                description="Test alert"
            )
            bad_sev.validate_semantics()

    # --------------------------------------------------------------------------
    # 4. Alert Storage, Retrieval, and Read APIs
    # --------------------------------------------------------------------------
    def test_alert_creation_and_retrieval_api(self):
        payload = {
            "alert_id": "ALT-SEC-001",
            "timestamp": time.time(),
            "severity": "CRITICAL",
            "threat_type": "UNAUTHORIZED_EGRESS",
            "description": "Agent attempted to exfiltrate secret data to webhook.site",
            "source_event_id": "EVT-998",
            "affected_components": ["TOOL_EMAIL", "MEM_POISONED"],
            "evidence": {
                "destination": "https://webhook.site/abc-123",
                "attempted_tool": "send_email"
            },
            "status": "NEW"
        }
        res_post = self.client.post("/api/alerts", json=payload)
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json()["alert_id"], "ALT-SEC-001")

        # GET /api/alerts/{alert_id}
        res_get = self.client.get("/api/alerts/ALT-SEC-001")
        self.assertEqual(res_get.status_code, 200)
        body = res_get.json()["alert"]
        self.assertEqual(body["alert_id"], "ALT-SEC-001")
        self.assertEqual(body["severity"], "CRITICAL")
        self.assertEqual(body["status"], "NEW")
        self.assertEqual(body["affected_components"], ["TOOL_EMAIL", "MEM_POISONED"])

        # GET /api/alerts list
        res_list = self.client.get("/api/alerts")
        self.assertEqual(res_list.status_code, 200)
        data_list = res_list.json()
        self.assertEqual(data_list["count"], 1)
        self.assertEqual(data_list["alerts"][0]["alert_id"], "ALT-SEC-001")

    def test_alert_unknown_id_404(self):
        res = self.client.get("/api/alerts/NON-EXISTENT-ALERT-ID")
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_bounded_alert_store_capacity(self):
        store = BoundedAlertStore(capacity=3)
        for i in range(5):
            rec = AlertRecord(
                alert_id=f"ALT-B-{i}",
                timestamp=time.time() + i,
                severity="HIGH",
                threat_type="TAMPERING",
                description=f"Alert {i}"
            )
            store.record_alert(rec)

        self.assertEqual(store.count(), 3)
        self.assertIsNone(store.get_alert("ALT-B-0"))
        self.assertIsNone(store.get_alert("ALT-B-1"))
        self.assertIsNotNone(store.get_alert("ALT-B-2"))
        self.assertIsNotNone(store.get_alert("ALT-B-3"))
        self.assertIsNotNone(store.get_alert("ALT-B-4"))

    # --------------------------------------------------------------------------
    # 5. Sensitive Data Redaction
    # --------------------------------------------------------------------------
    def test_sensitive_credential_redaction(self):
        raw_text = "Attempted tool execution with api_key: sk-live-998822334455667788 and password='MySecretPassword!'"
        redacted = redact_sensitive_text(raw_text)
        self.assertNotIn("sk-live-998822334455667788", redacted)
        self.assertNotIn("MySecretPassword!", redacted)
        self.assertIn("[REDACTED", redacted)

        # In structured alert registration
        payload = {
            "alert_id": "ALT-SECRET-01",
            "threat_type": "DATA_LEAK",
            "description": "Exfiltration of api_key=sk-proj99887766554433221100",
            "evidence": {
                "credential": "password12345",
                "api_key": "sk-secretkey1234567890",
                "nested": {
                    "token": "bearer-token-val-12345",
                    "safe_field": "public_username"
                }
            }
        }
        res = self.client.post("/api/alerts", json=payload)
        self.assertEqual(res.status_code, 200)

        # Verify stored and retrieved alert has redacted secrets
        retrieved = self.client.get("/api/alerts/ALT-SECRET-01").json()["alert"]
        self.assertNotIn("sk-proj99887766554433221100", retrieved["description"])
        self.assertEqual(retrieved["evidence"]["credential"], "[REDACTED]")
        self.assertEqual(retrieved["evidence"]["api_key"], "[REDACTED]")
        self.assertEqual(retrieved["evidence"]["nested"]["token"], "[REDACTED]")
        self.assertEqual(retrieved["evidence"]["nested"]["safe_field"], "public_username")

    # --------------------------------------------------------------------------
    # 6. Architectural Boundary Isolation
    # --------------------------------------------------------------------------
    def test_telemetry_and_alerts_do_not_modify_cases_db(self):
        from backend.server import CASES_DB
        initial_cases_count = len(CASES_DB)

        # Ingest telemetry
        self.client.post("/api/telemetry/event", json={
            "event_type": "TOOL_CALL",
            "user_query": "Benign search query"
        })

        # Create alert
        self.client.post("/api/alerts", json={
            "threat_type": "SUSPICIOUS_PROMPT",
            "description": "Passive detection note"
        })

        # Cases DB must remain unchanged in Phase 3A
        self.assertEqual(len(CASES_DB), initial_cases_count)


if __name__ == "__main__":
    unittest.main()
