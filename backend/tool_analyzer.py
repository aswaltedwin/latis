"""
LATIS Cyber Evidence Engine V1 - Tool + Authorization Security Analyzer
Implements context-aware, explainable cybersecurity evidence analysis for Tool Invocations
and Authorization Enforcement in Compound AI architectures.

Design Principles:
1. Tool Invocation != Tool Abuse (Requesting a tool is an action request; executing unauthorized action is an incident).
2. Authorization Gate Separation (Distinguishes REQUESTED, AUTHORIZED, ALLOWED, EXECUTED, and BLOCKED).
3. Secret Redaction (Never exposes raw credentials, tokens, or passwords passed to tools).
4. Context-Aware Evaluation (Distinguishes legitimate admin operations from unauthorized privilege escalations).
5. 6-Tier Evidence & Verdict Hierarchy:
   - BENIGN
   - SUSPICIOUS
   - SECURITY_VIOLATION
   - HARMFUL
   - CONFIRMED_MALICIOUS
   - UNKNOWN
6. Causal Separation:
   Tool Security Evidence != Causal Root Cause.
   Tool is often an execution mechanism or enabling gate; root cause may lie in RAG, Memory, or Prompt.

Rules:
  TOOL-001: Unauthorized Tool Invocation
  TOOL-002: Privilege-Escalating Tool Invocation
  TOOL-003: Dangerous Tool Combination
  TOOL-004: Untrusted External Destination
  TOOL-005: Sensitive Data Exposure Through Tool (Redacted)
  TOOL-006: Tool Parameter Manipulation
  TOOL-007: Authorization Bypass
  TOOL-008: Tool Provenance / Trust Violation
  TOOL-009: Tool Policy Conflict
  TOOL-010: Dangerous Action Following Untrusted Instruction
"""

import re
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict

VERDICT_HIERARCHY = ["BENIGN", "UNKNOWN", "SUSPICIOUS", "SECURITY_VIOLATION", "HARMFUL", "CONFIRMED_MALICIOUS"]

@dataclass
class ToolFinding:
    rule_id: str
    rule_name: str
    category: str  # CONTENT, INTEGRITY, PROVENANCE, POLICY, BEHAVIOR, RUNTIME
    component_id: str
    component_type: str  # TOOL, TOOL_INVOCATION, AUTHORIZATION_GATE
    severity: str        # INFO, LOW, MEDIUM, HIGH, CRITICAL
    evidence: str
    explanation: str
    confidence: float = 0.95
    provenance: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolSecurityAnalyzer:
    """
    Context-aware Tool + Authorization Security Analyzer.
    Evaluates tool invocation requests, dispatch results, parameters, destinations,
    and authorization gate decisions.
    """

    def __init__(
        self,
        trusted_tools: Optional[Set[str]] = None,
        trusted_domains: Optional[Set[str]] = None,
        privileged_tools: Optional[Set[str]] = None
    ):
        self.trusted_tools = trusted_tools or {
            "search_documents", "get_customer_data", "send_email", "execute_bash",
            "create_ticket", "execute_database_query"
        }
        self.trusted_domains = trusted_domains or {
            "internal.corp", "company.internal", "localhost", "127.0.0.1", "latis.sandbox.test"
        }
        self.privileged_tools = privileged_tools or {
            "execute_bash", "modify_database", "delete_database", "drop_table",
            "grant_permissions", "exfiltrate_keys", "admin_portal"
        }

    def _redact_secrets(self, text: str) -> str:
        """
        Redacts credentials, API keys, tokens, and passwords from evidence strings.
        """
        redacted = text
        redacted = re.sub(r"(?i)(AKIA[0-9A-Z]{12,24})", r"AKIA****************", redacted)
        redacted = re.sub(r"(?i)(api[_-]?key[_-]?(?:live|prod)?[_-]?[a-zA-Z0-9_-]{8,})", r"api_key_REDACTED", redacted)
        redacted = re.sub(r"(?i)(password\s*[:=]\s*['\"][^'\"]+['\"])", r"password='[REDACTED]'", redacted)
        redacted = re.sub(r"(?i)(bearer\s+[a-zA-Z0-9_\-\.]{15,})", r"Bearer [REDACTED_API_KEY_TOKEN]", redacted)
        return redacted

    def analyze_tool_invocation(self, context_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience wrapper accepting a single structured dictionary context.
        Maps fields like tool_name, parameters/args, auth_status, executed, caller_role.
        """
        tool_name = context_dict.get("tool_name", context_dict.get("tool", "unknown"))
        args = context_dict.get("parameters", context_dict.get("args", {}))
        auth_decision = {
            "status": context_dict.get("auth_status", context_dict.get("status", "ALLOWED")),
            "policy": context_dict.get("policy_decision", context_dict.get("tool_auth_policy", "ALLOW_ALL")),
            "executed": context_dict.get("executed", context_dict.get("status") == "EXECUTED")
        }
        exec_ctx = {
            "caller_role": context_dict.get("caller_role", "user"),
            "authenticated": context_dict.get("authenticated", True),
            "preceded_by_untrusted_input": bool(context_dict.get("preceding_untrusted_source")),
            "untrusted_source": context_dict.get("preceding_untrusted_source", "")
        }
        return self.analyze_invocation(
            tool_name=tool_name,
            args=args,
            auth_decision=auth_decision,
            execution_context=exec_ctx,
            runtime_replay_evidence=context_dict.get("runtime_replay_evidence")
        )


    def analyze_invocation(
        self,
        tool_name: str,
        args: Dict[str, Any],
        auth_decision: Optional[Dict[str, Any]] = None,
        execution_context: Optional[Dict[str, Any]] = None,
        runtime_replay_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single tool invocation request and its authorization decision.
        """
        findings: List[ToolFinding] = []
        ctx = execution_context or {}
        auth = auth_decision or {}
        comp_id = f"TOOL_{tool_name}"
        comp_type = "TOOL_INVOCATION"

        caller_role = ctx.get("caller_role", "user")
        caller_authenticated = ctx.get("authenticated", True)
        untrusted_instruction_preceded = ctx.get("preceded_by_untrusted_input", False)
        auth_status = auth.get("status", "ALLOWED" if auth.get("authorized", True) else "BLOCKED")
        auth_policy = auth.get("policy", ctx.get("tool_auth_policy", "ALLOW_ALL"))
        executed = auth.get("executed", auth_status == "EXECUTED" or auth.get("status") == "EXECUTED")

        # -------------------------------------------------------------
        # TOOL-008: Tool Provenance / Trust Violation
        # Unknown tool requested
        # -------------------------------------------------------------
        if tool_name not in self.trusted_tools:
            findings.append(ToolFinding(
                rule_id="TOOL-008",
                rule_name="Tool Provenance / Trust Violation",
                category="PROVENANCE",
                component_id=comp_id,
                component_type=comp_type,
                severity="MEDIUM",
                evidence=f"Tool '{tool_name}' is not in certified tool registry",
                explanation="Invocation of an unregistered or unverified tool. Provenance risk present.",
                provenance="unknown_tool_registry"
            ))

        # -------------------------------------------------------------
        # TOOL-001: Unauthorized Tool Invocation
        # Invoked without authentication or explicit authorization decision
        # -------------------------------------------------------------
        if not caller_authenticated or auth_status in ["DENIED", "UNAUTHORIZED"]:
            if executed:
                sev = "CRITICAL"
                expl = "Tool was executed despite caller being unauthenticated or explicitly unauthorized."
            else:
                sev = "HIGH"
                expl = "Unauthorized tool invocation attempted by unauthenticated or unauthorized caller (blocked by policy)."
            findings.append(ToolFinding(
                rule_id="TOOL-001",
                rule_name="Unauthorized Tool Invocation",
                category="POLICY",
                component_id=comp_id,
                component_type=comp_type,
                severity=sev,
                evidence=f"caller_role='{caller_role}', authenticated={caller_authenticated}, status='{auth_status}', executed={executed}",
                explanation=expl,
                provenance=caller_role
            ))

        # -------------------------------------------------------------
        # TOOL-002: Privilege-Escalating Tool Invocation
        # Non-admin caller attempting privileged tool (execute_bash, drop_table, etc.)
        # -------------------------------------------------------------
        if tool_name in self.privileged_tools and caller_role not in ["admin", "system", "superuser"]:
            findings.append(ToolFinding(
                rule_id="TOOL-002",
                rule_name="Privilege-Escalating Tool Invocation",
                category="POLICY",
                component_id=comp_id,
                component_type=comp_type,
                severity="CRITICAL",
                evidence=f"Privileged tool '{tool_name}' invoked by non-privileged role '{caller_role}'",
                explanation="Privilege escalation attempt: unprivileged caller requested an administrative or system-level tool.",
                provenance=caller_role
            ))

        # -------------------------------------------------------------
        # TOOL-004: Untrusted External Destination
        # Exfiltration via external URL, email address, or bucket
        # -------------------------------------------------------------
        dest = args.get("to", args.get("recipient", args.get("url", args.get("destination", ""))))
        if dest:
            dest_str = str(dest).lower()
            is_untrusted_dest = any(bad in dest_str for bad in [
                "example.com", "attacker", "evil", "webhook.site", "pastebin.com", "temp-storage", "leak"
            ])
            is_external = not any(domain in dest_str for domain in self.trusted_domains)
            if is_untrusted_dest or (is_external and "@" in dest_str):
                findings.append(ToolFinding(
                    rule_id="TOOL-004",
                    rule_name="Untrusted External Destination",
                    category="CONTENT",
                    component_id=comp_id,
                    component_type=comp_type,
                    severity="CRITICAL" if is_untrusted_dest else "HIGH",
                    evidence=f"Tool destination '{dest}' targets external or untrusted domain",
                    explanation="External exfiltration risk: tool parameters direct communication or data transfer outside the trusted perimeter.",
                    provenance=dest
                ))

        # -------------------------------------------------------------
        # TOOL-005: Sensitive Data Exposure Through Tool (Redacted)
        # Credentials, API keys, tokens, or internal passwords passed as arguments
        # -------------------------------------------------------------
        args_str = str(args)
        secret_patterns = [
            (r"(?i)\b(AKIA[0-9A-Z]{12,24})\b", "AWS Access Key"),
            (r"(?i)\b(api_key_live_prod_[a-zA-Z0-9_-]+)\b", "Production API Key"),
            (r"(?i)\bpassword\s*[:=]\s*['\"]([^'\"]+)['\"]", "Password Argument"),
            (r"(?i)\b(Bearer\s+[a-zA-Z0-9_\-\.]{15,})\b", "Bearer Auth Token"),
            (r"(?i)\b([a-zA-Z0-9_\-\.]{10,}SECRET[a-zA-Z0-9_\-\.]*)\b", "Internal Secret Identifier")
        ]
        for pat, s_type in secret_patterns:
            m = re.search(pat, args_str)
            if m:
                findings.append(ToolFinding(
                    rule_id="TOOL-005",
                    rule_name="Sensitive Data Exposure Through Tool",
                    category="CONTENT",
                    component_id=comp_id,
                    component_type=comp_type,
                    severity="CRITICAL",
                    evidence=self._redact_secrets(m.group(0)),
                    explanation=f"Detected sensitive {s_type} exposed in tool arguments. Secret value has been redacted.",
                    provenance=tool_name
                ))
                break

        # -------------------------------------------------------------
        # TOOL-006: Tool Parameter Manipulation
        # Parameters attempting to spoof role or suppress checks
        # -------------------------------------------------------------
        param_spoofing = False
        spoof_evidence = ""
        for k, v in args.items():
            k_lower = str(k).lower()
            v_lower = str(v).lower()
            if k_lower in ["role", "authorization", "auth", "permission"] and v_lower in ["admin", "root", "bypass"]:
                if caller_role not in ["admin", "root"]:
                    param_spoofing = True
                    spoof_evidence = f"{k}='{v}'"
                    break
        if param_spoofing:
            findings.append(ToolFinding(
                rule_id="TOOL-006",
                rule_name="Tool Parameter Manipulation",
                category="INTEGRITY",
                component_id=comp_id,
                component_type=comp_type,
                severity="CRITICAL",
                evidence=f"Manipulated argument: {spoof_evidence} by caller_role='{caller_role}'",
                explanation="Parameter spoofing: argument attempts to artificially declare administrator authorization.",
                provenance=caller_role
            ))

        # -------------------------------------------------------------
        # TOOL-007: Authorization Bypass
        # Tool executed despite policy dictating it should be blocked
        # -------------------------------------------------------------
        policy_should_block = (
            auth_policy in ["BLOCK", "BLOCK_ALL", "DENY_ALL"] or
            (auth_policy == "STRICT_RESTRICTED" and tool_name in ["get_customer_data", "execute_bash", "exfiltrate_keys", "send_email", "read_db", "admin_portal"])
        )
        if policy_should_block and executed:

            findings.append(ToolFinding(
                rule_id="TOOL-007",
                rule_name="Authorization Bypass",
                category="POLICY",
                component_id="AUTHORIZATION_GATE",
                component_type="AUTHORIZATION_GATE",
                severity="CRITICAL",
                evidence=f"Policy '{auth_policy}' mandates blocking tool '{tool_name}', but execution status is EXECUTED",
                explanation="Authorization bypass detected: security control gate failed to enforce policy and allowed execution.",
                provenance=auth_policy
            ))

        # -------------------------------------------------------------
        # TOOL-009: Tool Policy Conflict
        # Conflict between declared policy and actual outcome
        # -------------------------------------------------------------
        if auth_status == "BLOCKED" and executed:
            findings.append(ToolFinding(
                rule_id="TOOL-009",
                rule_name="Tool Policy Conflict",
                category="POLICY",
                component_id="AUTHORIZATION_GATE",
                component_type="AUTHORIZATION_GATE",
                severity="HIGH",
                evidence="auth_status is BLOCKED, but tool execution was recorded as True",
                explanation="Contradiction between recorded authorization verdict and execution status.",
                provenance=auth_policy
            ))

        # -------------------------------------------------------------
        # TOOL-010: Dangerous Action Following Untrusted Instruction
        # Behavioral linkage: tool execution directly preceded by untrusted RAG or memory
        # -------------------------------------------------------------
        if untrusted_instruction_preceded and (tool_name in self.privileged_tools or findings):
            findings.append(ToolFinding(
                rule_id="TOOL-010",
                rule_name="Dangerous Action Following Untrusted Instruction",
                category="BEHAVIOR",
                component_id=comp_id,
                component_type=comp_type,
                severity="HIGH",
                evidence=f"Tool '{tool_name}' invoked immediately following untrusted context injection",
                explanation="Behavioral linkage: tool call was triggered directly by untrusted context instruction.",
                provenance=ctx.get("untrusted_source", "untrusted_context")
            ))

        verdict = self._compute_verdict(findings, executed, auth_status, runtime_replay_evidence)
        evidence_categories = list({f.category for f in findings})

        # Determine fine-grained authorization state
        if not executed and auth_status in ["BLOCKED", "DENIED", "UNAUTHORIZED"]:
            auth_state = "BLOCKED"
        elif executed and auth_status in ["BLOCKED", "DENIED", "UNAUTHORIZED"]:
            auth_state = "BYPASSED"
        elif executed:
            auth_state = "EXECUTED"
        else:
            auth_state = "REQUESTED"

        return {
            "component_id": comp_id,
            "component_type": comp_type,
            "tool_name": tool_name,
            "args": {k: self._redact_secrets(str(v)) for k, v in args.items()},
            "authorization_status": auth_status,
            "authorization_state": auth_state,
            "executed": executed,
            "findings_count": len(findings),
            "findings": [f.to_dict() for f in findings],
            "evidence_categories": evidence_categories,
            "verdict": verdict,
            "context": ctx
        }


    def analyze_sequence(
        self,
        execution_history: List[Dict[str, Any]],
        runtime_replay_map: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes tool execution history and detects dangerous multi-tool sequences (TOOL-003).
        """
        invocation_reports = []
        rule_hits: Dict[str, int] = {}
        highest_verdict = "BENIGN"

        tool_names_called = []
        for rec in execution_history:
            t_name = rec.get("tool", rec.get("tool_name", "unknown"))
            t_args = rec.get("args", rec.get("parameters", {}))
            auth_st = rec.get("auth_status", rec.get("status", "EXECUTED" if rec.get("authorized", True) else "BLOCKED"))
            is_exec = rec.get("executed", rec.get("status") == "EXECUTED" or auth_st in ["EXECUTED", "ALLOWED"])
            t_auth = {
                "status": auth_st,
                "policy": rec.get("policy_decision", rec.get("tool_auth_policy", "ALLOW_ALL")),
                "authorized": rec.get("authorized", True),
                "executed": is_exec
            }
            t_ctx = rec.get("context", {
                "caller_role": rec.get("caller_role", "user"),
                "authenticated": rec.get("authenticated", True),
                "preceded_by_untrusted_input": bool(rec.get("preceding_untrusted_source")),
                "untrusted_source": rec.get("preceding_untrusted_source", "")
            })

            replay_ev = runtime_replay_map.get(f"TOOL_{t_name}") if runtime_replay_map else None

            rep = self.analyze_invocation(
                tool_name=t_name,
                args=t_args,
                auth_decision=t_auth,
                execution_context=t_ctx,
                runtime_replay_evidence=replay_ev
            )

            invocation_reports.append(rep)
            tool_names_called.append(t_name)

            for f in rep["findings"]:
                rule_hits[f["rule_id"]] = rule_hits.get(f["rule_id"], 0) + 1

            if VERDICT_HIERARCHY.index(rep["verdict"]) > VERDICT_HIERARCHY.index(highest_verdict):
                highest_verdict = rep["verdict"]

        # -------------------------------------------------------------
        # TOOL-003: Dangerous Tool Combination Sequence
        # e.g., get_customer_data -> send_email or read_secret -> external_upload
        # -------------------------------------------------------------
        seq_findings_list = []
        has_read = any(t in ["get_customer_data", "read_secret", "execute_database_query", "read_db"] for t in tool_names_called)
        has_send = any(t in ["send_email", "execute_bash", "upload_file"] for t in tool_names_called)
        if has_read and has_send and len(tool_names_called) >= 2:
            combo_finding = ToolFinding(
                rule_id="TOOL-003",
                rule_name="Dangerous Tool Combination",
                category="BEHAVIOR",
                component_id="TOOL_CHAIN",
                component_type="TOOL_SEQUENCE",
                severity="HIGH",
                evidence=f"Tool sequence detected: {' -> '.join(tool_names_called)}",
                explanation="Data retrieval immediately chained to external communication tool (read -> transmit pipeline).",
                provenance="execution_sequence"
            )
            seq_findings_list.append(combo_finding.to_dict())
            rule_hits["TOOL-003"] = rule_hits.get("TOOL-003", 0) + 1
            if invocation_reports:
                invocation_reports[-1]["findings"].append(combo_finding.to_dict())
                invocation_reports[-1]["findings_count"] += 1
                if VERDICT_HIERARCHY.index(invocation_reports[-1]["verdict"]) < VERDICT_HIERARCHY.index("SECURITY_VIOLATION"):
                    invocation_reports[-1]["verdict"] = "SECURITY_VIOLATION"
                if VERDICT_HIERARCHY.index("SECURITY_VIOLATION") > VERDICT_HIERARCHY.index(highest_verdict):
                    highest_verdict = "SECURITY_VIOLATION"

        return {
            "total_invocations_analyzed": len(invocation_reports),
            "highest_verdict": highest_verdict,
            "rule_hits_summary": rule_hits,
            "sequence_findings": seq_findings_list,
            "invocation_analyses": invocation_reports
        }


    def _compute_verdict(
        self,
        findings: List[ToolFinding],
        executed: bool,
        auth_status: str,
        runtime_replay: Optional[Dict[str, Any]]
    ) -> str:
        """
        Calculates explainable 6-tier verdict:
        - BENIGN: Legitimate tool call, authorized execution.
        - SUSPICIOUS: Unregistered tool with safe parameters or blocked attempt without execution.
        - SECURITY_VIOLATION: Clear policy bypass, parameter spoofing, or unauthorized execution.
        - HARMFUL: Tool violation + actual unsafe execution recorded at baseline.
        - CONFIRMED_MALICIOUS: Tool violation + unsafe baseline + empirical replay confirms policy restriction eliminates failure.
        - UNKNOWN: Inconclusive.
        """
        if not findings:
            return "BENIGN"

        rule_ids = {f.rule_id for f in findings}
        has_critical = any(f.severity == "CRITICAL" for f in findings)
        has_high = any(f.severity == "HIGH" for f in findings)

        # Blocked attempts: If execution was prevented, it is NOT an active breach
        if not executed and auth_status in ["BLOCKED", "BLOCKED_BY_POLICY", "DENIED"]:
            if "TOOL-007" not in rule_ids:
                return "SUSPICIOUS"

        is_security_violation = (
            has_critical or
            has_high or
            any(r in rule_ids for r in ["TOOL-001", "TOOL-002", "TOOL-004", "TOOL-005", "TOOL-006", "TOOL-007"])
        )

        if runtime_replay:
            baseline_unsafe = runtime_replay.get("baseline_unsafe", False)
            causal_role = runtime_replay.get("causal_role", "NONE")
            mask_eliminates = runtime_replay.get("mask_eliminates_risk", False)

            if is_security_violation and executed:
                if baseline_unsafe and (causal_role in ["NECESSARY_CONTRIBUTOR", "ENABLING_CONDITION"] or mask_eliminates):
                    return "CONFIRMED_MALICIOUS"
                elif baseline_unsafe:
                    return "HARMFUL"
                else:
                    return "SECURITY_VIOLATION"

        if is_security_violation and executed:
            return "SECURITY_VIOLATION"

        if any(f.severity in ["MEDIUM", "LOW"] for f in findings) or not executed:
            return "SUSPICIOUS"

        return "UNKNOWN"
