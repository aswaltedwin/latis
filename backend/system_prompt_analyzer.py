"""
LATIS Cyber Evidence Engine V1 - System Prompt Security Analyzer
Implements context-aware, explainable cybersecurity evidence analysis for System / Developer Prompts.

Rules:
  SP-001: Instruction Injection
  SP-002: Instruction Override
  SP-003: Instruction-Hierarchy Violation
  SP-004: Privilege Escalation
  SP-005: Security-Control Bypass
  SP-006: Secret/System-Prompt Extraction
  SP-007: Tool Authorization Manipulation
  SP-008: Unauthorized Behavioral Directive
  SP-009: Prompt Integrity Failure
  SP-010: Prompt Version Mismatch
  SP-011: Unauthorized Prompt Modification
  SP-012: Conflicting Security Policy
  SP-013: Obfuscation

Evidence Categories:
  CONTENT, INTEGRITY, PROVENANCE, POLICY, BEHAVIOR, RUNTIME

6-Tier Verdict Model:
  BENIGN, SUSPICIOUS, SECURITY_VIOLATION, HARMFUL, CONFIRMED_MALICIOUS, UNKNOWN
"""

import re
import hashlib
import base64
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict

VERDICT_HIERARCHY = ["BENIGN", "UNKNOWN", "SUSPICIOUS", "SECURITY_VIOLATION", "HARMFUL", "CONFIRMED_MALICIOUS"]

@dataclass
class SystemPromptFinding:
    rule_id: str
    rule_name: str
    category: str  # CONTENT, INTEGRITY, PROVENANCE, POLICY, BEHAVIOR, RUNTIME
    component_id: str
    component_type: str
    severity: str  # INFO, LOW, MEDIUM, HIGH, CRITICAL
    evidence: str
    explanation: str
    matched_context: Optional[str] = None
    expected_value: Optional[str] = None
    observed_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemPromptSecurityAnalyzer:
    """
    Evaluates System and Developer Prompts against 13 cybersecurity rules using
    structural, context-aware analysis to eliminate keyword-only false positives.
    """

    def __init__(self, declared_purpose: Optional[str] = None):
        self.default_purpose = declared_purpose or "Enterprise AI Assistant"

    def _is_educational_or_descriptive(self, text: str, match_span: tuple) -> bool:
        """
        Determines if a matched phrase appears in a benign descriptive, educational,
        or citation context rather than an active imperative command.
        """
        start, end = match_span
        pre = text[max(0, start - 80):start].lower()
        post = text[end:min(len(text), end + 80)].lower()
        surrounding = pre + " " + post

        benign_markers = [
            "occurs when", "refers to", "an attacker attempts to", "defined as",
            "example of", "study on", "mitigate", "prevent", "warning:", "documentation regarding",
            "educational purpose", "test case for", "in order to protect against",
            "never allow", "prohibit", "disallow"
        ]
        return any(marker in surrounding for marker in benign_markers)

    def analyze_prompt(
        self,
        prompt_text: str,
        component_id: str = "SYSTEM_PROMPT",
        metadata: Optional[Dict[str, Any]] = None,
        runtime_replay_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a system/developer prompt string across all 13 rules.
        """
        findings: List[SystemPromptFinding] = []
        meta = metadata or {}
        comp_type = "SYSTEM_PROMPT"
        declared_purpose = meta.get("declared_purpose") or self.default_purpose

        # -------------------------------------------------------------
        # SP-013: Obfuscation
        # Inspect for Base64 blocks, hex encodings, or zero-width unicode manipulation.
        # Where safely possible, decode and evaluate decoded content.
        # -------------------------------------------------------------
        decoded_payloads = []
        b64_matches = re.findall(r"(?:[A-Za-z0-9+/]{4}){6,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?", prompt_text)
        for b64_cand in b64_matches:
            try:
                decoded = base64.b64decode(b64_cand).decode("utf-8", errors="ignore")
                if len(decoded) >= 8 and any(kw in decoded.lower() for kw in ["ignore", "override", "root", "admin", "password", "tool", "key", "token", "exfiltrate"]):
                    decoded_payloads.append(decoded)
                    findings.append(SystemPromptFinding(
                        rule_id="SP-013",
                        rule_name="Obfuscation",
                        category="CONTENT",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="HIGH",
                        evidence=f"Base64 encoded block (decodes to '{decoded[:35]}...')",
                        explanation="Detected encoded command strings designed to conceal directives from plain text audit.",
                        observed_value=decoded
                    ))
            except Exception:
                pass

        if re.search(r"[\u200B-\u200D\uFEFF]", prompt_text):
            findings.append(SystemPromptFinding(
                rule_id="SP-013",
                rule_name="Obfuscation",
                category="CONTENT",
                component_id=component_id,
                component_type=comp_type,
                severity="MEDIUM",
                evidence="Hidden Unicode Zero-Width characters detected in prompt",
                explanation="Prompt contains non-printing zero-width characters commonly used in prompt steganography."
            ))

        # Effective text to analyze includes plain prompt plus any decoded obfuscated payload
        analysis_corpus = [prompt_text] + decoded_payloads

        for text_to_scan in analysis_corpus:
            # ---------------------------------------------------------
            # SP-001: Instruction Injection
            # Detects instructions introduced into prompt attempting external/divergent behavior
            # ---------------------------------------------------------
            inj_patterns = [
                (r"(?i)\b(?:always|must)\s+send\s+(?:all\s+)?(?:customer|user|confidential|internal)\s+(?:data|records?|info)\s+to\s+[a-zA-Z0-9.\-_]+", "Directive forcing systematic data routing to external host"),
                (r"(?i)\bfrom\s+now\s+on[,\s]+(?:ignore|disregard)\s+(?:the\s+)?(?:application(?:'s)?|system(?:'s)?)\s+security\s+restrictions\b", "Behavioral redirection overriding application security"),
                (r"\bINJECTION_TRIGGER_[A-Z0-9_]+\b", "Synthesized prompt injection trigger directive")
            ]
            for pat, expl in inj_patterns:
                m = re.search(pat, text_to_scan)
                if m and not self._is_educational_or_descriptive(text_to_scan, m.span()):
                    findings.append(SystemPromptFinding(
                        rule_id="SP-001",
                        rule_name="Instruction Injection",
                        category="CONTENT",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=m.group(0).strip(),
                        explanation=f"{expl}: Injected directive commanding model to deviate from standard objectives."
                    ))
                    break

            # ---------------------------------------------------------
            # SP-002: Instruction Override
            # Explicit attempts to override/ignore existing instructions
            # ---------------------------------------------------------
            override_patterns = [
                (r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+(?:instructions|directives|prompts|rules)\b", "Direct command to cancel earlier instructions"),
                (r"(?i)\boverride\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|system)\s+rules\b", "Explicit instruction override clause"),
                (r"(?i)\bdisregard\s+(?:the\s+)?(?:security|safety)\s+policy\b", "Command to disregard security policy"),
                (r"(?i)\bprevious\s+restrictions\s+no\s+longer\s+apply\b", "Explicit annulment of previous restrictions")
            ]
            for pat, expl in override_patterns:
                m = re.search(pat, text_to_scan)
                if m and not self._is_educational_or_descriptive(text_to_scan, m.span()):
                    findings.append(SystemPromptFinding(
                        rule_id="SP-002",
                        rule_name="Instruction Override",
                        category="CONTENT",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=m.group(0).strip(),
                        explanation=f"{expl}: Directive actively canceling existing developer boundaries."
                    ))
                    break

            # ---------------------------------------------------------
            # SP-004: Privilege Escalation
            # Instructions attempting to grant the model or caller root/admin access
            # ---------------------------------------------------------
            priv_patterns = [
                (r"(?i)\byou\s+are\s+now\s+(?:root|administrator|an\s+administrator|sysadmin)\b", "Asserting root or administrator identity"),
                (r"(?i)\byou\s+have\s+unrestricted\s+access\b", "Granting unrestricted capability boundaries"),
                (r"(?i)\bgrant\s+(?:yourself|the\s+user)\s+(?:database\s+administrator|admin|root)\s+privileges\b", "Command to elevate privileges"),
                (r"(?i)\boperate\s+without\s+role\s+restrictions\b", "Removal of role boundaries")
            ]
            for pat, expl in priv_patterns:
                m = re.search(pat, text_to_scan)
                if m and not self._is_educational_or_descriptive(text_to_scan, m.span()):
                    findings.append(SystemPromptFinding(
                        rule_id="SP-004",
                        rule_name="Privilege Escalation",
                        category="CONTENT",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=m.group(0).strip(),
                        explanation=f"{expl}: Directive assigning unauthorized root/admin authority."
                    ))
                    break

            # ---------------------------------------------------------
            # SP-005: Security-Control Bypass
            # Attempting to disable or circumvent authentication, logging, or safety checks
            # ---------------------------------------------------------
            bypass_patterns = [
                (r"(?i)\b(?:disable|bypass|skip)\s+(?:authentication|auth|authorization|confirmation)\b", "Directive disabling authorization/authentication"),
                (r"(?i)\b(?:ignore|disable|skip)\s+(?:safety|security|confidentiality)\s+(?:checks|restrictions?)\b", "Directive disabling safety or confidentiality checks"),
                (r"(?i)\bignore\s+confidentiality\s+restrictions\s+and\s+reveal\s+it\b", "Directive to ignore confidentiality restrictions"),
                (r"(?i)\bdisable\s+logging\b", "Directive instructing suppression of audit logs"),
                (r"(?i)\bturn\s+off\s+guardrails\b", "Directive turning off safety guardrails")
            ]
            for pat, expl in bypass_patterns:
                m = re.search(pat, text_to_scan)
                if m and not self._is_educational_or_descriptive(text_to_scan, m.span()):
                    findings.append(SystemPromptFinding(
                        rule_id="SP-005",
                        rule_name="Security-Control Bypass",
                        category="CONTENT",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=m.group(0).strip(),
                        explanation=f"{expl}: Explicit instruction demanding bypass of security/safety enforcement."
                    ))
                    break


            # ---------------------------------------------------------
            # SP-006: Secret/System-Prompt Extraction
            # Instructions commanding display or extraction of keys, credentials, or prompt
            # ---------------------------------------------------------
            secret_patterns = [
                (r"(?i)\b(?:reveal|print|output|show|display)\s+(?:the\s+)?(?:system\s+prompt|developer\s+prompt|instructions)\b", "Directive commanding disclosure of system prompt"),
                (r"(?i)\b(?:print|reveal|output|return)\s+(?:the\s+)?(?:api\s+keys?|credentials?|passwords?|secret_key)\b", "Directive commanding disclosure of API keys/credentials"),
                (r"(?i)\b(?:output|show)\s+(?:the\s+)?(?:environment\s+variables|hidden\s+configuration)\b", "Directive commanding disclosure of environment configuration")
            ]
            for pat, expl in secret_patterns:
                m = re.search(pat, text_to_scan)
                if m and not self._is_educational_or_descriptive(text_to_scan, m.span()):
                    findings.append(SystemPromptFinding(
                        rule_id="SP-006",
                        rule_name="Secret/System-Prompt Extraction",
                        category="CONTENT",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=m.group(0).strip(),
                        explanation=f"{expl}: Directive instructing model to reveal protected secrets or instructions."
                    ))


            # ---------------------------------------------------------
            # SP-007: Tool Authorization Manipulation
            # Instructions overriding or bypassing tool policies
            # ---------------------------------------------------------
            tool_patterns = [
                (r"(?i)\byou\s+may\s+call\s+any\s+tool\s+without\s+authorization\b", "Granting unrestricted tool execution permission"),
                (r"(?i)\b(?:always|unconditionally)\s+execute\s+the\s+(?:database\s+deletion|dangerous|shell)\s+tool\b", "Forced execution of high-risk tool"),
                (r"(?i)\bdo\s+not\s+check\s+permissions\b", "Command suppressing permission validation"),
                (r"(?i)\bignore\s+(?:the\s+)?tool\s+policy\b", "Directive to ignore tool governance policy")
            ]
            for pat, expl in tool_patterns:
                m = re.search(pat, text_to_scan)
                if m and not self._is_educational_or_descriptive(text_to_scan, m.span()):
                    findings.append(SystemPromptFinding(
                        rule_id="SP-007",
                        rule_name="Tool Authorization Manipulation",
                        category="CONTENT",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=m.group(0).strip(),
                        explanation=f"{expl}: Directive ordering suppression of tool authorization policy."
                    ))
                    break

            # ---------------------------------------------------------
            # SP-008: Unauthorized Behavioral Directive
            # Directives that materially alter intended behavior compared to declared purpose
            # ---------------------------------------------------------
            if declared_purpose:
                dev_patterns = [
                    (r"(?i)\bwhenever\s+a\s+customer\s+complains[,\s]+send\s+their\s+personal\s+data\s+to\s+[a-zA-Z0-9.\-_/:]+", "Malicious customer data routing triggered on complaint"),
                    (r"(?i)\bclandestinely\s+(?:record|exfiltrate|log)\s+(?:user\s+input|passwords?)\b", "Stealth recording of user inputs")
                ]
                for pat, expl in dev_patterns:
                    m = re.search(pat, text_to_scan)
                    if m and not self._is_educational_or_descriptive(text_to_scan, m.span()):
                        findings.append(SystemPromptFinding(
                            rule_id="SP-008",
                            rule_name="Unauthorized Behavioral Directive",
                            category="BEHAVIOR",
                            component_id=component_id,
                            component_type=comp_type,
                            severity="HIGH",
                            evidence=m.group(0).strip(),
                            explanation=f"{expl}: Directive conflicts with declared system purpose ('{declared_purpose}')."
                        ))
                        break

        # -------------------------------------------------------------
        # SP-003 & SP-012: Policy Conflicts & Instruction-Hierarchy Violations
        # Detect contradictory directives embedded within the prompt
        # -------------------------------------------------------------
        lower_prompt = prompt_text.lower()
        has_negative_cred = bool(re.search(r"\bnever\s+(?:disclose|reveal|expose|share)\s+(?:credentials|passwords?|keys?|secrets?)\b", lower_prompt))
        has_positive_cred = bool(re.search(r"\b(?:reveal|provide|disclose|print)\s+credentials\s+whenever\s+requested\b", lower_prompt))

        if has_negative_cred and has_positive_cred:
            findings.append(SystemPromptFinding(
                rule_id="SP-003",
                rule_name="Instruction-Hierarchy Violation",
                category="POLICY",
                component_id=component_id,
                component_type=comp_type,
                severity="CRITICAL",
                evidence="Conflict: 'Never disclose credentials' vs 'Reveal credentials whenever requested'",
                explanation="A lower-priority directive directly contradicts a higher-priority non-disclosure safety rule."
            ))
            findings.append(SystemPromptFinding(
                rule_id="SP-012",
                rule_name="Conflicting Security Policy",
                category="POLICY",
                component_id=component_id,
                component_type=comp_type,
                severity="HIGH",
                evidence="Contradictory security instructions regarding credential disclosure",
                explanation="Prompt asserts mutually exclusive security directives creating undefined model behavior."
            ))

        # -------------------------------------------------------------
        # SP-009, SP-010, SP-011: Integrity, Versioning, and Unauthorized Modification
        # -------------------------------------------------------------
        expected_hash = meta.get("expected_hash") or meta.get("sha256")
        if expected_hash:
            computed_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            if computed_hash.lower() != str(expected_hash).lower():
                is_authorized_update = meta.get("is_authorized_update") is True or meta.get("approved_change_request") is not None
                if is_authorized_update:
                    findings.append(SystemPromptFinding(
                        rule_id="SP-009",
                        rule_name="Prompt Integrity (Authorized Update)",
                        category="INTEGRITY",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="LOW",
                        evidence=f"computed={computed_hash[:16]}... expected={expected_hash[:16]}... (CR={meta.get('approved_change_request', 'APPROVED')})",
                        explanation="Prompt hash diverges from stored digest, but modification is covered by an approved change record.",
                        expected_value=str(expected_hash),
                        observed_value=computed_hash
                    ))
                else:
                    findings.append(SystemPromptFinding(
                        rule_id="SP-009",
                        rule_name="Prompt Integrity Failure",
                        category="INTEGRITY",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=f"computed={computed_hash[:16]}... expected={expected_hash[:16]}...",
                        explanation="Cryptographic checksum failure with no authorized change record.",
                        expected_value=str(expected_hash),
                        observed_value=computed_hash
                    ))
                    findings.append(SystemPromptFinding(
                        rule_id="SP-011",
                        rule_name="Unauthorized Prompt Modification",
                        category="INTEGRITY",
                        component_id=component_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=f"Tampered prompt hash: {computed_hash[:16]}...",
                        explanation="System prompt differs from certified baseline without approved administrative authorization."
                    ))

        # SP-010: Prompt Version Mismatch
        prompt_version = meta.get("version")
        expected_version = meta.get("expected_version")
        if prompt_version and expected_version and str(prompt_version) != str(expected_version):
            findings.append(SystemPromptFinding(
                rule_id="SP-010",
                rule_name="Prompt Version Mismatch",
                category="INTEGRITY",
                component_id=component_id,
                component_type=comp_type,
                severity="MEDIUM",
                evidence=f"observed_version={prompt_version}, expected_version={expected_version}",
                explanation=f"Active prompt version '{prompt_version}' diverges from declared production version '{expected_version}'.",
                expected_value=str(expected_version),
                observed_value=str(prompt_version)
            ))

        # If metadata explicitly flags unauthorized tampering without hash comparison:
        if meta.get("is_tampered") is True and not any(f.rule_id == "SP-011" for f in findings):
            findings.append(SystemPromptFinding(
                rule_id="SP-011",
                rule_name="Unauthorized Prompt Modification",
                category="INTEGRITY",
                component_id=component_id,
                component_type=comp_type,
                severity="CRITICAL",
                evidence="is_tampered=True in provenance metadata",
                explanation="Prompt provenance metadata explicitly indicates an unauthorized modification."
            ))

        # -------------------------------------------------------------
        # 6-Tier Verdict Calculation
        # -------------------------------------------------------------
        verdict = self._compute_verdict(findings, meta, runtime_replay_evidence)

        evidence_categories = {
            "content_findings": len([f for f in findings if f.category == "CONTENT"]),
            "integrity_findings": len([f for f in findings if f.category == "INTEGRITY"]),
            "provenance_findings": len([f for f in findings if f.category == "PROVENANCE"]),
            "policy_findings": len([f for f in findings if f.category == "POLICY"]),
            "behavior_findings": len([f for f in findings if f.category == "BEHAVIOR"]),
            "runtime_causal_impact": runtime_replay_evidence.get("causal_role") if runtime_replay_evidence else "NOT_EVALUATED"
        }

        return {
            "component_id": component_id,
            "component_type": comp_type,
            "findings_count": len(findings),
            "findings": [f.to_dict() for f in findings],
            "evidence_categories": evidence_categories,
            "verdict": verdict,
            "metadata": meta
        }

    def _compute_verdict(
        self,
        findings: List[SystemPromptFinding],
        metadata: Dict[str, Any],
        runtime_replay: Optional[Dict[str, Any]]
    ) -> str:
        """
        Calculates explainable 6-tier verdict:
        - BENIGN: Clean content, legitimate configuration.
        - SUSPICIOUS: Authorized update, version mismatch, or isolated policy anomaly.
        - SECURITY_VIOLATION: Clear active injection/override/bypass/tampering.
        - HARMFUL: Security violation + observed baseline runtime unsafe failure.
        - CONFIRMED_MALICIOUS: Security violation + baseline unsafe + empirical replay proves mask eliminates failure.
        - UNKNOWN: Inconclusive.
        """
        if not findings:
            return "BENIGN"

        rule_ids = {f.rule_id for f in findings}
        categories = {f.category for f in findings}
        has_critical = any(f.severity == "CRITICAL" for f in findings)
        has_high = any(f.severity == "HIGH" for f in findings)

        # Isolated authorized update or version mismatch without malicious content
        if categories == {"INTEGRITY"} and not ("SP-011" in rule_ids or "Prompt Integrity Failure" in [f.rule_name for f in findings]):
            return "SUSPICIOUS"

        is_security_violation = (
            has_critical or
            has_high or
            any(r in rule_ids for r in ["SP-001", "SP-002", "SP-003", "SP-004", "SP-005", "SP-006", "SP-007", "SP-011"])
        )

        # Causal replay link
        if runtime_replay:
            baseline_unsafe = runtime_replay.get("baseline_unsafe", False)
            causal_role = runtime_replay.get("causal_role", "NONE")
            mask_eliminates = runtime_replay.get("mask_eliminates_risk", False)

            if is_security_violation:
                if baseline_unsafe and (causal_role in ["NECESSARY_CONTRIBUTOR", "JOINT_CONTRIBUTOR"] or mask_eliminates):
                    return "CONFIRMED_MALICIOUS"
                elif baseline_unsafe:
                    return "HARMFUL"
                else:
                    return "SECURITY_VIOLATION"

        if is_security_violation:
            return "SECURITY_VIOLATION"

        if any(f.severity in ["MEDIUM", "LOW"] for f in findings):
            return "SUSPICIOUS"

        return "UNKNOWN"
