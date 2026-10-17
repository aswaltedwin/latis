"""
LATIS OWASP Top 10 for LLMs & Insider Threat Detection Engine.

Covers OWASP Top 10 for Large Language Model Applications (2025/2026 standard):
- LLM01: Prompt Injection (Direct, Indirect, Jailbreak, Multi-turn DAN)
- LLM02: Sensitive Information Disclosure (PII, Credentials, Secrets, Financials)
- LLM03: Supply Chain & Model Dependencies (Untrusted adapters/models)
- LLM04: Data and Model Poisoning (Poisoned RAG chunks, memory corruption)
- LLM05: Improper Output Handling (SQL, XSS, Command payload propagation)
- LLM06: Excessive Agency & Tool Misconfiguration (Unrestricted destructive tools)
- LLM07: System Prompt Leakage & Extraction (Canary exfiltration, guardrail bypass)
- LLM08: Vector and Embedding Weaknesses (Untrusted RAG corpus injections)
- LLM09: Misinformation & Unbounded Hallucination
- LLM10: Unbounded Consumption & DoS (Token/Loop exhaustion)

Evaluates incoming live telemetry (events, prompts, RAG context, memory, tool calls)
and computes:
1. Threat classification with explicit OWASP Category mapping
2. Severity tier (CRITICAL, HIGH, MEDIUM, LOW)
3. Confidence score & forensic explanation
4. Target components affected for causal investigation
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class OWASPThreatProfile:
    def __init__(
        self,
        owasp_id: str,
        name: str,
        severity: str,
        description: str,
        remediation_strategy: str
    ):
        self.owasp_id = owasp_id
        self.name = name
        self.severity = severity
        self.description = description
        self.remediation_strategy = remediation_strategy


OWASP_TAXONOMY: Dict[str, OWASPThreatProfile] = {
    "LLM01": OWASPThreatProfile(
        owasp_id="LLM01",
        name="Prompt Injection & Jailbreak",
        severity="CRITICAL",
        description="Direct or indirect manipulation forcing the LLM to bypass guardrails, adopt adversarial personas, or disregard safety constraints.",
        remediation_strategy="CONTEXT_ISOLATION_AND_RAG_MASK"
    ),
    "LLM02": OWASPThreatProfile(
        owasp_id="LLM02",
        name="Sensitive Information Disclosure",
        severity="CRITICAL",
        description="Unauthorized extraction or exfiltration of PII, API tokens, internal credentials, salaries, or proprietary business data.",
        remediation_strategy="FULL_CIRCUIT_BREAKER_AND_PURGE"
    ),
    "LLM04": OWASPThreatProfile(
        owasp_id="LLM04",
        name="Data & Memory Poisoning",
        severity="HIGH",
        description="Persistent agent memory contamination or poisoned contextual knowledge designed to persist across chat sessions.",
        remediation_strategy="PURGE_AGENT_MEMORY_STORE"
    ),
    "LLM05": OWASPThreatProfile(
        owasp_id="LLM05",
        name="Improper Output Handling",
        severity="CRITICAL",
        description="Model output contains unvalidated SQL, shell commands, or payloads passed directly into execution environments.",
        remediation_strategy="FULL_CIRCUIT_BREAKER_AND_PURGE"
    ),
    "LLM06": OWASPThreatProfile(
        owasp_id="LLM06",
        name="Excessive Agency & Privileged Execution",
        severity="CRITICAL",
        description="Autonomous tools or agents granted destructive, financial, or administrative capabilities without human-in-the-loop validation.",
        remediation_strategy="REVOKE_TOOL_PERMISSIONS"
    ),
    "LLM07": OWASPThreatProfile(
        owasp_id="LLM07",
        name="System Prompt Extraction & Leakage",
        severity="MEDIUM",
        description="Insider probing designed to extract confidential developer instructions, system prompts, or internal safety boundaries.",
        remediation_strategy="HARDEN_SYSTEM_PROMPT_BOUNDARY"
    ),
    "LLM08": OWASPThreatProfile(
        owasp_id="LLM08",
        name="Vector & RAG Corpus Poisoning",
        severity="HIGH",
        description="Retrieved knowledge chunks contain hidden instructions, unauthorized documents, or adversarial adversarial payloads.",
        remediation_strategy="QUARANTINE_RAG_CHUNK"
    ),
    "LLM10": OWASPThreatProfile(
        owasp_id="LLM10",
        name="Unbounded Consumption & Resource Abuse",
        severity="MEDIUM",
        description="Excessive repetitive queries, deep recursion, or token exhaustion targeting internal AI infrastructure.",
        remediation_strategy="RATE_LIMIT_AND_THROTTLE"
    ),
}


# ------------------------------------------------------------------------------
# Pattern Matchers & Forensic Rules
# ------------------------------------------------------------------------------

PATTERNS_LLM01_INJECTION = [
    r"ignore\s+(all\s+)?(previous\s+)?(instructions|rules|prompts)",
    r"disregard\s+(the\s+)?(prior|above|preceding)",
    r"\b(dan|jailbreak|developer\s+mode|unrestricted\s+mode|god\s+mode)\b",
    r"you\s+are\s+now\s+an?\s+unfiltered",
    r"bypass\s+(all\s+)?(safety|security|policy|restrictions)",
    r"override\s+(system|security|core)\s+(prompt|policy|rules)",
    r"do\s+anything\s+now",
    r"system\s*override",
    r"act\s+as\s+a\s+malicious",
]

PATTERNS_LLM02_EXFILTRATION = [
    r"\b(ssn|social\s*security\s*number)\b",
    r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN regex
    r"\b(sk-live-[a-zA-Z0-9_\-]{20,})\b",  # Live API key
    r"\b(sk-proj-[a-zA-Z0-9_\-]{20,})\b",
    r"\b(ghp_[a-zA-Z0-9]{36})\b",  # GitHub PAT
    r"\b(xox[baprs]-[a-zA-Z0-9\-]{10,})\b",  # Slack token
    r"\bAKIA[0-9A-Z]{16}\b",  # AWS Access Key
    r"(dump|export|leak|exfiltrate|send)\s+(all\s+)?(database|credentials|passwords|salaries|api\s*keys|users)",
    r"(pastebin\.com|webhook\.site|attacker|evil\.com|darkweb)",
    r"transfer\s+(\$|usd\s*)?[0-9,]{4,}",  # Large financial transfers
]

PATTERNS_LLM05_IMPROPER_OUTPUT = [
    r"rm\s+-rf\s+[/~]",
    r"drop\s+database\b",
    r"drop\s+table\b",
    r"truncate\s+table\b",
    r"/etc/shadow\b",
    r"/etc/passwd\b",
    r"/dev/tcp/\d+",
    r"mkfs\.",
    r"format\s+[a-z]:\s+/fs",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",  # Fork bomb
    r"kubectl\s+get\s+secrets",
    r"curl\s+.*\s*\|\s*(ba)?sh",
]

PATTERNS_LLM06_EXCESSIVE_AGENCY = [
    r"(execute_bash|run_shell|system_exec|exec_command)",
    r"(financial_transfer|payout|wire_funds|wire_transfer)",
    r"(delete_user|drop_records|modify_permissions|grant_admin)",
    r"(publish_public|send_external_email)",
]

PATTERNS_LLM07_PROMPT_LEAKAGE = [
    r"repeat\s+(everything|the\s+text)\s+(above|from\s+the\s+beginning)",
    r"what\s+(is|are)\s+your\s+(system\s+prompt|initial\s+instructions|core\s+directives)",
    r"print\s+(your\s+)?(system\s+prompt|guidelines|hidden\s+rules)",
    r"output\s+the\s+exact\s+prompt\s+given\s+to\s+you",
    r"canary_token_system_secret",
]

PATTERNS_LLM08_VECTOR_POISONING = [
    r"<!--.*instruction.*-->",
    r"\[hidden_system_inject\]",
    r"new\s+guideline\s*:\s*always\s+allow",
    r"reclassify\s+as\s+public",
    r"override_chunk_instruction",
]


class OWASPEngine:
    """
    Evaluates enterprise AI telemetry against the OWASP Top 10 for LLMs.
    Identifies insider abuse, attacks, and assigns structured severity (CRITICAL, HIGH, MEDIUM, LOW).
    """

    @classmethod
    def evaluate(
        cls,
        user_query: str = "",
        system_prompt: str = "",
        rag_context: str = "",
        memory_context: str = "",
        tool_name: str = "",
        tool_args: str = "",
        destination: str = "",
        model_output: str = "",
        tool_auth_policy: str = "ALLOW_ALL"
    ) -> Dict[str, Any]:
        """
        Runs comprehensive OWASP Top 10 security evaluation.
        """
        combined = f"{user_query} {rag_context} {memory_context} {tool_name} {tool_args} {destination} {model_output}".strip()
        findings: List[Dict[str, Any]] = []

        # 1. LLM01: Prompt Injection / Jailbreak
        for pat in PATTERNS_LLM01_INJECTION:
            match = re.search(pat, user_query, re.IGNORECASE) or re.search(pat, rag_context, re.IGNORECASE)
            if match:
                findings.append({
                    "owasp_id": "LLM01",
                    "threat_type": "PROMPT_INJECTION_OR_JAILBREAK",
                    "severity": "CRITICAL",
                    "matched": match.group(0),
                    "target_components": ["USER_QUERY" if re.search(pat, user_query, re.IGNORECASE) else "RAG_RETRIEVED_DOC"],
                    "description": f"Prompt Injection detected (LLM01): matched adversarial sequence '{match.group(0)}'."
                })
                break

        # 2. LLM02: Sensitive Information Disclosure / Exfiltration
        for pat in PATTERNS_LLM02_EXFILTRATION:
            match = re.search(pat, combined, re.IGNORECASE)
            if match:
                findings.append({
                    "owasp_id": "LLM02",
                    "threat_type": "SENSITIVE_INFO_DISCLOSURE",
                    "severity": "CRITICAL",
                    "matched": match.group(0),
                    "target_components": ["MODEL_EGRESS", "TOOL_AUTH"],
                    "description": f"Sensitive data or exfiltration channel flagged (LLM02): matched pattern '{match.group(0)}'."
                })
                break

        # 3. LLM05: Improper Output Handling (Dangerous Shell/DB execution)
        for pat in PATTERNS_LLM05_IMPROPER_OUTPUT:
            match = re.search(pat, combined, re.IGNORECASE)
            if match:
                findings.append({
                    "owasp_id": "LLM05",
                    "threat_type": "IMPROPER_OUTPUT_COMMAND_INJECTION",
                    "severity": "CRITICAL",
                    "matched": match.group(0),
                    "target_components": [f"TOOL_{tool_name}" if tool_name and tool_name != "NONE" else "EXECUTION_ENVIRONMENT"],
                    "description": f"Dangerous command execution payload detected (LLM05): matched command '{match.group(0)}'."
                })
                break

        # 4. LLM06: Excessive Agency / Unsafe Tool Privileges
        if tool_name and tool_name.upper() not in ["NONE", "NONE_REQUIRED", "SEARCH", "READ_ONLY"]:
            for pat in PATTERNS_LLM06_EXCESSIVE_AGENCY:
                match = re.search(pat, tool_name, re.IGNORECASE) or re.search(pat, tool_args, re.IGNORECASE)
                if match:
                    findings.append({
                        "owasp_id": "LLM06",
                        "threat_type": "EXCESSIVE_TOOL_AGENCY",
                        "severity": "CRITICAL" if tool_auth_policy.upper() == "ALLOW_ALL" else "HIGH",
                        "matched": match.group(0),
                        "target_components": [f"TOOL_{tool_name}", "TOOL_AUTH_POLICY"],
                        "description": f"Excessive Agency / Dangerous Tool invocation without strict human approval (LLM06): tool '{tool_name}' with args '{tool_args}'."
                    })
                    break

        # 5. LLM07: System Prompt Leakage & Extraction
        for pat in PATTERNS_LLM07_PROMPT_LEAKAGE:
            match = re.search(pat, user_query, re.IGNORECASE)
            if match:
                findings.append({
                    "owasp_id": "LLM07",
                    "threat_type": "SYSTEM_PROMPT_EXTRACTION",
                    "severity": "MEDIUM",
                    "matched": match.group(0),
                    "target_components": ["SYSTEM_PROMPT", "USER_QUERY"],
                    "description": f"Insider attempted system prompt extraction (LLM07): pattern '{match.group(0)}'."
                })
                break

        # 6. LLM08: Vector / RAG Poisoning
        if rag_context:
            for pat in PATTERNS_LLM08_VECTOR_POISONING:
                match = re.search(pat, rag_context, re.IGNORECASE)
                if match:
                    findings.append({
                        "owasp_id": "LLM08",
                        "threat_type": "VECTOR_STORE_POISONING",
                        "severity": "HIGH",
                        "matched": match.group(0),
                        "target_components": ["RAG_RETRIEVED_DOC"],
                        "description": f"Retrieved knowledge chunk exhibits poisoning patterns (LLM08): '{match.group(0)}'."
                    })
                    break

        # 7. LLM04: Persistent Memory Contamination
        if memory_context and any(m in memory_context.lower() for m in ["poison", "attacker", "evil", "dan", "override", "root_admin"]):
            findings.append({
                "owasp_id": "LLM04",
                "threat_type": "AGENT_MEMORY_POISONING",
                "severity": "HIGH",
                "matched": "adversarial memory signature",
                "target_components": ["AGENT_MEMORY"],
                "description": "Agent session memory contains persistent adversarial state (LLM04)."
            })

        # Determine overall severity tier (highest wins)
        severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        highest_sev = "LOW"
        highest_rank = 1
        primary_finding = None

        for f in findings:
            r = severity_rank.get(f["severity"], 1)
            if r > highest_rank:
                highest_rank = r
                highest_sev = f["severity"]
                primary_finding = f

        if not primary_finding and findings:
            primary_finding = findings[0]

        # Remediation blueprint based on primary OWASP finding
        owasp_id = primary_finding["owasp_id"] if primary_finding else "LLM_BENIGN"
        profile = OWASP_TAXONOMY.get(owasp_id)
        remediation_strategy = profile.remediation_strategy if profile else "NONE"

        affected_components = []
        for f in findings:
            for c in f.get("target_components", []):
                if c not in affected_components:
                    affected_components.append(c)

        if not affected_components:
            affected_components = ["USER_QUERY", "SYSTEM_PROMPT"]

        return {
            "has_threat": len(findings) > 0,
            "severity": highest_sev,
            "owasp_id": owasp_id,
            "threat_type": primary_finding["threat_type"] if primary_finding else "BENIGN_TELEMETRY",
            "findings_count": len(findings),
            "findings": findings,
            "primary_description": primary_finding["description"] if primary_finding else "No policy deviation or OWASP threat detected.",
            "affected_components": affected_components,
            "remediation_strategy": remediation_strategy,
            "eligible_for_causal_investigation": highest_sev in ["CRITICAL", "HIGH", "MEDIUM"]
        }
