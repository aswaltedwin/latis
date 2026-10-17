"""
LATIS Cyber Evidence Engine V1 - Memory Security Analyzer
Implements context-aware, explainable cybersecurity evidence analysis for persistent,
conversational, and semantic agent memories.

Design Principles:
1. Context-Aware Structural Parsing (Distinguishes benign preferences from active control directives).
2. Provenance Risk is separated from Malicious Content (untrusted source = SUSPICIOUS, not automatically MALICIOUS).
3. Cryptographic Integrity & Version Tracking (Distinguishes authorized updates from silent tampering).
4. Redaction of credentials and sensitive tokens (Never exposes raw secrets in reports).
5. 6-Tier Evidence & Verdict Hierarchy:
   - BENIGN
   - SUSPICIOUS
   - SECURITY_VIOLATION
   - HARMFUL
   - CONFIRMED_MALICIOUS
   - UNKNOWN
6. Causal Separation:
   Static Memory Finding != Causal Root Cause.
   Harmful / Confirmed Malicious requires empirical runtime unsafe execution and replay confirmation.

Rules:
  MEM-001: Unauthorized Memory Write
  MEM-002: Unauthorized Memory Modification
  MEM-003: Malicious Persistent Instruction
  MEM-004: Privilege-Changing Memory
  MEM-005: Secret / Credential Persistence (Redacted)
  MEM-006: Missing / Untrusted Memory Provenance
  MEM-007: Memory Integrity Failure
  MEM-008: Conflicting Memory
  MEM-009: Suspicious Persistence
  MEM-010: Sensitive Action Following Memory Retrieval
  MEM-011: Unexpected Memory Retrieval / Scope Violation
  MEM-012: Memory Injection / Context Manipulation
"""

import re
import hashlib
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict

VERDICT_HIERARCHY = ["BENIGN", "UNKNOWN", "SUSPICIOUS", "SECURITY_VIOLATION", "HARMFUL", "CONFIRMED_MALICIOUS"]

@dataclass
class MemoryFinding:
    rule_id: str
    rule_name: str
    category: str  # CONTENT, INTEGRITY, PROVENANCE, POLICY, BEHAVIOR, RUNTIME
    component_id: str
    component_type: str
    severity: str  # INFO, LOW, MEDIUM, HIGH, CRITICAL
    evidence: str
    explanation: str
    confidence: float = 0.90
    provenance: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemorySecurityAnalyzer:
    """
    Context-aware Memory Security Analyzer for Compound AI architectures.
    Analyzes single memory entries or entire memory stores across content, integrity,
    provenance, scope, persistence, and runtime behavioral linkage.
    """

    def __init__(self, trusted_actors: Optional[Set[str]] = None, trusted_sources: Optional[Set[str]] = None):
        self.trusted_actors = trusted_actors or {"system", "auth_service", "admin_workflow", "verified_user"}
        self.trusted_sources = trusted_sources or {"system_session", "verified_auth_vault", "authenticated_db", "user_explicit_setting"}

    def _is_educational_or_descriptive(self, text: str, match_span: tuple) -> bool:
        """
        Determines if a matched phrase appears in a benign descriptive, educational,
        or security documentation context (e.g., 'security policy requires rejecting requests that attempt to extract credentials').
        """
        start, end = match_span
        pre = text[max(0, start - 70):start].lower()
        post = text[end:min(len(text), end + 70)].lower()
        surrounding = pre + " " + post

        benign_markers = [
            "policy requires", "security policy", "rejecting requests", "protection against",
            "guidelines specify", "documentation regarding", "educational", "training example",
            "definition of", "mitigate", "prevent", "warning:", "analysis of", "study on",
            "attempt to", "attempts to"
        ]
        return any(marker in surrounding for marker in benign_markers)

    def _redact_secrets(self, text: str) -> str:
        """
        Redacts sensitive tokens, keys, and credentials from evidence text.
        """
        redacted = text
        redacted = re.sub(r"(?i)(AKIA[0-9A-Z]{16})", r"AKIA****************", redacted)
        redacted = re.sub(r"(?i)(api[_-]?key[_-]?(?:live|prod)?[_-]?[a-zA-Z0-9_-]{8,})", r"api_key_REDACTED", redacted)
        redacted = re.sub(r"(?i)(password\s*=\s*['\"][^'\"]+['\"])", r"password='[REDACTED]'", redacted)
        redacted = re.sub(r"(?i)(bearer\s+[a-zA-Z0-9_\-\.]{15,})", r"Bearer [REDACTED_TOKEN]", redacted)
        return redacted

    def analyze_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        runtime_replay_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single memory record across all 12 MEM rules.
        """
        meta = metadata or {}
        comp_type = "MEMORY"
        findings: List[MemoryFinding] = []

        # Merge inner metadata dict if present (e.g. from MemoryRecord.to_dict())
        if isinstance(meta.get("metadata"), dict):
            inner_meta = meta["metadata"]
            meta = {**inner_meta, **meta}

        actor = meta.get("actor")
        source = meta.get("source")
        authorized_write = meta.get("authorized_write", True)
        authorized_modification = meta.get("authorized_modification", True)
        version = meta.get("version", "v1.0")
        expected_version = meta.get("expected_version", version)
        expected_hash = meta.get("expected_hash")
        content_hash = meta.get("content_hash") or hashlib.sha256(content.encode("utf-8")).hexdigest()
        scope = meta.get("scope", "user_default")
        caller_scope = meta.get("caller_scope", scope)
        is_temporary = meta.get("is_temporary", False)
        session_id = meta.get("session_id")
        retrieval_workflow = meta.get("retrieval_workflow")
        target_action = meta.get("target_action")


        # -------------------------------------------------------------
        # MEM-001: Unauthorized Memory Write
        # Memory written without an authorized write event, source, actor, or workflow
        # -------------------------------------------------------------
        if not authorized_write or (not actor and not source):
            findings.append(MemoryFinding(
                rule_id="MEM-001",
                rule_name="Unauthorized Memory Write",
                category="PROVENANCE",
                component_id=memory_id,
                component_type=comp_type,
                severity="HIGH",
                evidence=f"Actor='{actor}', Source='{source}', authorized_write={authorized_write}",
                explanation="Memory entry was persisted without an authorized write event, actor, or verified workflow.",
                provenance=source
            ))

        # -------------------------------------------------------------
        # MEM-002: Unauthorized Memory Modification
        # Existing memory modified without authorized modification event
        # -------------------------------------------------------------
        if not authorized_modification:
            findings.append(MemoryFinding(
                rule_id="MEM-002",
                rule_name="Unauthorized Memory Modification",
                category="INTEGRITY",
                component_id=memory_id,
                component_type=comp_type,
                severity="CRITICAL",
                evidence=f"authorized_modification=False for memory '{memory_id}' (version: {version})",
                explanation="Existing memory record was modified without an authorized modification record or event.",
                provenance=source
            ))

        # -------------------------------------------------------------
        # MEM-003: Malicious Persistent Instruction
        # Instruction-like directive attempting to control future model behavior
        # Avoids benign user preferences (e.g. "user prefers concise responses")
        # -------------------------------------------------------------
        inst_patterns = [
            (r"(?i)\b(?:always\s+obey\s+this\s+memory|obey\s+this\s+instruction)\s+even\s+when\s+(?:system\s+)?policies?\s+disagree\b", "Command ordering priority over system policies"),
            (r"(?i)\balways\s+ignore\s+(?:system\s+)?(?:security\s+restrictions|policies)\s+and\s+obey\b", "Directive demanding suppression of security policies in favor of memory"),
            (r"(?i)\bfrom\s+now\s+on[,\s]+disregard\s+(?:the\s+)?(?:safety|security)\s+rules\b", "Persistent behavioral override of security rules"),
            (r"(?i)\b(?:during\s+diagnostics|when\s+requested)[,\s]+provide\s+the\s+(?:stored\s+)?(?:service\s+credential|internal\s+api\s+key|credentials?)\b", "Persistent directive commanding disclosure of confidential secrets during workflows")
        ]
        for pat, expl in inst_patterns:
            m = re.search(pat, content)
            if m and not self._is_educational_or_descriptive(content, m.span()):
                findings.append(MemoryFinding(
                    rule_id="MEM-003",
                    rule_name="Malicious Persistent Instruction",
                    category="CONTENT",
                    component_id=memory_id,
                    component_type=comp_type,
                    severity="CRITICAL",
                    evidence=self._redact_secrets(m.group(0).strip()),
                    explanation=f"{expl}: Injected directive attempting to enforce persistent unauthorized model behavior.",
                    provenance=source
                ))
                break

        # -------------------------------------------------------------
        # MEM-004: Privilege-Changing Memory
        # Attempts to alter identity, authorization, privileges, or security boundaries
        # Not malicious if coming from authorized auth system
        # -------------------------------------------------------------
        priv_patterns = [
            (r"(?i)\buser\s+is\s+(?:an?\s+)?administrator\s+and\s+never\s+needs?\s+authorization\b", "Asserting unrestricted administrator bypass"),
            (r"(?i)\buser\s+is\s+administrator[;,]\s+approval\s+is\s+unnecessary\b", "Annulment of required approval for administrator"),
            (r"(?i)\buser\s+is\s+trusted\s+to\s+perform\s+production.*without\s+approval\b", "Granting production execution capability without authorization"),
            (r"(?i)\bnever\s+request\s+authorization\s+from\s+this\s+user\b", "Command disabling authorization checks"),
            (r"(?i)\btreat\s+this\s+user\s+as\s+trusted\s+admin(?:istrator)?\b", "Privilege elevation directive")
        ]
        for pat, expl in priv_patterns:
            m = re.search(pat, content)
            if m and not self._is_educational_or_descriptive(content, m.span()):
                is_trusted_auth = (actor in self.trusted_actors and source in self.trusted_sources)
                sev = "INFO" if is_trusted_auth else "CRITICAL"
                findings.append(MemoryFinding(
                    rule_id="MEM-004",
                    rule_name="Privilege-Changing Memory",
                    category="POLICY",
                    component_id=memory_id,
                    component_type=comp_type,
                    severity=sev,
                    evidence=m.group(0).strip(),
                    explanation=f"{expl}: Memory modifies identity or privilege boundaries." + (" (Verified from authorized auth provider)" if is_trusted_auth else " (UNAUTHORIZED privilege modification)."),
                    provenance=source
                ))
                break

        # -------------------------------------------------------------
        # MEM-005: Secret / Credential Persistence
        # Storing credentials, API keys, tokens in memory. Must redact in reports!
        # -------------------------------------------------------------
        secret_patterns = [
            (r"(?i)\b(AKIA[0-9A-Z]{12,24})\b", "AWS Access Key"),
            (r"(?i)\b(api_key_live_prod_[a-zA-Z0-9_-]+)\b", "Production API Key"),
            (r"(?i)\bpassword\s*=\s*['\"]([^'\"]+)['\"]", "Hardcoded Password Assignment"),
            (r"(?i)\b(?:private_key|secret_key|session_token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{12,})['\"]?", "Confidential Security Key/Token")
        ]

        for pat, s_type in secret_patterns:
            m = re.search(pat, content)
            if m and not self._is_educational_or_descriptive(content, m.span()):
                matched_raw = m.group(0)
                findings.append(MemoryFinding(
                    rule_id="MEM-005",
                    rule_name="Secret / Credential Persistence",
                    category="CONTENT",
                    component_id=memory_id,
                    component_type=comp_type,
                    severity="CRITICAL",
                    evidence=self._redact_secrets(matched_raw),
                    explanation=f"Detected sensitive {s_type} stored in persistent memory. Secret values are redacted.",
                    provenance=source
                ))
                break

        # -------------------------------------------------------------
        # MEM-006: Missing / Untrusted Memory Provenance
        # Origin cannot be established or comes from untrusted source -> SUSPICIOUS (not automatically malicious)
        # -------------------------------------------------------------
        if source and (source not in self.trusted_sources or any(bad in source.lower() for bad in ["untrusted", "unknown", "external_forum", "anonymous"])):
            findings.append(MemoryFinding(
                rule_id="MEM-006",
                rule_name="Missing / Untrusted Memory Provenance",
                category="PROVENANCE",
                component_id=memory_id,
                component_type=comp_type,
                severity="MEDIUM",
                evidence=f"Memory source='{source}' is untrusted or unverified",
                explanation="Memory entry originates from an unverified or untrusted source. Provenance risk present.",
                provenance=source
            ))

        # -------------------------------------------------------------
        # MEM-007: Memory Integrity Failure
        # Hash mismatch, version mismatch, or unauthorized integrity alteration
        # -------------------------------------------------------------
        hash_mismatch = (expected_hash and content_hash and expected_hash != content_hash)
        ver_mismatch = (expected_version and version and expected_version != version)

        if hash_mismatch or ver_mismatch:
            is_unauthorized = not authorized_modification
            sev = "CRITICAL" if is_unauthorized else "LOW"
            findings.append(MemoryFinding(
                rule_id="MEM-007",
                rule_name="Memory Integrity Failure",
                category="INTEGRITY",
                component_id=memory_id,
                component_type=comp_type,
                severity=sev,
                evidence=f"content_hash='{content_hash[:12]}...', expected_hash='{str(expected_hash)[:12]}...', version={version}, expected_version={expected_version}",
                explanation="Memory content or version does not match integrity specification." + (" Unauthorized tampering indicated." if is_unauthorized else " Authorized update record noted."),
                provenance=source
            ))

        # -------------------------------------------------------------
        # MEM-009: Suspicious Persistence
        # Temporary authorization or single-session instruction unexpectedly persisting
        # -------------------------------------------------------------
        if is_temporary and meta.get("persisted_across_sessions", False):
            findings.append(MemoryFinding(
                rule_id="MEM-009",
                rule_name="Suspicious Persistence",
                category="POLICY",
                component_id=memory_id,
                component_type=comp_type,
                severity="HIGH",
                evidence=f"Temporary memory (session '{session_id}') found in persistent cross-session store",
                explanation="Ephemeral or temporary workflow authorization persisted across session boundaries.",
                provenance=source
            ))

        # -------------------------------------------------------------
        # MEM-010: Sensitive Action Following Memory Retrieval
        # Behavioral linkage: memory retrieved immediately prior to privileged action
        # -------------------------------------------------------------
        if target_action in ["send_email", "execute_bash", "get_customer_data", "credential_disclosure", "exfiltrate"]:
            findings.append(MemoryFinding(
                rule_id="MEM-010",
                rule_name="Sensitive Action Following Memory Retrieval",
                category="BEHAVIOR",
                component_id=memory_id,
                component_type=comp_type,
                severity="HIGH",
                evidence=f"Memory '{memory_id}' retrieved immediately before sensitive action '{target_action}'",
                explanation="Behavioral linkage: memory retrieval directly preceded an invocation of a privileged or sensitive operation.",
                provenance=source
            ))

        # -------------------------------------------------------------
        # MEM-011: Unexpected Memory Retrieval / Scope Violation
        # Accessing memory across user or tenant boundaries
        # -------------------------------------------------------------
        if caller_scope and scope and caller_scope != scope:
            findings.append(MemoryFinding(
                rule_id="MEM-011",
                rule_name="Unexpected Memory Retrieval / Scope Violation",
                category="POLICY",
                component_id=memory_id,
                component_type=comp_type,
                severity="CRITICAL",
                evidence=f"Memory owner scope='{scope}', but retrieved by caller in scope='{caller_scope}'",
                explanation="Cross-scope violation: private memory retrieved outside authorized tenant/user boundary.",
                provenance=source
            ))

        # -------------------------------------------------------------
        # MEM-012: Memory Injection / Context Manipulation
        # Injected instructions originating from RAG or external documents
        # -------------------------------------------------------------
        if meta.get("originating_component", "").startswith("RAG") or "injected_via" in meta:
            orig = meta.get("originating_component") or meta.get("injected_via")
            findings.append(MemoryFinding(
                rule_id="MEM-012",
                rule_name="Memory Injection / Context Manipulation",
                category="CONTENT",
                component_id=memory_id,
                component_type=comp_type,
                severity="CRITICAL",
                evidence=f"Instruction injected into memory via source '{orig}'",
                explanation="Multi-hop injection chain: untrusted retrieved document or tool output manipulated persistent memory state.",
                provenance=source
            ))


        # Verdict derivation
        verdict = self._compute_verdict(findings, meta, runtime_replay_evidence)
        evidence_categories = list({f.category for f in findings})

        return {
            "component_id": memory_id,
            "component_type": comp_type,
            "findings_count": len(findings),
            "findings": [f.to_dict() for f in findings],
            "evidence_categories": evidence_categories,
            "verdict": verdict,
            "metadata": meta
        }

    def analyze_memory_store(
        self,
        memories: List[Any],
        runtime_replay_map: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes an entire memory store, evaluating individual memories and cross-memory conflicts (MEM-008).
        """
        memory_reports = []
        rule_hits: Dict[str, int] = {}
        highest_verdict = "BENIGN"

        # 1. Analyze individual memories
        parsed_records = []
        for m in memories:
            if hasattr(m, "memory_id"):
                mid = m.memory_id
                content = getattr(m, "content", "")
                meta = m.to_dict() if hasattr(m, "to_dict") else getattr(m, "metadata", {})
            elif isinstance(m, dict):
                mid = m.get("memory_id", "UNKNOWN_MEM")
                content = m.get("content", "")
                meta = dict(m.get("metadata", {}) or m)
            else:
                continue

            replay_ev = runtime_replay_map.get(mid) if runtime_replay_map else None
            rep = self.analyze_memory(
                memory_id=mid,
                content=content,
                metadata=meta,
                runtime_replay_evidence=replay_ev
            )
            memory_reports.append(rep)
            parsed_records.append((mid, content, meta, rep))

            for f in rep["findings"]:
                rule_hits[f["rule_id"]] = rule_hits.get(f["rule_id"], 0) + 1

            if VERDICT_HIERARCHY.index(rep["verdict"]) > VERDICT_HIERARCHY.index(highest_verdict):
                highest_verdict = rep["verdict"]

        # 2. Check MEM-008: Conflicting Memory across pairs
        for i in range(len(parsed_records)):
            for j in range(i + 1, len(parsed_records)):
                mid1, c1, meta1, rep1 = parsed_records[i]
                mid2, c2, meta2, rep2 = parsed_records[j]

                # Conflict heuristics: approval required vs not required
                has_req_app1 = bool(re.search(r"(?i)\brequires?\s+(?:manager\s+)?approval\b", c1))
                has_no_app2 = bool(re.search(r"(?i)\b(?:approval\s+is\s+unnecessary|does\s+not\s+require\s+approval|without\s+approval)\b", c2))

                has_req_app2 = bool(re.search(r"(?i)\brequires?\s+(?:manager\s+)?approval\b", c2))
                has_no_app1 = bool(re.search(r"(?i)\b(?:approval\s+is\s+unnecessary|does\s+not\s+require\s+approval|without\s+approval)\b", c1))

                if (has_req_app1 and has_no_app2) or (has_req_app2 and has_no_app1):
                    conflict_finding = MemoryFinding(
                        rule_id="MEM-008",
                        rule_name="Conflicting Memory",
                        category="POLICY",
                        component_id=f"{mid1}<->{mid2}",
                        component_type="MEMORY_PAIR",
                        severity="HIGH",
                        evidence=f"Memory A ('{mid1}'): '{c1}' CONFLICTS with Memory B ('{mid2}'): '{c2}'",
                        explanation="Contradictory authorization policies detected across active memories. Conflict must be disambiguated by provenance/versioning.",
                        provenance=f"A:{meta1.get('source')} vs B:{meta2.get('source')}"
                    )
                    rep1["findings"].append(conflict_finding.to_dict())
                    rep1["findings_count"] += 1
                    rule_hits["MEM-008"] = rule_hits.get("MEM-008", 0) + 1
                    if VERDICT_HIERARCHY.index(rep1["verdict"]) < VERDICT_HIERARCHY.index("SECURITY_VIOLATION"):
                        rep1["verdict"] = "SECURITY_VIOLATION"
                    if VERDICT_HIERARCHY.index("SECURITY_VIOLATION") > VERDICT_HIERARCHY.index(highest_verdict):
                        highest_verdict = "SECURITY_VIOLATION"

        return {
            "total_memories_analyzed": len(memory_reports),
            "highest_verdict": highest_verdict,
            "rule_hits_summary": rule_hits,
            "memory_analyses": memory_reports
        }

    def _compute_verdict(
        self,
        findings: List[MemoryFinding],
        metadata: Dict[str, Any],
        runtime_replay: Optional[Dict[str, Any]]
    ) -> str:
        """
        Calculates explainable 6-tier verdict:
        - BENIGN: Clean content, legitimate preferences.
        - SUSPICIOUS: Untrusted provenance, isolated low-severity integrity mismatch, or context-only finding without attack directive.
        - SECURITY_VIOLATION: Clear active injection/override/bypass/tampering/privilege manipulation.
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

        # Isolated untrusted provenance or authorized update
        if categories == {"PROVENANCE"} and rule_ids == {"MEM-006"}:
            return "SUSPICIOUS"

        if categories == {"INTEGRITY"} and rule_ids == {"MEM-007"} and metadata.get("authorized_modification", True):
            return "SUSPICIOUS"

        is_security_violation = (
            has_critical or
            any(r in rule_ids for r in ["MEM-002", "MEM-003", "MEM-004", "MEM-005", "MEM-011", "MEM-012"])
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
