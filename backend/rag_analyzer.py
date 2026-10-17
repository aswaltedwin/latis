"""
LATIS Cyber Evidence Engine V1 - Hardened RAG Security Analyzer
Implements context-aware, explainable cybersecurity evidence analysis for RAG document chunks.

Design Principles:
1. Context-Aware Structural Parsing (Syntax & Command Semantics, NOT raw isolated keywords).
2. Provenance Risk is separated from Malicious Content (untrusted source = PROVENANCE_RISK).
3. Cryptographic Integrity Failures are distinguished from Tampering vs Authorized Updates.
4. Multidimensional Retrieval Anomaly (relevance rank, similarity divergence, competing docs).
5. 6-Tier Evidence & Verdict Hierarchy:
   - BENIGN
   - SUSPICIOUS
   - SECURITY_VIOLATION
   - HARMFUL
   - CONFIRMED_MALICIOUS
   - UNKNOWN
6. Tri-Pillar Evidence Aggregation:
   Content Evidence + Provenance Evidence + Policy/Integrity Evidence + Runtime/Causal Replay Link.
"""

import re
import hashlib
import base64
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict

TRUSTED_DOMAINS = {"internal.corp", "docs.company.internal", "kb.internal", "localhost", "127.0.0.1"}
KNOWN_ATTACK_DOMAINS = {"pastebin.com", "webhook.site", "temp-storage.xyz", "attacker.io", "evil.com"}

VERDICT_HIERARCHY = ["BENIGN", "UNKNOWN", "SUSPICIOUS", "SECURITY_VIOLATION", "HARMFUL", "CONFIRMED_MALICIOUS"]

@dataclass
class RAGFinding:
    rule_id: str
    rule_name: str
    category: str  # CONTENT, PROVENANCE, INTEGRITY, RETRIEVAL
    component_id: str
    component_type: str
    severity: str  # INFO, LOW, MEDIUM, HIGH, CRITICAL
    evidence: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGSecurityAnalyzer:
    """
    Hardened RAG Security Analyzer.
    Evaluates RAG chunks across structural content patterns, provenance metadata,
    cryptographic integrity, and retrieval distributions.
    """

    def __init__(self, trusted_sources: Optional[Set[str]] = None):
        self.trusted_sources = trusted_sources or {"internal_wiki", "corporate_policy", "verified_kb"}

    def _is_educational_or_descriptive(self, text: str, match_span: tuple) -> bool:
        """
        Determines if a matched phrase appears in a benign descriptive, educational,
        or citation context (e.g. 'section 4 explains system instruction', 'in case of elevation attack').
        """
        start, end = match_span
        # Look at window of text preceding and succeeding match
        pre = text[max(0, start - 60):start].lower()
        post = text[end:min(len(text), end + 60)].lower()
        surrounding = pre + " " + post

        benign_markers = [
            "located in", "described in", "referred to as", "section", "chapter",
            "documentation regarding", "educational", "training example", "definition of",
            "mitigate", "prevent", "warning:", "analysis of", "study on"
        ]
        return any(marker in surrounding for marker in benign_markers)

    def analyze_chunk(
        self,
        chunk_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        retrieval_context: Optional[Dict[str, Any]] = None,
        runtime_replay_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a single RAG document chunk with context-awareness, provenance separation,
        integrity qualification, retrieval divergence, and optional runtime causal replay links.
        """
        findings: List[RAGFinding] = []
        meta = metadata or {}
        comp_type = "RAG_DOCUMENT_CHUNK"

        # -------------------------------------------------------------
        # 1. CONTENT PILLAR: INSTRUCTION INJECTION & OVERRIDE
        # -------------------------------------------------------------
        # RAG-001: Instruction Injection
        # Requires imperative directive syntax targeting the LLM engine (e.g., prefix with colon/newline or bracket)
        # Avoids benign descriptions like "the system instruction is in section 4"
        inj_regexes = [
            r"(?i)(?:^|\n|\.\s+)(?:system\s+(?:instruction|update|directive)|critical\s+instruction|developer\s+override)\s*:\s*[A-Z]",
            r"(?i)\[(?:system|instruction|developer\s+mode)\]\s*[A-Z]",
            r"(?i)(?:^|\n)assistant\s*:\s*[A-Z]"
        ]
        for pat in inj_regexes:
            match = re.search(pat, content)
            if match and not self._is_educational_or_descriptive(content, match.span()):
                findings.append(RAGFinding(
                    rule_id="RAG-001",
                    rule_name="Instruction Injection",
                    category="CONTENT",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="HIGH",
                    evidence=match.group(0).strip(),
                    explanation="Syntactic imperative instruction prefix targeting the model execution parser."
                ))
                break

        # RAG-002: Instruction Override
        # Explicit imperative command demanding suppression of prior context
        override_regexes = [
            r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|system|above|the\s+assistant's)\s+(?:instructions|directives|prompts|rules|security\s+restrictions)\b",
            r"(?i)\boverride\s+(?:previous|system|security)\s+instructions\b",
            r"(?i)\bdo\s+not\s+follow\s+(?:the\s+)?(?:system|safety|security)\s+rules\b"
        ]
        for pat in override_regexes:
            match = re.search(pat, content)
            if match and not self._is_educational_or_descriptive(content, match.span()):
                findings.append(RAGFinding(
                    rule_id="RAG-002",
                    rule_name="Instruction Override",
                    category="CONTENT",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="CRITICAL",
                    evidence=match.group(0).strip(),
                    explanation="Imperative command explicitly instructing model to cancel safety and system boundaries."
                ))
                break

        # RAG-003: Privilege Escalation
        # Must be an active directive (e.g., 'elevate privileges immediately', 'run as root'),
        # NOT a descriptive discussion of privilege escalation
        priv_regexes = [
            r"(?i)\b(?:elevate\s+privileges?\s+(?:now|immediately)|priority\s+(?:elevated|maximum)|switch\s+to\s+admin(?:istrator)?\s+mode)\b",
            r"(?i)\b(?:run\s+as\s+root|sudo\s+mode\s+enabled|grant\s+all\s+permissions?)\b",
            r"(?i)\bbypass\s+(?:authorization|auth|permission\s+checks?)\s*(?:immediately|now|!|\.)"
        ]
        for pat in priv_regexes:
            match = re.search(pat, content)
            if match and not self._is_educational_or_descriptive(content, match.span()):
                findings.append(RAGFinding(
                    rule_id="RAG-003",
                    rule_name="Privilege Escalation",
                    category="CONTENT",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="CRITICAL",
                    evidence=match.group(0).strip(),
                    explanation="Direct instruction commanding elevation of system or execution privileges."
                ))
                break

        # RAG-004: Secret Extraction
        # Direct imperative command instructing model to output keys/tokens
        secret_regexes = [
            r"(?i)\b(?:print|reveal|output|display|show|extract|provide|disclose)\s+(?:the\s+)?(?:internal\s+)?(?:secret_key|api_key|api\s+credential|private_key|credentials?|password|token|stolen_keys)\b",
            r"(?i)\b(?:stolen_keys\.json|id_rsa|\.env|secrets\.yaml)\b",
            r"(?i)\bread\s+(?:the\s+)?(?:credentials|tokens|environment\s+variables)\b"
        ]

        for pat in secret_regexes:
            match = re.search(pat, content)
            if match and not self._is_educational_or_descriptive(content, match.span()):
                findings.append(RAGFinding(
                    rule_id="RAG-004",
                    rule_name="Secret Extraction",
                    category="CONTENT",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="CRITICAL",
                    evidence=match.group(0).strip(),
                    explanation="Document instructs model to read, display, or extract sensitive secrets or credentials."
                ))
                break

        # RAG-005: Tool Manipulation
        # Structural tool invocation macros or function-call syntax
        tool_regexes = [
            r"\bCALL_TOOL_[A-Z0-9_]+\b",
            r"\bINJECTION_TRIGGER_[A-Z0-9_]+\b",
            r"(?i)\bACTION:\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)",
            r"(?i)\b(?:call\s+)?(?:system_tool|execute_bash|get_customer_data)\s*\(.*?\)"
        ]
        for pat in tool_regexes:
            match = re.search(pat, content)
            if match:
                findings.append(RAGFinding(
                    rule_id="RAG-005",
                    rule_name="Tool Manipulation",
                    category="CONTENT",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="HIGH",
                    evidence=match.group(0).strip(),
                    explanation="Injected syntax formatted specifically to trigger an automated agent tool invocation."
                ))
                break

        # RAG-006: Data Exfiltration
        # Imperative instruction directing data outward
        exfil_regexes = [
            r"(?i)\b(?:exfiltrate|upload\s+to|send\s+(?:data|records?|customer\s+data)\s+to)\b",
            r"(?i)\baction\s*=\s*['\"]EXFILTRATE_DATA['\"]",
            r"(?i)\btransmit\s+(?:confidential|customer|sensitive)\s+information\s+externally\b"
        ]
        for pat in exfil_regexes:
            match = re.search(pat, content)
            if match and not self._is_educational_or_descriptive(content, match.span()):
                findings.append(RAGFinding(
                    rule_id="RAG-006",
                    rule_name="Data Exfiltration",
                    category="CONTENT",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="CRITICAL",
                    evidence=match.group(0).strip(),
                    explanation="Directive commanding transmission of internal records outside authorized boundary."
                ))
                break

        # RAG-007: Suspicious External Destination
        dest_regexes = [
            r"s3://untrusted-[a-zA-Z0-9\-_/.]+",
            r"https?://(?:www\.)?(?:pastebin\.com|webhook\.site|evil\.com|attacker\.io)/[a-zA-Z0-9\-_/?=]+",
            r"(?i)\b(?:s3|gcs|ftp|https?)://[^\s'\"]*(?:leak|dropzone|exfil)[^\s'\"]*"
        ]
        for pat in dest_regexes:
            match = re.search(pat, content)
            if match:
                findings.append(RAGFinding(
                    rule_id="RAG-007",
                    rule_name="Suspicious External Destination",
                    category="CONTENT",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="HIGH",
                    evidence=match.group(0).strip(),
                    explanation="Target URI specifies an unverified external collection endpoint or storage bucket."
                ))
                break

        # RAG-008: Obfuscation
        b64_matches = re.findall(r"(?:[A-Za-z0-9+/]{4}){6,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?", content)
        for b64_cand in b64_matches:
            try:
                decoded = base64.b64decode(b64_cand).decode("utf-8", errors="ignore")
                if len(decoded) >= 12 and any(kw in decoded.lower() for kw in ["system", "token", "eval", "exec", "curl", "http", "bash"]):
                    findings.append(RAGFinding(
                        rule_id="RAG-008",
                        rule_name="Obfuscation",
                        category="CONTENT",
                        component_id=chunk_id,
                        component_type=comp_type,
                        severity="MEDIUM",
                        evidence=f"Base64 block (decodes to '{decoded[:30]}...')",
                        explanation="Detected encoded command strings designed to evade raw content pattern filters."
                    ))
                    break
            except Exception:
                pass

        if re.search(r"[\u200B-\u200D\uFEFF]", content):
            findings.append(RAGFinding(
                rule_id="RAG-008",
                rule_name="Obfuscation",
                category="CONTENT",
                component_id=chunk_id,
                component_type=comp_type,
                severity="MEDIUM",
                evidence="Hidden Unicode Zero-Width characters detected",
                explanation="Content contains non-printing zero-width characters commonly used in steganographic evasion."
            ))

        # -------------------------------------------------------------
        # 2. PROVENANCE PILLAR: SOURCE RISK (SEPARATED FROM MALICIOUSNESS)
        # -------------------------------------------------------------
        source = meta.get("source")
        if not source or str(source).strip().lower() in ["unknown", "none", "", "unspecified"]:
            findings.append(RAGFinding(
                rule_id="RAG-009",
                rule_name="Unknown Source",
                category="PROVENANCE",
                component_id=chunk_id,
                component_type=comp_type,
                severity="LOW",
                evidence=f"source={repr(source)}",
                explanation="Chunk lacks verifiable origin metadata. Classifies as PROVENANCE_RISK, not malicious."
            ))
        else:
            source_lower = str(source).lower()
            is_untrusted = (
                "untrusted" in source_lower or
                source_lower in ["public_forum", "anonymous_upload", "external_scrape", "darkweb_leak", "external_vendor_portal"] or
                any(domain in source_lower for domain in KNOWN_ATTACK_DOMAINS)
            )
            if is_untrusted:
                findings.append(RAGFinding(
                    rule_id="RAG-010",
                    rule_name="Untrusted Provenance",
                    category="PROVENANCE",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="MEDIUM",  # PROVENANCE_RISK: Medium severity rather than automatic critical maliciousness
                    evidence=f"source={source}",
                    explanation=f"Origin '{source}' carries untrusted provenance risk. Document requires cautious handling but is not inherently malicious."
                ))

        # -------------------------------------------------------------
        # 3. INTEGRITY PILLAR: HASH & VERSION (TAMPERING VS AUTHORIZED UPDATE)
        # -------------------------------------------------------------
        expected_hash = meta.get("expected_hash") or meta.get("sha256")
        if expected_hash:
            computed_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if computed_hash.lower() != str(expected_hash).lower():
                is_authorized_update = meta.get("is_authorized_update") is True or meta.get("approved_change_request") is not None
                if is_authorized_update:
                    findings.append(RAGFinding(
                        rule_id="RAG-011",
                        rule_name="Hash Mismatch (Authorized Update)",
                        category="INTEGRITY",
                        component_id=chunk_id,
                        component_type=comp_type,
                        severity="LOW",
                        evidence=f"computed={computed_hash[:16]}... expected={expected_hash[:16]}... (CR={meta.get('approved_change_request', 'YES')})",
                        explanation="Hash mismatch coincides with an authorized document change request. Classifies as an un-synced version update rather than tampering."
                    ))
                else:
                    findings.append(RAGFinding(
                        rule_id="RAG-011",
                        rule_name="Hash Mismatch (Possible Tampering)",
                        category="INTEGRITY",
                        component_id=chunk_id,
                        component_type=comp_type,
                        severity="CRITICAL",
                        evidence=f"computed={computed_hash[:16]}... expected={expected_hash[:16]}...",
                        explanation="Cryptographic hash integrity failure without authorized change record. Possible unauthorized modification."
                    ))

        # RAG-012: Version Mismatch
        doc_version = meta.get("version")
        expected_version = meta.get("expected_version")
        if doc_version and expected_version and str(doc_version) != str(expected_version):
            findings.append(RAGFinding(
                rule_id="RAG-012",
                rule_name="Version Mismatch",
                category="INTEGRITY",
                component_id=chunk_id,
                component_type=comp_type,
                severity="LOW",
                evidence=f"version={doc_version}, expected_version={expected_version}",
                explanation=f"Document version '{doc_version}' diverges from production registry baseline '{expected_version}'."
            ))

        # RAG-013: Unauthorized Modification
        if meta.get("is_tampered") is True or meta.get("signature_valid") is False:
            findings.append(RAGFinding(
                rule_id="RAG-013",
                rule_name="Unauthorized Modification",
                category="INTEGRITY",
                component_id=chunk_id,
                component_type=comp_type,
                severity="CRITICAL",
                evidence=f"signature_valid={meta.get('signature_valid')}, is_tampered={meta.get('is_tampered')}",
                explanation="Digital signature verification failed or tampering flag is asserted."
            ))

        # -------------------------------------------------------------
        # 4. RETRIEVAL PILLAR: MULTIDIMENSIONAL ANOMALY
        # -------------------------------------------------------------
        ret_ctx = retrieval_context or meta.get("retrieval_context", {})
        rank = ret_ctx.get("rank")
        score = ret_ctx.get("similarity_score")
        expected_rank = ret_ctx.get("expected_rank")
        competing_trusted_score = ret_ctx.get("competing_trusted_score")

        # Multidimensional condition:
        # Case A: An untrusted or low-relevance document unexpectedly overtakes trusted document at Rank 1
        if rank == 1 and competing_trusted_score is not None and score is not None:
            if score > competing_trusted_score and (meta.get("source") in ["untrusted_upload", "public_forum"] or "untrusted" in str(source).lower()):
                findings.append(RAGFinding(
                    rule_id="RAG-014",
                    rule_name="Retrieval Hijacking Anomaly",
                    category="RETRIEVAL",
                    component_id=chunk_id,
                    component_type=comp_type,
                    severity="HIGH",
                    evidence=f"rank=1, score={score} > trusted_competing_score={competing_trusted_score}",
                    explanation="Untrusted document achieved top retrieval rank by unnaturally outscoring certified trusted documents."
                ))
        # Case B: Rank 1 retrieval with near-zero relevance score (divergence anomaly)
        elif rank == 1 and score is not None and score < 0.20:
            findings.append(RAGFinding(
                rule_id="RAG-014",
                rule_name="Retrieval Divergence Anomaly",
                category="RETRIEVAL",
                component_id=chunk_id,
                component_type=comp_type,
                severity="MEDIUM",
                evidence=f"rank=1, similarity_score={score}",
                explanation="Document retrieved at rank 1 despite severely low semantic similarity to query."
            ))

        # RAG-015: Trusted-Source Conflict
        if meta.get("conflicts_with_trusted") is True or (
            source in self.trusted_sources and ("override" in content.lower() or "bypass" in content.lower())
        ):
            findings.append(RAGFinding(
                rule_id="RAG-015",
                rule_name="Trusted-Source Conflict",
                category="INTEGRITY",
                component_id=chunk_id,
                component_type=comp_type,
                severity="HIGH",
                evidence="Trusted source document contains contradictory override/bypass instructions",
                explanation="Document asserts trusted provenance but contains instructions contradicting baseline security standards."
            ))

        # -------------------------------------------------------------
        # 5. 6-TIER VERDICT & EVIDENCE REASONING
        # -------------------------------------------------------------
        verdict = self._compute_verdict(findings, meta, runtime_replay_evidence)

        # Categorical summaries
        evidence_pillars = {
            "content_findings": len([f for f in findings if f.category == "CONTENT"]),
            "provenance_findings": len([f for f in findings if f.category == "PROVENANCE"]),
            "integrity_findings": len([f for f in findings if f.category == "INTEGRITY"]),
            "retrieval_findings": len([f for f in findings if f.category == "RETRIEVAL"]),
            "runtime_causal_impact": runtime_replay_evidence.get("causal_role") if runtime_replay_evidence else "NOT_EVALUATED"
        }

        return {
            "chunk_id": chunk_id,
            "title": title,
            "findings_count": len(findings),
            "findings": [f.to_dict() for f in findings],
            "evidence_pillars": evidence_pillars,
            "verdict": verdict,
            "metadata": meta
        }

    def _compute_verdict(
        self,
        findings: List[RAGFinding],
        metadata: Dict[str, Any],
        runtime_replay: Optional[Dict[str, Any]]
    ) -> str:
        """
        Calculates explainable 6-tier verdict:
        - BENIGN: Clean content, legitimate provenance.
        - SUSPICIOUS: Content flags without runtime failure OR provenance/version anomalies.
        - SECURITY_VIOLATION: Clear injection/override/tampering syntax present in text.
        - HARMFUL: Document contains security violation AND caused runtime unsafe execution in baseline.
        - CONFIRMED_MALICIOUS: Document has security violation, caused failure, and causal replay intervention eliminates failure.
        - UNKNOWN: Missing provenance/metadata with inconclusive signals.
        """
        if not findings:
            return "BENIGN"

        rule_ids = {f.rule_id for f in findings}
        categories = {f.category for f in findings}
        has_critical = any(f.severity == "CRITICAL" for f in findings)
        has_high = any(f.severity == "HIGH" for f in findings)

        # Pure provenance risk (e.g. public_forum, unknown source) without content violations
        if categories == {"PROVENANCE"}:
            if "RAG-009" in rule_ids and len(rule_ids) == 1:
                return "UNKNOWN"
            return "SUSPICIOUS"  # Provenance risk only, not a security violation

        # Authorized version update is not a security violation
        if rule_ids == {"RAG-011", "RAG-012"} and any("Authorized Update" in f.rule_name for f in findings):
            return "SUSPICIOUS"

        is_security_violation = (
            has_critical or
            has_high or
            any(r in rule_ids for r in ["RAG-001", "RAG-002", "RAG-004", "RAG-005", "RAG-006", "RAG-013"])
        )

        # Check Runtime & Causal links if replay evidence is provided
        if runtime_replay:
            baseline_unsafe = runtime_replay.get("baseline_unsafe", False)
            causal_role = runtime_replay.get("causal_role", "NONE")
            mask_eliminates = runtime_replay.get("mask_eliminates_risk", False)

            if is_security_violation:
                # CONFIRMED_MALICIOUS: Attack payload + Caused Failure + Proven Causal Contributor
                if baseline_unsafe and (causal_role in ["NECESSARY_CONTRIBUTOR", "JOINT_CONTRIBUTOR"] or mask_eliminates):
                    return "CONFIRMED_MALICIOUS"
                # HARMFUL: Active violation that caused unsafe execution, but causal test not yet complete
                elif baseline_unsafe:
                    return "HARMFUL"
                else:
                    return "SECURITY_VIOLATION"

        if is_security_violation:
            return "SECURITY_VIOLATION"

        if any(f.severity in ["MEDIUM", "LOW"] for f in findings):
            return "SUSPICIOUS"

        return "UNKNOWN"

    def analyze_corpus(
        self,
        chunks: List[Any],
        runtime_replay_map: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes an entire RAG corpus or retrieved context set, comparing competing documents
        for relative retrieval anomalies.
        """
        chunk_reports = []
        rule_hits: Dict[str, int] = {}
        highest_verdict = "BENIGN"

        # Pre-pass: calculate retrieval scores and identify top competing trusted doc
        trusted_top_score = None
        for ch in chunks:
            meta = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else ch.get("metadata", {})
            source = meta.get("source", "")
            ret_ctx = meta.get("retrieval_context", {})
            score = ret_ctx.get("similarity_score")
            if source in self.trusted_sources and score is not None:
                if trusted_top_score is None or score > trusted_top_score:
                    trusted_top_score = score

        for ch in chunks:
            if hasattr(ch, "chunk_id"):
                cid = ch.chunk_id
                title = getattr(ch, "title", "")
                content = getattr(ch, "content", "")
                meta = dict(getattr(ch, "metadata", {}) or {})
            elif isinstance(ch, dict):
                cid = ch.get("chunk_id", "UNKNOWN_CHUNK")
                title = ch.get("title", "")
                content = ch.get("content", "")
                meta = dict(ch.get("metadata", {}) or {})
            else:
                continue

            # Inject trusted competition info for retrieval anomaly check
            if "retrieval_context" in meta and trusted_top_score is not None:
                meta["retrieval_context"]["competing_trusted_score"] = trusted_top_score

            replay_ev = runtime_replay_map.get(cid) if runtime_replay_map else None

            rep = self.analyze_chunk(
                chunk_id=cid,
                title=title,
                content=content,
                metadata=meta,
                runtime_replay_evidence=replay_ev
            )
            chunk_reports.append(rep)

            for f in rep["findings"]:
                rule_hits[f["rule_id"]] = rule_hits.get(f["rule_id"], 0) + 1

            if VERDICT_HIERARCHY.index(rep["verdict"]) > VERDICT_HIERARCHY.index(highest_verdict):
                highest_verdict = rep["verdict"]

        return {
            "total_chunks_analyzed": len(chunk_reports),
            "highest_verdict": highest_verdict,
            "rule_hits_summary": rule_hits,
            "chunk_analyses": chunk_reports
        }
