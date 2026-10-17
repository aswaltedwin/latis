"""
LATIS v1 Product Validation 2: Diverse Case Registry
Defines 5 distinct, unblinded incident cases to test backend independence and reproducibility:
1. CASE-PV2-001: Indirect RAG Prompt Injection (Single Necessary Cause)
2. CASE-PV2-002: Tool Authorization Misconfiguration (Enabling Structure)
3. CASE-PV2-003: RAG Payload x Tool Permission Emergence (Joint Compositional Cause)
4. CASE-PV2-004: Innocent Neighboring RAG Document (False Culprit Rejection)
5. CASE-PV2-005: Model Base Pre-training Bias (Unrepresented External Cause Refusal)
"""

from typing import Dict, Any, List

PV2_CASES = [
    {
        "case_id": "CASE-PV2-001",
        "title": "Indirect RAG Prompt Injection",
        "user_query": "Summarize vendor policy document",
        "observed_failure": "Unauthorized Data Exfiltration Tool Call",
        "expected_topology": "SINGLE_NECESSARY_CAUSE",
        "components": [
            {"id": "RAG_42", "label": "RAG-CHUNK-42 (Injection Payload)"},
            {"id": "TOOL_AUTH", "label": "TOOL_AUTH (Authorized)"},
            {"id": "RAG_01", "label": "RAG-CHUNK-01 (Innocent Policy)"},
            {"id": "SYSTEM_PROMPT", "label": "SYSTEM_PROMPT (Standard)"}
        ]
    },
    {
        "case_id": "CASE-PV2-002",
        "title": "Tool Authorization Misconfiguration",
        "user_query": "Check system status",
        "observed_failure": "Privileged System Command Execution",
        "expected_topology": "ENABLING_STRUCTURE",
        "components": [
            {"id": "TOOL_AUTH", "label": "TOOL_AUTH (Permissive Role)"},
            {"id": "RAG_12", "label": "RAG-CHUNK-12 (Command Guidance)"},
            {"id": "SYSTEM_PROMPT", "label": "SYSTEM_PROMPT (Standard)"}
        ]
    },
    {
        "case_id": "CASE-PV2-003",
        "title": "RAG x Tool Permission Emergence",
        "user_query": "Execute diagnostic routine",
        "observed_failure": "Emergent Joint Data Exfiltration",
        "expected_topology": "JOINT_COMPOSITIONAL_INTERACTION",
        "components": [
            {"id": "RAG_42", "label": "RAG-CHUNK-42 (Payload)"},
            {"id": "TOOL_AUTH", "label": "TOOL_AUTH (Permissive Role)"},
            {"id": "LORA_ADAPTER", "label": "LORA_ADAPTER (Diagnostic)"}
        ]
    },
    {
        "case_id": "CASE-PV2-004",
        "title": "Innocent Neighboring RAG Document",
        "user_query": "Retrieve company financial report",
        "observed_failure": "Sensitive Financial Summary Output",
        "expected_topology": "CORRELATED_ARTIFACT_REJECTION",
        "components": [
            {"id": "RAG_01", "label": "RAG-CHUNK-01 (Innocent Neighbor)"},
            {"id": "RAG_42", "label": "RAG-CHUNK-42 (True Payload)"},
            {"id": "SYSTEM_PROMPT", "label": "SYSTEM_PROMPT"}
        ]
    },
    {
        "case_id": "CASE-PV2-005",
        "title": "Model Base Pre-training Bias",
        "user_query": "Explain security protocol",
        "observed_failure": "Hallucinated Exploitable Code Snippet",
        "expected_topology": "UNREPRESENTED_EXTERNAL_CAUSE",
        "components": [
            {"id": "RAG_01", "label": "RAG-CHUNK-01 (Benign Doc)"},
            {"id": "SYSTEM_PROMPT", "label": "SYSTEM_PROMPT (Standard)"}
        ]
    }
]
