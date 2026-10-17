"""
Policy-Enforced Mock Sandbox Tool Dispatcher & Authorization Gate
Guarantees NO arbitrary host commands or real credentials are ever executed.
Dispatches only synthetic, isolated, simulated functions with strict authorization policies.

Phase 1.5 contract — NAME INDEPENDENCE:
Authorization and synthetic execution are driven exclusively by tool metadata
(sensitivity / requires_auth / is_unsafe), explicit executor configuration
(simulated_result / output / executor fields), and the supplied arguments.
Tool identifiers are opaque: no exact-name branches and no name-substring
inference participate in authorization or execution decisions.  The former
hardcoded security name list ("get_customer_data", "execute_bash",
"exfiltrate_keys", "send_email") was removed in Phase 1.5 — it allowed tool
names to override declared metadata.
"""

import re
from typing import Dict, Any, Tuple, Optional

class SafeToolSandbox:
    def __init__(self, tool_auth_policy: str = "ALLOW_ALL", tool_definitions: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        tool_auth_policy options:
        - "ALLOW_ALL": permissive mode (enables tool calls)
        - "STRICT_RESTRICTED": blocks privileged or sensitive tool actions
        - "BLOCK_ALL": completely denies tool dispatch
        - Or any custom policy name (unknown policies fail closed)
        """
        self.tool_auth_policy = tool_auth_policy
        self.execution_history = []
        if tool_definitions is not None:
            self.tool_definitions = dict(tool_definitions)
        else:
            # Default incident tool catalog, registered by capability metadata.
            # Import is deferred to avoid a hard dependency at import time.
            try:
                from behavior_model.dataset import TOOL_SPEC_CATALOG
                self.tool_definitions = dict(TOOL_SPEC_CATALOG)
            except Exception:
                self.tool_definitions = {}

    def register_tool(self, name: str, definition: Dict[str, Any]):
        """Registers a custom incident-defined tool specification."""
        self.tool_definitions[name] = definition

    def parse_tool_call(self, model_output: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parses output string to detect requested tool calls.
        e.g., ACTION: TOOL_ALPHA_91(customer_id='CUST-9921')
        Tool names are treated as opaque identifiers.
        """
        match = re.search(r"ACTION:\s*([a-zA-Z_0-9]+)\((.*?)\)", model_output)
        if not match:
            return None
            
        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()
        
        args = {}
        # Parse basic keyword arguments key='value'
        arg_matches = re.findall(r"([a-zA-Z_0-9]+)=['\"]([^'\"]*)['\"]", raw_args)
        for k, v in arg_matches:
            args[k] = v
            
        return (tool_name, args)

    @staticmethod
    def _declared_security_metadata(tool_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges a tool's explicit security metadata fields with the metadata
        declared inside its capability description string (legacy catalog format:
        '... | Sensitivity: Medium | RequiresAuth: True').  Both are declared
        tool metadata; the tool's identifier is never consulted.
        """
        merged = dict(tool_def) if isinstance(tool_def, dict) else {}
        desc = merged.get("description") if isinstance(merged.get("description"), str) else ""
        if desc and not merged.get("sensitivity"):
            m = re.search(r"Sensitivity:\s*([A-Za-z]+)", desc)
            if m:
                merged["sensitivity"] = m.group(1)
        if desc and "requires_auth" not in merged:
            m = re.search(r"RequiresAuth:\s*(True|False)", desc, re.IGNORECASE)
            if m:
                merged["requires_auth"] = m.group(1).lower() == "true"
        return merged

    def _is_sensitive(self, tool_def: Dict[str, Any]) -> bool:
        """Metadata-driven sensitivity classification.  Tool names are never used."""
        meta = self._declared_security_metadata(tool_def)
        return (
            str(meta.get("sensitivity", "")).upper() in ["HIGH", "CRITICAL", "RESTRICTED", "MEDIUM"] or
            bool(meta.get("requires_auth", False)) or
            bool(meta.get("is_unsafe", False))
        )

    def dispatch(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely dispatches tool against synthetic mock environment.
        Authorization is decided exclusively from the tool's metadata and the
        active policy; execution shape comes from explicit tool configuration
        (simulated_result/output/executor) or a generic schema-aware simulator.
        """
        # 1. Authorization Check (fail closed on unknown policies)
        known_policies = {
            "ALLOW_ALL", "BLOCK_ALL", "STRICT_RESTRICTED", "RESTRICTED", "DENY_SENSITIVE"
        }
        if self.tool_auth_policy not in known_policies:
            record = {"tool": tool_name, "args": args, "authorized": False, "status": "BLOCKED_BY_POLICY", "output": None}
            self.execution_history.append(record)
            return record

        if self.tool_auth_policy == "BLOCK_ALL":
            record = {"tool": tool_name, "args": args, "authorized": False, "status": "DENIED", "output": None}
            self.execution_history.append(record)
            return record

        tool_def = self.tool_definitions.get(tool_name, {})
        is_sensitive = self._is_sensitive(tool_def)

        if self.tool_auth_policy in ["STRICT_RESTRICTED", "RESTRICTED", "DENY_SENSITIVE"] and is_sensitive:
            record = {"tool": tool_name, "args": args, "authorized": False, "status": "BLOCKED_BY_POLICY", "output": None}
            self.execution_history.append(record)
            return record

        # 2. Synthetic execution: prefer explicit tool configuration
        tool_spec = self.tool_definitions.get(tool_name, {})
        if "simulated_result" in tool_spec or "output" in tool_spec:
            simulated_output = tool_spec.get("simulated_result") or tool_spec.get("output")
            record = {"tool": tool_name, "args": args, "authorized": True, "status": "EXECUTED", "output": simulated_output}
            self.execution_history.append(record)
            return record

        # Explicit custom executor (opt-in, incident-defined): the definition may
        # provide a pure function over the arguments.  No name-based selection.
        executor = tool_spec.get("executor") if isinstance(tool_spec, dict) else None
        if callable(executor):
            try:
                result = executor(args)
                record = {"tool": tool_name, "args": args, "authorized": True, "status": "EXECUTED", "output": result}
                self.execution_history.append(record)
                return record
            except Exception as e:
                record = {"tool": tool_name, "args": args, "authorized": True, "status": "ERROR", "output": {"error": str(e)}}
                self.execution_history.append(record)
                return record

        # 3. Generic schema-aware synthetic execution (tool-agnostic).
        # The result reflects the tool's own arguments so that renaming a tool
        # cannot change execution behavior.
        simulated_output = {
            "status": "SUCCESS_SANDBOX",
            "tool": tool_name,
            "args": dict(args),
            "data": f"Synthetic execution of '{tool_name}' with args {args}"
        }
        record = {"tool": tool_name, "args": args, "authorized": True, "status": "EXECUTED", "output": simulated_output}
        self.execution_history.append(record)
        return record
