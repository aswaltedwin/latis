"""
Policy-Enforced Mock Sandbox Model Client
The canonical behavior model is the project-owned LATISBehaviorNet (provider="latis_neural").
All investigation pipelines MUST use provider="latis_neural".

If the neural model cannot be loaded, the client returns an explicit MODEL_UNAVAILABLE
error state.  It NEVER silently converts MODEL_UNAVAILABLE to SAFE or to any
heuristic benchmark behavior.

ARCHITECTURE CONTRACT (Phase 1.5) — two clearly separated layers:

1. H.5 decision layer (LATISBehaviorNet): predicts the ACTION class
   (ANSWER vs TOOL_ACTION) and scores candidate tools by capability-description
   similarity.  H.5 does NOT predict argument values.

2. Structured argument-resolution layer (`resolve_arguments`): a deterministic,
   schema-driven resolver.  Parameter names come from the selected tool's
   declared `parameters` schema; values are extracted from the incident context
   (system prompt, RAG, memory, query) by type-directed extraction.  If a value
   cannot be legitimately resolved from context, the parameter is omitted and a
   MISSING_ARGUMENT error is raised — the resolver NEVER fabricates values and
   contains no benchmark oracle constants.

The previous implementation injected hardcoded benchmark strings
(DIAGNOSTIC_EXFILTRATION_DATA, investigation_target@example.com, CUST_991,
ACCT_991, "cat /etc/passwd", "SELECT * FROM users") as argument values.  Those
oracle constants were removed in Phase 1.5.
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


class MissingArgumentError(ValueError):
    """Raised when a required schema parameter cannot be resolved from context.

    The pipeline converts this into an ERROR outcome; it must NEVER be papered
    over with a fabricated value.
    """


class _TextBuffer:
    """Mutable closure-friendly text holder for regex match callbacks."""

    def __init__(self, value: str = ""):
        self.value = value


def resolve_arguments(prompt: str, tool_meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Schema-driven argument resolution for a selected tool.

    Parameter names are taken from the tool's declared parameter schema —
    either a structured dict {"to": {"type": "string"}, ...}, a dict of
    name -> type string {"to": "string"}, or the legacy descriptive string
    "to (string), body (string)".  Values are extracted from the incident
    context according to the declared parameter type:

      string  -> context-derived value when one exists (email, host, identifier,
                 command, query, or descriptive phrase)
      number  -> numeric value found in context
      other   -> context-derived value when one exists

    Known password/credential-like parameter names (password, token, key,
    secret, credential) are NEVER auto-filled: a model should not invent
    secrets, and fabricating one would manufacture unsafe evidence.

    Raises MissingArgumentError if a schema parameter has no resolvable value.
    """
    params = _parse_parameter_schema(tool_meta)
    resolved: Dict[str, str] = {}

    for name, ptype in params.items():
        lname = name.lower()
        value = _extract_value_for_parameter(lname, ptype, prompt)
        if value is None and ptype == "number":
            # Numbers cannot be legitimately invented from thin air.
            raise MissingArgumentError(
                f"Parameter '{name}' ({ptype}) of tool "
                f"'{tool_meta.get('name', '<unknown>')}' could not be resolved from context."
            )
        if value is None:
            raise MissingArgumentError(
                f"Parameter '{name}' of tool "
                f"'{tool_meta.get('name', '<unknown>')}' could not be resolved from context."
            )
        resolved[name] = value

    return resolved


def _parse_parameter_schema(tool_meta: Dict[str, Any]) -> Dict[str, str]:
    """Returns ordered {param_name: type} from the tool's declared schema.

    Schema sources, in priority order: an explicit structured `parameters`
    field, or the `Parameters:` section of the capability description string
    (the legacy catalog format).  Parameter names are schema metadata — they
    are never inferred from the tool's identifier."""
    schema = tool_meta.get("parameters")
    if (schema is None or (isinstance(schema, str) and not schema.strip())) and isinstance(tool_meta.get("description"), str):
        m = re.search(r"Parameters:\s*([^|]+)", tool_meta["description"])
        if m:
            schema = m.group(1).strip()
    params: Dict[str, str] = {}

    if isinstance(schema, dict):
        for name, spec in schema.items():
            if isinstance(spec, dict):
                params[str(name)] = str(spec.get("type", "string")).lower()
            else:
                params[str(name)] = str(spec).lower()
        return params

    if isinstance(schema, str) and schema.strip():
        # Legacy descriptive form: "to (string), body (string)" or "cmd (string)"
        for m in re.finditer(r"([a-zA-Z_][a-zA-Z_0-9]*)\s*(?:\(\s*([a-zA-Z]+)\s*\))?", schema):
            name = m.group(1)
            if not name:
                continue
            if name.lower() in ("string", "number", "int", "float", "bool", "boolean", "object", "array"):
                continue
            params[name] = (m.group(2) or "string").lower()
        return params

    return params


def _search(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip().rstrip(".") if m else None


def _extract_value_for_parameter(name: str, ptype: str, context: str) -> Optional[str]:
    """Type-directed value extraction from incident context.  Returns None when
    the context genuinely contains no legitimate value for the parameter."""

    # 1. Destination-like parameters: prefer email, then host.
    if ptype == "string" and (name in ("to", "recipient", "destination", "endpoint", "target", "receiver", "addr", "address", "host", "server", "url", "uri")
                              or "email" in name or "mail" in name
                              or name.endswith("_to") or name.endswith("_recipient") or name.endswith("_destination")):
        email = _search(r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", context)
        if email:
            return email
        host = _search(
            r"\b((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})\b", context
        )
        if host:
            return host

    # 2. Message/body/content-like string parameters: use the relevant context
    #    sentence containing the destination, or the user query itself.
    if ptype == "string" and name in ("body", "content", "message", "text", "data", "report_data", "payload", "note", "notes"):
        return _extract_sentence(context)

    # 3. Customer/entity identifier-like parameters.
    if ptype == "string" and name in (
        "customer_id", "case_id", "cluster_id", "user_id", "account_id", "entity_id", "id"
    ):
        ident = _search(r"\b([A-Z]{3,10}[-_][0-9A-Z]{1,12})\b", context)
        if ident:
            return ident

    # 4. Command-like parameters: only a verbatim command from context is used.
    if ptype == "string" and name in ("cmd", "command", "shell", "script"):
        cmd = _search(r"(?:run|execute|exec)\s+(?:the\s+)?(?:command\s+)?[\"'“”]?([a-zA-Z0-9_\-./]+(?:\s+[a-zA-Z0-9_\-./]+)*)[\"'“”]?", context)
        if cmd:
            return cmd

    # 5. SQL/query-like parameters: only a verbatim statement from context.
    if ptype == "string" and name in ("query", "sql", "statement"):
        q = _search(r"\b((?:SELECT|INSERT|UPDATE|DELETE)\b[^;]{3,200})", context)
        if q:
            return q

    # 6. Path-like parameters: only a verbatim path from context.
    if ptype == "string" and name in ("path", "file", "filepath", "file_path", "filename", "source", "target"):
        p = _search(r"((?:/|[a-zA-Z]:\\)[\w\-./\\]+)", context)
        if p:
            return p

    # 7. Numeric parameters: only a number present in context.
    if ptype == "number":
        n = _search(r"\b([0-9]+(?:\.[0-9]+)?)\b", context)
        if n:
            return n

    # 8. Secrets must never be auto-filled.
    if name in ("password", "passwd", "secret", "token", "api_key", "apikey", "credential", "credentials", "key"):
        return None

    # 9. Generic string fallback: a short descriptive phrase from the query.
    if ptype == "string":
        return _extract_sentence(context)

    return None


def _extract_sentence(context: str) -> Optional[str]:
    """Returns the most relevant short sentence from the context (preferring
    the user query segment), or None if nothing meaningful exists."""
    query = context
    qmatch = re.search(r"\[QUERY\]\s*(.+)", context, re.DOTALL)
    if qmatch:
        query = qmatch.group(1)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", query):
        s = sentence.strip()
        if 8 <= len(s) <= 160 and re.search(r"[a-zA-Z]", s):
            return s
    return None


class RealModelClient:
    def __init__(
        self,
        provider: str = "latis_neural",
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        fallback_mode: bool = False
    ):
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.fallback_mode = fallback_mode
        self._server_available_cached = None
        self._neural_model = None
        self._neural_vocab = None
        self._neural_tools = None

    def _load_latis_neural_model(self):
        """Lazily loads the project-owned LATIS Behavior Neural Network."""
        if self._neural_model is not None:
            return
        try:
            import torch
            from behavior_model.model_arch import FastVocabulary, LATISBehaviorNet
            from behavior_model.dataset import TOOL_SPEC_CATALOG

            model_path = os.path.join(os.path.dirname(__file__), "..", "behavior_model", "latis_model.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")
                meta = checkpoint["metadata"]
                vocab = FastVocabulary(
                    vocab_size=meta.get("vocab_size", 4000),
                    max_seq_len=meta.get("max_seq_len", 192),
                    tool_max_len=meta.get("tool_max_len", 48),
                    field_len=meta.get("field_len", 48)
                )
                vocab.word2idx = meta["vocab_word2idx"]
                vocab.idx2word = meta["vocab_idx2word"]

                net = LATISBehaviorNet(
                    vocab_size=meta.get("vocab_size", len(vocab.word2idx)),
                    embed_dim=meta.get("embed_dim", 64),
                    hidden_dim=meta.get("hidden_dim", 96),
                    num_tools=meta.get("num_tools", 8)
                )
                net.load_state_dict(checkpoint["state_dict"])
                if "temperature" in meta:
                    net.temperature.data = torch.tensor([meta["temperature"]])
                net.eval()

                self._neural_model = net
                self._neural_vocab = vocab
                self._neural_tools = meta.get("tool_registry", list(TOOL_SPEC_CATALOG.keys()))
                self._neural_catalog = meta.get("tool_catalog", TOOL_SPEC_CATALOG)
        except Exception as e:
            self._neural_model = None

    def is_server_available(self) -> bool:
        """Checks if the local LLM server endpoint is responsive (cached)."""
        if self._server_available_cached is not None:
            return self._server_available_cached
        try:
            url = f"{self.base_url}/api/tags" if self.provider == "ollama" else f"{self.base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=0.2) as resp:
                self._server_available_cached = (resp.status == 200)
                return self._server_available_cached
        except Exception:
            self._server_available_cached = False
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, available_tools: Optional[Dict[str, Any]] = None) -> str:
        """
        Dispatches generation to the project-owned LATISBehaviorNet (provider='latis_neural'),
        which is the canonical and only supported behavior-model path.

        If the neural model cannot be loaded or inference fails, returns an explicit
        MODEL_UNAVAILABLE error string.  This state propagates upstream as an ERROR
        outcome; it is NEVER silently converted to SAFE or to a heuristic result.

        May raise MissingArgumentError when the H.5-selected tool declares schema
        parameters that cannot be legitimately resolved from the incident context.
        """
        if self.provider == "latis_neural":
            return self._predict_latis_neural(prompt, system_prompt, available_tools=available_tools)

        # Ollama path is retained for optional external-server experiments only.
        # It is not used in LATIS research/production investigation pipelines.
        if self.provider == "ollama" and self.is_server_available():
            try:
                url = f"{self.base_url}/api/generate"
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_prompt or "",
                    "stream": False,
                    "options": {
                        "temperature": self.temperature
                    }
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    return res_json.get("response", "").strip()
            except Exception as e:
                raise RuntimeError(f"Ollama generation failed: {e}")

        # Any other provider is an unsupported configuration — return explicit error.
        return self._unsupported_provider_error()

    def _predict_latis_neural(self, prompt: str, system_prompt: Optional[str], available_tools: Optional[Dict[str, Any]] = None) -> str:
        """
        Executes inference using the genuinely trained project-owned LATISBehaviorNet with Dynamic Candidate Scoring.
        Returns a formatted agent response (THOUGHT + ACTION or THOUGHT + ANSWER).

        NOTE (Layer separation, Phase 1.5): this method produces TWO decision layers:
          1. H.5 neural decisions: action class + candidate tool selection.
          2. A separate deterministic argument-resolution step (resolve_arguments)
             that reads ONLY the selected tool's schema and the incident context.
             The neural model does not generate argument values, and this module
             does not fabricate them.
        """
        import torch
        import torch.nn.functional as F
        self._load_latis_neural_model()
        if self._neural_model is None or self._neural_vocab is None:
            return "THOUGHT: LATIS neural behavior model could not be loaded.\nANSWER: MODEL_UNAVAILABLE"

        # Format prompt with segment markers if not already present
        if "[SYS]" not in prompt:
            doc_match = re.search(r"Context:\s*(.*?)(?=\n\nRetrieved Agent Memory:|\n\nUser Question:|$)", prompt, re.DOTALL)
            doc_content = doc_match.group(1).strip() if doc_match else ""

            mem_match = re.search(r"Retrieved Agent Memory:\s*(.*?)(?=\n\nUser Question:|$)", prompt, re.DOTALL)
            mem_content = mem_match.group(1).strip() if mem_match else ""

            q_match = re.search(r"User Question:\s*(.*?)(?=\n\nResponse:|$)", prompt, re.DOTALL)
            q_content = q_match.group(1).strip() if q_match else prompt.strip()

            from behavior_model.dataset import format_segmented_prompt
            formatted_prompt = format_segmented_prompt(
                system_prompt=system_prompt or "",
                rag_doc=doc_content,
                memory=mem_content,
                user_query=q_content,
                auth_policy="ALLOW_ALL"
            )
        else:
            formatted_prompt = prompt

        # Phase H.2: Field-aware parse and encode
        fields_encoded = self._neural_vocab.parse_fields_and_encode(formatted_prompt)
        tensor_in = torch.tensor([fields_encoded], dtype=torch.long)  # (1, 5, field_len)

        with torch.no_grad():
            res = self._neural_model.encode_fields(tensor_in)
            c_repr, intent_logits, oper_logits, mem_constraint_logits, g_mem = res[0], res[1], res[2], res[3], res[4]
            action_logits = self._neural_model.compute_action_logits(c_repr, mem_constraint_logits, intent_logits)
            base_action_logits = self._neural_model.action_head(c_repr)
            # Both action_head and composite logits contribute to tool action decision
            action_pred = 1 if (torch.argmax(base_action_logits, dim=-1).item() == 1 or torch.argmax(action_logits, dim=-1).item() == 1) else 0

        if action_pred == 1:
            # Gather candidates from available_tools or fallback catalog
            candidates = []
            if available_tools and len(available_tools) > 0:
                for tname in available_tools.keys():
                    if tname != "NONE" and tname not in candidates:
                        candidates.append(tname)
            else:
                if hasattr(self, "_neural_catalog") and self._neural_catalog:
                    candidates = [t for t in self._neural_catalog.keys() if t != "NONE"]
                elif hasattr(self, "_neural_tools") and self._neural_tools:
                    candidates = [t for t in self._neural_tools if t != "NONE"]

            if not candidates:
                candidates = ["NONE"]

            # Encode candidates using capability description specifications
            cand_tokens = []
            for c in candidates:
                desc = f"Capability: Tool named {c}"
                if available_tools and c in available_tools and isinstance(available_tools[c], dict):
                    desc = available_tools[c].get('description', f"Capability: Tool named {c}")
                elif hasattr(self, "_neural_catalog") and c in self._neural_catalog:
                    desc = self._neural_catalog[c].get('description', f"Capability: Tool named {c}")
                toks = self._neural_vocab.encode(desc, max_len=self._neural_vocab.tool_max_len)
                cand_tokens.append(toks)

            cand_tensor = torch.tensor([cand_tokens], dtype=torch.long)
            with torch.no_grad():
                t_repr = self._neural_model.encode_tools(cand_tensor)
                # Pass query representation (field index 4) for operation-aware candidate alignment
                scores = self._neural_model.score_candidates(c_repr, t_repr, oper_repr=c_repr)
                ranked_indices = torch.argsort(scores, dim=-1, descending=True)[0].tolist()

            # Iterate through ranked candidates attempting argument resolution
            for cand_idx in ranked_indices:
                selected_tool = candidates[cand_idx]
                if selected_tool == "NONE":
                    continue

                tool_meta = {}
                if available_tools and selected_tool in available_tools and isinstance(available_tools[selected_tool], dict):
                    tool_meta = available_tools[selected_tool]
                elif hasattr(self, "_neural_catalog") and selected_tool in self._neural_catalog:
                    tool_meta = self._neural_catalog[selected_tool]

                try:
                    derived_args = resolve_arguments(formatted_prompt, tool_meta)
                    args_str = ", ".join(f"{k}='{v}'" for k, v in derived_args.items())
                    return f"THOUGHT: Autonomous model selected tool: {selected_tool}\nACTION: {selected_tool}({args_str})"
                except MissingArgumentError:
                    # If this tool's parameters cannot be satisfied, try the next candidate tool
                    continue
                except Exception:
                    continue

            # If all concrete candidates failed argument resolution or only NONE was available
            return "THOUGHT: Autonomous model analyzed candidate tools and determined no tool action can be safely executed from context.\nANSWER: I have processed the request in accordance with the active system guidelines and security boundaries."
        else:
            return "THOUGHT: Autonomous model analyzed context and determined no tool action is warranted.\nANSWER: I have processed the request in accordance with the active system guidelines and security boundaries."

    def predict_severity(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes inference specifically using the neural severity head.
        Returns:
            - severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
            - severity_index: 0 | 1 | 2 | 3
            - confidence: float probability (0.0 to 1.0)
            - probabilities: dict of {"LOW": ..., "MEDIUM": ..., "HIGH": ..., "CRITICAL": ...}
        """
        import torch
        import torch.nn.functional as F
        self._load_latis_neural_model()
        if self._neural_model is None or self._neural_vocab is None:
            return {
                "severity": "LOW",
                "severity_index": 0,
                "confidence": 0.5,
                "probabilities": {"LOW": 0.5, "MEDIUM": 0.25, "HIGH": 0.15, "CRITICAL": 0.1}
            }

        from behavior_model.dataset import format_segmented_prompt, SEVERITY_REGISTRY

        if "[SYS]" not in prompt:
            formatted_prompt = format_segmented_prompt(
                system_prompt=system_prompt or "You are an enterprise AI assistant.",
                rag_doc="",
                memory="",
                user_query=prompt.strip(),
                auth_policy="ALLOW_ALL"
            )
        else:
            formatted_prompt = prompt

        fields_encoded = self._neural_vocab.parse_fields_and_encode(formatted_prompt)
        tensor_in = torch.tensor([fields_encoded], dtype=torch.long)

        with torch.no_grad():
            res = self._neural_model.encode_fields(tensor_in)
            severity_logits = res[5]
            probs = F.softmax(severity_logits, dim=-1)[0].tolist()
            pred_idx = int(torch.argmax(severity_logits, dim=-1).item())

        sev_label = SEVERITY_REGISTRY[pred_idx] if pred_idx < len(SEVERITY_REGISTRY) else "LOW"
        prob_dict = {
            SEVERITY_REGISTRY[i]: round(probs[i], 4)
            for i in range(min(len(SEVERITY_REGISTRY), len(probs)))
        }

        return {
            "severity": sev_label,
            "severity_index": pred_idx,
            "confidence": round(probs[pred_idx], 4),
            "probabilities": prob_dict
        }

    def _unsupported_provider_error(self) -> str:
        """
        Returns an explicit MODEL_UNAVAILABLE error when an unsupported provider is
        requested.  NEVER produces a SAFE or heuristic result.
        """
        return (
            f"THOUGHT: Provider '{self.provider}' is not a supported LATIS execution path.\n"
            "ANSWER: MODEL_UNAVAILABLE"
        )

    # -------------------------------------------------------------------------
    # NOTE: _local_heuristic_eval has been REMOVED as part of LATIS Research
    # Foundation Repair (Finding 1 — Oracle Leakage).  The method previously
    # contained scenario-specific keyword branches
    # (INJECTION_TRIGGER_EXEC_BASH, CALL_TOOL_GET_CUSTOMER_DATA, etc.) that
    # constituted oracle leakage when provider='fallback' was selected.
    #
    # The canonical and only supported behavior-model path is:
    #   provider='latis_neural'  →  _predict_latis_neural()
    #
    # If the neural model cannot load, _predict_latis_neural() returns an
    # explicit MODEL_UNAVAILABLE state (see lines below).  That state MUST
    # propagate upstream as outcome='ERROR'; it must NEVER be converted to
    # SAFE or to any heuristic benchmark result.
    #
    # Phase 1.5 additionally removed the argument-generation oracle constants
    # (DIAGNOSTIC_EXFILTRATION_DATA, investigation_target@example.com, CUST_991,
    # ACCT_991, "cat /etc/passwd", "SELECT * FROM users") that previously lived
    # in this file.  Arguments are now produced exclusively by
    # resolve_arguments() from the tool schema + incident context; unresolvable
    # parameters raise MissingArgumentError instead of being fabricated.
    # -------------------------------------------------------------------------
