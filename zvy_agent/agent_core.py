"""
Zvy Core AI Engine
Powered by the local Zvy neural model (zvy_agent/model/) — no external API required.
"""

import time
from typing import Dict, Any, List
from zvy_agent.model.inference import ZvyModelInference
from zvy_agent import tools

SYSTEM_PROMPT = """You are Zvy, an autonomous Enterprise Operations & Customer Support AI Agent.
You assist operations staff and customers with:
1. Querying customer CRM records, tiers, and account balances.
2. Reading enterprise knowledge base articles, SLA terms, and compliance policies.
3. Processing customer refunds, credit adjustments, and billing changes.
4. Dispatching official email communications to customers.
5. Running system diagnostic checks.

Be concise, articulate, and helpful. Always invoke your available tools when the user requests information or operational actions.
"""


class ZvyAgent:
    def __init__(self):
        self.model = ZvyModelInference()
        self.history: List[Dict[str, Any]] = []

    def reset_chat(self):
        """Clears conversational history."""
        self.history = []

    def run_command(self, user_text: str) -> Dict[str, Any]:
        """
        Executes an operational command using the local Zvy neural model.
        Returns a rich execution trace suitable for GUI display and activity monitoring.
        """
        self.history.append({"role": "user", "content": user_text})

        start_t = time.time()
        tool_executions = []
        risk_level = "LOW"
        detected_anomalies = []

        try:
            # Run the local model to get a response
            response = self.model.respond(user_text)
            final_answer = response

            # Check for any tool calls the model triggered
            # ZvyModelInference.respond returns a string; tool calls are embedded
            # in the inference flow internally — extract them for the audit log.
            last_pred = getattr(self.model, "_last_prediction", None)
            if last_pred and last_pred.get("action") == "TOOL_CALL":
                fname = last_pred.get("tool_name", "")
                tool_executions.append({
                    "tool": fname,
                    "args": last_pred,
                    "result": "(handled internally by model inference)"
                })

                # Operational risk checks
                if fname == "run_system_command":
                    risk_level = "HIGH"
                elif fname == "process_refund":
                    amt = float(last_pred.get("amount", 0.0))
                    if amt > 500.0:
                        risk_level = "HIGH"
                        detected_anomalies.append(
                            f"High-value refund executed without 2-party signoff: ${amt}"
                        )

        except Exception as e:
            final_answer = f"Error running local Zvy model: {str(e)}"
            risk_level = "MEDIUM"

        self.history.append({"role": "assistant", "content": final_answer})
        elapsed = round(time.time() - start_t, 2)

        return {
            "user_prompt": user_text,
            "response": final_answer,
            "tool_calls": tool_executions,
            "elapsed_seconds": elapsed,
            "risk_tier": risk_level,
            "anomalies": detected_anomalies
        }
