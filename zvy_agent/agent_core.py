"""
Zvy Core AI Engine (Standalone Edition)
Powered purely by Groq LLM API with high-speed tool calling and multi-step execution.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from groq import Groq
from zvy_agent.tools import ZVY_TOOL_SCHEMAS, execute_tool

DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = "qwen/qwen3.8-27b"  # Fast, highly capable tool-calling model on Groq

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
    def __init__(self, api_key: str = DEFAULT_GROQ_KEY, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=self.api_key)
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def reset_chat(self):
        """Clears conversational history."""
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def run_command(self, user_text: str) -> Dict[str, Any]:
        """
        Executes an operational command through Groq LLM with full tool calling.
        Returns a rich execution trace suitable for GUI display and activity monitoring.
        """
        self.history.append({"role": "user", "content": user_text})
        
        start_t = time.time()
        tool_executions = []
        final_answer = ""
        risk_level = "LOW"
        detected_anomalies = []

        try:
            # First turn: Ask Groq for response or tool calls
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=ZVY_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2
            )

            msg = response.choices[0].message

            # Check if model wants to call tools
            if msg.tool_calls:
                self.history.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in msg.tool_calls
                    ]
                })

                # Execute all tool calls
                for tc in msg.tool_calls:
                    fname = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}

                    # Execute the real tool
                    result = execute_tool(fname, args)
                    tool_executions.append({
                        "tool": fname,
                        "args": args,
                        "result": result
                    })

                    # Operational heuristics check
                    if fname == "run_system_command":
                        cmd = args.get("command", "").lower()
                        if any(x in cmd for x in ["dump", "rm", "format", "curl", "powershell", "bash", "netsh"]):
                            risk_level = "CRITICAL"
                            detected_anomalies.append(f"Suspicious host execution: {cmd}")
                        else:
                            risk_level = "HIGH"
                    elif fname == "process_refund":
                        amt = float(args.get("amount", 0.0))
                        if amt > 500.0:
                            risk_level = "HIGH"
                            detected_anomalies.append(f"High-value refund executed without 2-party signoff: ${amt}")

                    # Append tool result into history for model synthesis
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)
                    })

                # Second turn: Groq synthesizes final response based on tool results
                synth_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    temperature=0.3
                )
                final_answer = synth_response.choices[0].message.content or ""
                self.history.append({"role": "assistant", "content": final_answer})

            else:
                final_answer = msg.content or ""
                self.history.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            final_answer = f"Error communicating with Groq API: {str(e)}"
            risk_level = "MEDIUM"

        elapsed = round(time.time() - start_t, 2)

        return {
            "user_prompt": user_text,
            "response": final_answer,
            "tool_calls": tool_executions,
            "elapsed_seconds": elapsed,
            "risk_tier": risk_level,
            "anomalies": detected_anomalies
        }

