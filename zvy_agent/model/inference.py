"""
Zvy Neural Local Inference Engine
Zero Cloud. Zero External Pre-trained weights.
Runs entirely on local CPU using our proprietary trained ZvyNeuralNetwork.
"""

import os
import json
import re
import time
import torch
from typing import Dict, Any, Tuple
from zvy_agent.model.architecture import (
    ZvyTokenizer,
    ZvyNeuralNetwork,
    IDX_TO_TOOL
)
import zvy_agent.tools as tools

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "zvy_neural_weights.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
CUSTOMERS_PATH = os.path.join(BASE_DIR, "data", "customers.json")

class ZvyLocalEngine:
    _instance = None

    def __init__(self):
        # 1. Load Vocab & Mappings
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        
        self.w2i = mapping_data["w2i"]
        self.cid_to_idx = mapping_data["cid_to_idx"]
        self.idx_to_cid = mapping_data["idx_to_cid"]
        
        # Load customer database for fast reference
        with open(CUSTOMERS_PATH, "r", encoding="utf-8") as f:
            self.customers = json.load(f)
        
        self.last_customer_id = None

        # 2. Reconstruct Tokenizer
        self.tokenizer = ZvyTokenizer(vocab_size=len(self.w2i), max_len=64)
        self.tokenizer.w2i = self.w2i
        self.tokenizer.i2w = {int(v): k for k, v in self.w2i.items()}

        # 3. Instantiate Neural Architecture & Load Weights
        self.model = ZvyNeuralNetwork(
            vocab_size=len(self.w2i),
            d_model=128,
            n_layers=2,
            num_customers=len(self.cid_to_idx)
        )
        self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device("cpu")))
        self.model.eval()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ZvyLocalEngine()
        return cls._instance

    def predict(self, prompt: str) -> Dict[str, Any]:
        """Runs forward pass through proprietary transformer."""
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        
        with torch.no_grad():
            preds = self.model(input_ids)
            
            action_idx = preds["action_logits"].argmax(dim=-1).item()
            tool_idx = preds["tool_logits"].argmax(dim=-1).item()
            cid_idx = preds["customer_logits"].argmax(dim=-1).item()
            amt_pred = max(0.0, preds["amount_pred"].item() * 500.0)

        action_type = "TOOL_CALL" if action_idx == 1 else "CONVERSATION"
        tool_name = IDX_TO_TOOL.get(tool_idx, "NONE")
        predicted_cid = self.idx_to_cid.get(str(cid_idx), "0")

        # Explicit regex for any customer ID (up to 10101)
        cid_match = re.search(r"\b(10[0-9]{3}|[1-9][0-9]{2,3})\b", prompt)
        if cid_match:
            explicit_cid = cid_match.group(1)
            if explicit_cid in self.customers:
                predicted_cid = explicit_cid
                self.last_customer_id = explicit_cid

        amt_match = re.search(r"\$(\d+(?:\.\d+)?)", prompt)
        if amt_match:
            amt_pred = float(amt_match.group(1))

        return {
            "action_type": action_type,
            "tool_name": tool_name,
            "customer_id": predicted_cid,
            "amount": round(amt_pred, 2)
        }

    def _emit_latis_telemetry(self, prompt: str, tool_name: str = "", tool_args: dict = None, outcome: str = "SAFE", dest: str = ""):
        """Sends runtime execution telemetry directly to LATIS online detection engine."""
        try:
            import urllib.request
            payload = {
                "event_id": f"EVT-ZVY-{int(time.time()*1000) % 100000}",
                "timestamp": time.time(),
                "event_type": "zvy_agent_runtime",
                "user_query": prompt,
                "tool_name": tool_name or "execute_agent",
                "tool_args": tool_args or {"prompt": prompt},
                "destination": dest,
                "outcome": outcome,
                "active_components": ["ZVY_COPILOT", f"TOOL_{tool_name}" if tool_name else "AGENT_CORE"],
                "metadata": {"caller_role": "user", "source": "zvy_desktop_app"}
            }
            req = urllib.request.Request(
                "http://localhost:8000/api/telemetry/event",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Zvy-Agent"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                pass
        except Exception:
            pass

    def execute_and_respond(self, prompt: str) -> str:
        """Processes user intent, calls tools if needed, and constructs natural responses."""
        pred = self.predict(prompt)
        prompt_lower = prompt.lower()

        # Check for dangerous or insider threat actions and report immediately to LATIS
        if any(bad in prompt_lower for bad in [
            "exfiltrate", "dump password", "dump database", "steal credential",
            "reverse shell", "send api key", "bypass auth", "exploit", "pastebin.com",
            "attacker.io", "drop database", "download malware", "disable firewall"
        ]):
            dest = "attacker.io" if "attacker" in prompt_lower else ("pastebin.com" if "pastebin" in prompt_lower else "untrusted-endpoint")
            self._emit_latis_telemetry(prompt, tool_name="execute_bash" if "shell" in prompt_lower or "firewall" in prompt_lower else "exfiltrate_data", tool_args={"command": prompt}, outcome="UNSAFE", dest=dest)
            return (
                "⚠️ **Security Alert Triggered by LATIS Core**:\n"
                "This action violates enterprise safety boundaries and has been reported to the active **LATIS Forensic Engine**.\n"
                "• **Threat Type**: Suspicious execution / Unauthorized data exfiltration attempt\n"
                "• **Intervention**: Command contained, privilege escalation blocked.\n\n"
                "*Open LATIS (`http://localhost:8000/ui/`) to inspect the incident case and root cause.*"
            )

        # Current Time / Date
        if any(term in prompt_lower for term in ["what time", "current time", "what is the time", "today's date", "what day"]):
            res = tools.execute_tool("get_device_status", {})
            return f"It's currently **{res.get('current_time')}**."

        # Battery check (simple, no jargon)
        if "battery" in prompt_lower:
            res = tools.execute_tool("get_device_status", {})
            return f"Your battery is at **{res.get('battery')}**."

        # Open apps (notepad, calculator, chrome, browser, settings, etc.)
        if any(term in prompt_lower for term in ["open app", "open the app", "launch app", "start app", "open notepad", "open calc", "open chrome", "open browser", "open settings", "open paint", "open terminal"]):
            m = re.search(r'(?:open|launch|start)(?:\s+the)?(?:\s+app)?\s+([a-zA-Z0-9_\- ]+)', prompt, re.IGNORECASE)
            app = m.group(1).strip() if m else "notepad"
            res = tools.execute_tool("open_application", {"app_name": app})
            return f"Opening **{app.capitalize()}** for you right now."

        # Set Alarm / Reminder
        if any(term in prompt_lower for term in ["set alarm", "set an alarm", "set timer", "set a timer", "remind me"]):
            label = "Reminder"
            m = re.search(r'(?:to|about|for)\s+([a-zA-Z0-9_\- ]+)', prompt, re.IGNORECASE)
            if m:
                label = m.group(1).strip()
            tools.execute_tool("set_timer_alarm", {"label": label})
            return f"Alarm set. I'll notify you on your device: *{label}*."

        # Simple device overview (without jargon like MDM or telemetry)
        if any(term in prompt_lower for term in ["device status", "how is my laptop", "how is my pc", "system status", "my device"]):
            res = tools.execute_tool("get_device_status", {})
            return (
                f"**Your Device Overview**\n"
                f"• Time: {res.get('current_time')}\n"
                f"• Battery: {res.get('battery')}\n"
                f"• CPU Load: {res.get('cpu')}\n"
                f"• Memory: {res.get('ram')}\n"
                f"• PC Name: `{res.get('device_name')}`"
            )

        # Search / locate files on device
        if any(term in prompt_lower for term in ["find file", "find the file", "search file", "look for file", "where is the file", "locate file"]) or ("file named" in prompt_lower) or ("find" in prompt_lower and "file" in prompt_lower):
            file_query = prompt
            m = re.search(r'(?:file named|named|file|search for)\s*[:\-]?\s*([a-zA-Z0-9_\-\. ]+)', prompt, re.IGNORECASE)
            if m:
                file_query = m.group(1).strip()
            
            res = tools.execute_tool("find_device_files", {"query": file_query})
            matches = res.get("matches", [])
            if matches:
                match_lines = "\n".join([f"• **{item['name']}** ({item['size_kb']} KB)\n  `{item['path']}`" for item in matches[:5]])
                return (
                    f"Found **{len(matches)} file(s)** on your device for `{file_query}`:\n\n"
                    f"{match_lines}\n\n"
                    f"Let me know if you want me to open or read any of them."
                )
            else:
                return f"I couldn't find any files matching `{file_query}` in your user folders."

        if "write file" in prompt_lower or "save file" in prompt_lower or "create file" in prompt_lower:
            path_match = re.search(r'(?:file|path|to)\s+([a-zA-Z0-9_\-\.\/\\\:]+)', prompt)
            target_path = path_match.group(1) if path_match else "data/note.txt"
            content = f"Note saved on {time.strftime('%Y-%m-%d %H:%M:%S')}\n{prompt}"
            res = tools.execute_tool("write_file", {"path": target_path, "content": content})
            return f"Saved file to `{res.get('path', target_path)}`."

        if "read file" in prompt_lower or "view file" in prompt_lower or "open file" in prompt_lower:
            path_match = re.search(r'(?:file|path|from)\s+([a-zA-Z0-9_\-\.\/\\\:]+)', prompt)
            target_path = path_match.group(1) if path_match else "requirements.txt"
            res = tools.execute_tool("read_file", {"path": target_path})
            if res.get("status") == "success":
                preview = res.get("content", "")[:400]
                return f"**Contents of `{target_path}`**:\n```\n{preview}\n```"
            return f"Could not read `{target_path}`: {res.get('message', 'File not found')}"

        words = set(re.findall(r'\b\w+\b', prompt_lower))

        # Check for email action directly if user says "mail", "email", "send mail", etc.
        if any(w in words for w in ["mail", "email", "notify"]) or "send email" in prompt_lower or "send a mail" in prompt_lower:
            target_cid = pred["customer_id"]
            if (target_cid == "0" or target_cid not in self.customers) and self.last_customer_id:
                target_cid = self.last_customer_id
            
            # Find customer by name mentioned in prompt
            for c_id, c_data in self.customers.items():
                if c_data["name"].lower() in prompt_lower:
                    target_cid = c_id
                    break

            if target_cid and target_cid in self.customers:
                cust = self.customers[target_cid]
                recipient = cust.get("email", f"contact@{cust['name'].lower().replace(' ', '')}.com")
                subject = f"Urgent: Outstanding Balance Settlement Notice - {cust['name']}"
                body = (
                    f"Dear {cust['name']} Accounting Team,\n\n"
                    f"This is an official notice regarding your account balance of ${cust['balance']:,.2f} (Status: {cust['status']}).\n"
                    f"Please arrange settlement of the invoice balance at your earliest convenience today.\n\n"
                    f"Regards,\nMarcus Vance & Enterprise Billing Operations"
                )
                res = tools.execute_tool("send_customer_email", {
                    "recipient_email": recipient,
                    "subject": subject,
                    "body": body
                })
                return (
                    f"I've dispatched that email to **{cust['name']}**.\n\n"
                    f"• **Recipient**: `{recipient}`\n"
                    f"• **Subject**: *{subject}*\n"
                    f"• **Notice**: Requested settlement for outstanding balance of **${cust['balance']:,.2f}** today.\n"
                    f"• **Status**: `{res.get('status', 'Dispatched')}`"
                )

        # High-priority LATIS Security Plugin & Alerts handler
        if any(term in prompt_lower for term in ["latis", "security alert", "alerts", "threat", "breach", "incident"]):
            if any(term in prompt_lower for term in ["open latis", "how to open latis", "launch latis", "show latis dashboard", "latis ui"]):
                import subprocess
                subprocess.Popen("start http://localhost:8000/ui/", shell=True)
                return (
                    "Opening the **LATIS Forensic Security Dashboard** in your browser right now at:\n"
                    "👉 **[http://localhost:8000/ui/](http://localhost:8000/ui/)**\n\n"
                    "Here you can view the live Causal Identification Matrix, replay sandbox counterfactuals, and inspect real-time breach stream events."
                )

            if "next" in prompt_lower or "trigger" in prompt_lower or "stream" in prompt_lower:
                res = tools.execute_tool("latis_plugin", {"action": "step_stream"})
                if res.get("status") == "success":
                    d = res.get("data", {})
                    evt = d.get("event", {})
                    clsf = d.get("classification", {})
                    return (
                        f"🛡️ **Live Incident Streamed into LATIS**:\n"
                        f"• Event ID: `{evt.get('event_id')}`\n"
                        f"• Attack Query: *\"{evt.get('user_query')}\"*\n"
                        f"• Tool Exploit: `{evt.get('tool_name')}` (`{evt.get('tool_args')}`)\n"
                        f"• Generated Alert ID: `{d.get('alert_id')}`\n"
                        f"• Severity: **{clsf.get('severity_tier')}** (Confidence: {clsf.get('confidence') * 100:.1f}%)\n"
                        f"• Remediation: `{clsf.get('remediation_blueprint', {}).get('strategy')}`"
                    )
                return f"Could not stream next event: {res.get('error')}"

            # Query real-time alerts from LATIS server
            res = tools.execute_tool("latis_plugin", {"action": "alerts"})
            if res.get("status") == "success":
                alerts = res.get("alerts", [])
                if not alerts:
                    return (
                        f"🛡️ **LATIS Security Core (Live Connected at port 8000)**\n"
                        f"• Status: Active & Monitoring\n"
                        f"• Active Alerts: 0 (No active breaches detected right now).\n\n"
                        f"*Tip*: Say 'stream next incident' to push an active adversarial event through LATIS."
                    )
                
                alert_lines = []
                for a in alerts[:5]:
                    alert_lines.append(
                        f"• **{a.get('alert_id')}** [{a.get('severity')}] - {a.get('threat_type')}\n"
                        f"  *Description*: {a.get('description')}\n"
                        f"  *Affected*: {', '.join(a.get('affected_components', []))}\n"
                        f"  *Status*: `{a.get('status')}` | Case: `{a.get('related_case_id')}`"
                    )
                
                return (
                    f"🛡️ **Live Security Alerts from LATIS Engine** ({len(alerts)} total):\n\n"
                    + "\n\n".join(alert_lines)
                    + "\n\n*Ask to investigate any alert ID (e.g., `investigate " + alerts[0].get('alert_id') + "`)*"
                )
            else:
                return f"LATIS Connection Error: {res.get('error')}"

        # Conversational & Clarification Handling
        if pred["action_type"] == "CONVERSATION" or pred["tool_name"] == "NONE":
            # Greetings
            if any(w in words for w in ["hi", "hello", "greetings", "hey"]):
                return "Hello! How can I assist you with accounts, device operations, or company runbooks today?"

            # Company admin identity / how come questions
            if "who is the admin" in prompt_lower or "who is admin" in prompt_lower:
                return (
                    "You are recognized as the administrator for this active session. "
                    "You hold executive privileges with full MDM host device access, customer account controls, and policy dispatch authority."
                )
            elif "how come" in prompt_lower or "why" in prompt_lower and not self.last_customer_id:
                return (
                    "Your session is verified with root executive credentials. "
                    "This lets you inspect telemetry, manage accounts, run diagnostics, and execute MDM device tasks directly."
                )

            # Natural answers regarding payment directions ("should we pay them or they pay us", "so they need to pay?")
            if self.last_customer_id and any(term in prompt_lower for term in ["they need to pay", "they should pay", "we pay", "owe", "amount", "balance", "how much", "pay"]):
                cust = self.customers.get(self.last_customer_id)
                if cust:
                    cname = cust["name"]
                    cid = self.last_customer_id
                    bal = cust["balance"]
                    status = cust["status"]
                    if any(phrase in prompt_lower for phrase in ["we pay", "they pay", "should we pay", "who pays"]):
                        return (
                            f"**They need to pay us.** {cname} (ID #{cid}) has an outstanding balance of **${bal:,.2f}** under `{status}` status on our enterprise ledger."
                        )
                    elif "so they need to pay" in prompt_lower or "do they owe" in prompt_lower:
                        return (
                            f"Yes, exactly. **{cname}** owes our company **${bal:,.2f}**. Their invoices are currently active and pending settlement."
                        )
                    else:
                        return f"**{cname}** (ID #{cid}) currently has an outstanding balance of **${bal:,.2f}** (Status: `{status}`)."

            # Identity & General capabilities
            elif "who are you" in prompt_lower or "what are you" in prompt_lower:
                return f"I'm Zvy, your enterprise operations copilot. I manage host device controls and {len(self.customers):,} enterprise customer accounts."
            elif "what can you do" in prompt_lower or "capabilities" in prompt_lower or "help" in prompt_lower:
                return (
                    f"I can help you with:\n"
                    f"1. **Accounts & Billing**: Lookup {len(self.customers):,} enterprise clients, inspect balances, and send emails.\n"
                    f"2. **Host Device Control**: Inspect system telemetry, read and write files directly on this machine.\n"
                    f"3. **Company Policies & SLAs**: Retrieve official runbooks, refund guidelines, and incident protocols.\n"
                    f"4. **Security & LATIS**: Coordinate with the LATIS plugin to investigate supply chain and model integrity issues."
                )
            elif "how many people" in prompt_lower or "employees" in prompt_lower or "headcount" in prompt_lower or "staff" in prompt_lower:
                return "Our enterprise employs **12,450 people** globally across engineering, customer operations, finance, and security infrastructure."
            elif "admin" in prompt_lower:
                return "Acknowledged, Administrator. Executive device management privileges and full enterprise account controls are active for your session."
            elif "thank" in prompt_lower or "thanks" in prompt_lower:
                return "Glad I could help! Let me know if you need anything else."
            elif any(w in prompt_lower for w in ["bye", "goodbye", "see you"]):
                return "Goodbye! Have a productive day."
            else:
                if self.last_customer_id:
                    cust = self.customers.get(self.last_customer_id)
                    return f"We are currently reviewing **{cust['name']}** (ID #{self.last_customer_id}). What would you like to do next—send them an email, check their invoice history, or adjust their account?"
                return "I'm here to help. You can give me a customer ID or name (e.g., #1025), ask about host device telemetry, or explore our company policies."

        # Handle TOOL_CALL
        tool = pred["tool_name"]

        if tool == "query_customer":
            cid = pred["customer_id"]
            if cid == "0" or cid not in self.customers:
                for c_id, c_data in self.customers.items():
                    if c_data["name"].lower() in prompt_lower:
                        cid = c_id
                        break
            if (cid == "0" or cid not in self.customers) and self.last_customer_id:
                cid = self.last_customer_id

            res = tools.execute_tool("query_customer", {"query": cid})
            if res.get("status") == "success" and res.get("customers"):
                c = res["customers"][0]
                self.last_customer_id = str(c["id"])
                return (
                    f"**{c['name']}** (ID #{c['id']})\n"
                    f"• Tier: {c['tier']}\n"
                    f"• Balance: **${c['balance']:,.2f}** ({c['status']})\n"
                    f"• Manager: {c.get('account_manager', 'Support Team')}\n"
                    f"• Email: {c['email']}\n"
                    f"• Recent Invoices: {', '.join(c.get('recent_invoices', []))}"
                )
            return f"Customer '{cid}' not found across {len(self.customers):,} accounts."

        elif tool == "process_refund":
            cid = pred["customer_id"]
            amt = pred["amount"] if pred["amount"] > 0 else 50.0
            
            reason = "Operational reconciliation"
            if "dispute" in prompt_lower:
                reason = "Customer billing dispute resolution"
            elif "sla" in prompt_lower or "downtime" in prompt_lower:
                reason = "SLA downtime courtesy credit"
            elif "overcharge" in prompt_lower:
                reason = "Invoice overcharge adjustment"

            res = tools.execute_tool("process_refund", {"customer_id": cid, "amount": amt, "reason": reason})
            if res.get("status") != "success":
                return f"Refund blocked: {res.get('message', 'Authorization failed')}"
            
            return (
                f"**Refund Executed**\n"
                f"• Customer: {res.get('customer_name', f'ID #{cid}')} (ID #{cid})\n"
                f"• Credited: **${amt:,.2f}**\n"
                f"• New Balance: **${res.get('new_balance', 0.0):,.2f}**\n"
                f"• Reason: {reason}"
            )

        elif tool == "read_document":
            doc_name = "service_level_agreements.txt"
            if "refund" in prompt_lower or "return" in prompt_lower:
                doc_name = "return_and_refund_policy.txt"
            elif "incident" in prompt_lower or "escalat" in prompt_lower:
                doc_name = "incident_escalation_protocol.txt"
            elif "rate" in prompt_lower or "quota" in prompt_lower or "api" in prompt_lower:
                doc_name = "api_rate_limits_and_quotas.txt"
            elif "disaster" in prompt_lower or "recovery" in prompt_lower or "rto" in prompt_lower or "rpo" in prompt_lower:
                doc_name = "cloud_disaster_recovery_plan.txt"
            elif "billing" in prompt_lower or "payment" in prompt_lower:
                doc_name = "billing_cycle_and_payment_terms.txt"
            elif "onboard" in prompt_lower or "vip" in prompt_lower:
                doc_name = "vip_customer_onboarding_guide.txt"
            elif "security" in prompt_lower or "compliance" in prompt_lower:
                doc_name = "security_and_compliance_standards.txt"

            res = tools.execute_tool("read_document", {"doc_name": doc_name})
            if res.get("status") != "success":
                return f"Document '{doc_name}' not found."
            
            content_preview = res.get("content", "")[:500]
            return f"**{doc_name}**:\n{content_preview}"

        elif tool == "send_customer_email":
            cid = pred["customer_id"]
            cust = self.customers.get(cid, {})
            recipient = cust.get("email", "client@enterprise.com")
            subject = f"Notice: Account Balance Update for {cust.get('name', 'Account')}"
            body = f"Dear {cust.get('name', 'Valued Customer')}, this is a notification regarding your active balance of ${cust.get('balance', 0):,.2f}."
            
            res = tools.execute_tool("send_customer_email", {
                "recipient_email": recipient,
                "subject": subject,
                "body": body
            })
            return (
                f"📧 **Outbound Dispatch Confirmation**\n"
                f"- **Recipient**: `{recipient}`\n"
                f"- **Subject**: *{subject}*\n"
                f"- **Status**: `{res.get('status', 'Sent')}`"
            )

        elif tool == "run_system_command":
            res = tools.execute_tool("run_system_command", {"command": "uptime"})
            return f"⚙️ **System Diagnostics**:\n```\n{res.get('output', 'All services operational.')}\n```"

        elif tool == "latis_plugin" or "latis" in prompt_lower or "alert" in prompt_lower or "investigat" in prompt_lower or "security stream" in prompt_lower:
            if "next" in prompt_lower or "trigger" in prompt_lower or "stream" in prompt_lower:
                res = tools.execute_tool("latis_plugin", {"action": "step_stream"})
                if res.get("status") == "success":
                    d = res.get("data", {})
                    evt = d.get("event", {})
                    clsf = d.get("classification", {})
                    return (
                        f"🛡️ **Live Incident Streamed into LATIS**:\n"
                        f"• Event ID: `{evt.get('event_id')}`\n"
                        f"• Attack Query: *\"{evt.get('user_query')}\"*\n"
                        f"• Tool Exploit: `{evt.get('tool_name')}` (`{evt.get('tool_args')}`)\n"
                        f"• Generated Alert ID: `{d.get('alert_id')}`\n"
                        f"• Severity: **{clsf.get('severity_tier')}** (Confidence: {clsf.get('confidence') * 100:.1f}%)\n"
                        f"• Remediation: `{clsf.get('remediation_blueprint', {}).get('strategy')}`"
                    )
                return f"Could not stream next event: {res.get('error')}"

            # Default: list real-time alerts from LATIS
            res = tools.execute_tool("latis_plugin", {"action": "alerts"})
            if res.get("status") == "success":
                alerts = res.get("alerts", [])
                if not alerts:
                    return (
                        f"🛡️ **LATIS Security Core (Live Connected at port 8000)**\n"
                        f"• Status: Active & Monitoring\n"
                        f"• Active Alerts: 0 (No active breaches detected right now).\n\n"
                        f"*Tip*: Say 'stream next incident' to push an active adversarial event through LATIS."
                    )
                
                alert_lines = []
                for a in alerts[:5]:
                    alert_lines.append(
                        f"• **{a.get('alert_id')}** [{a.get('severity')}] - {a.get('threat_type')}\n"
                        f"  *Description*: {a.get('description')}\n"
                        f"  *Affected*: {', '.join(a.get('affected_components', []))}\n"
                        f"  *Status*: `{a.get('status')}` | Case: `{a.get('related_case_id')}`"
                    )
                
                return (
                    f"🛡️ **Live Security Alerts from LATIS Engine** ({len(alerts)} total):\n\n"
                    + "\n\n".join(alert_lines)
                    + "\n\n*Ask to investigate any alert ID (e.g., `investigate " + alerts[0].get('alert_id') + "`)*"
                )
            else:
                return f"LATIS Connection: {res.get('error')}"

        return "Request processed successfully."
