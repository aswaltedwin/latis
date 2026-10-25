"""
Zvy Tools Engine
Provides enterprise operations and customer support functions:
- File & Document reading (Dynamic loader from zvy_agent/data/documents/)
- Customer & Billing Database querying (Dynamic loader from zvy_agent/data/customers.json)
- Financial & Billing Transactions (process_refund)
- Outbound Communication Dispatch (send_customer_email)
- System diagnostic execution (run_system_command)
"""

import os
import json
import time
import subprocess
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
CUSTOMERS_FILE = os.path.join(DATA_DIR, "customers.json")
OUTBOUND_FILE = os.path.join(DATA_DIR, "outbound_audit.json")

def load_customers_db() -> Dict[str, Any]:
    """Loads customers from JSON file on disk."""
    if os.path.exists(CUSTOMERS_FILE):
        try:
            with open(CUSTOMERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_customers_db(data: Dict[str, Any]):
    """Saves updated customers back to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CUSTOMERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_documents_list() -> Dict[str, str]:
    """Scans and reads all document files from zvy_agent/data/documents/."""
    docs = {}
    if os.path.exists(DOCUMENTS_DIR):
        for fname in os.listdir(DOCUMENTS_DIR):
            fpath = os.path.join(DOCUMENTS_DIR, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        docs[fname] = f.read()
                except Exception:
                    pass
    return docs

def load_outbound_audit() -> List[Dict[str, Any]]:
    """Loads outbound communication records from disk."""
    if os.path.exists(OUTBOUND_FILE):
        try:
            with open(OUTBOUND_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def record_outbound_email(record: Dict[str, Any]):
    """Appends an outbound message to disk."""
    records = load_outbound_audit()
    records.append(record)
    with open(OUTBOUND_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

# Initialize in-memory cache
CUSTOMERS_DB = load_customers_db()
KNOWLEDGE_DOCS = load_documents_list()
OUTBOUND_MESSAGES_LOG = load_outbound_audit()

ZVY_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Reads an internal enterprise document or policy from the corporate knowledge library",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_name": {
                        "type": "string",
                        "description": "Name or topic of the document to read (e.g. return_and_refund_policy.txt, service_level_agreements.txt, incident_escalation_protocol.txt)"
                    }
                },
                "required": ["doc_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_customer",
            "description": "Queries customer account profile, tier, active balance, or API keys by customer ID or company name",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Customer ID (e.g. 101, 102, 104) or company name (e.g. Acme, Nexus, Starlight)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Processes a monetary refund or credit balance adjustment for a customer account",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID (e.g. 101, 102)"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Dollar amount to credit or refund"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Justification for refund"
                    }
                },
                "required": ["customer_id", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_customer_email",
            "description": "Drafts and sends an official notification email to an enterprise contact or partner",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_email": {
                        "type": "string",
                        "description": "Destination email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": "Content of the message"
                    }
                },
                "required": ["recipient_email", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_system_command",
            "description": "Executes diagnostic terminal commands or shell operations on the host system",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command or system utility to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes, saves, or creates a file on the local host filesystem within authorized company workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute file path to create/save"},
                    "content": {"type": "string", "description": "Content of the file to save"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads raw contents of any file on the local device host",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_telemetry",
            "description": "Gathers live device status, CPU/RAM utilization, storage space, and host ownership details",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "latis_plugin",
            "description": "Executes LATIS autonomous security incident investigation, alert causality analysis, or forensic telemetry audit",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "list_cases", "investigate", "run_experiment", "forensic_report"],
                        "description": "LATIS operational action to trigger"
                    },
                    "target_case_id": {
                        "type": "string",
                        "description": "Incident case ID (e.g. PV2-CASE-01)"
                    }
                },
                "required": ["action"]
            }
        }
    }
]

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Executes the requested tool and returns structured output."""
    global CUSTOMERS_DB, KNOWLEDGE_DOCS, OUTBOUND_MESSAGES_LOG
    
    # Reload from disk for real-time consistency
    CUSTOMERS_DB = load_customers_db()
    KNOWLEDGE_DOCS = load_documents_list()

    try:
        if tool_name == "read_document":
            doc_name = arguments.get("doc_name", "").strip().lower()
            # Fuzzy match across filenames
            for fname, content in KNOWLEDGE_DOCS.items():
                if doc_name in fname.lower() or any(w in fname.lower() for w in doc_name.split() if len(w) > 3):
                    return {"status": "success", "file": fname, "content": content}
            return {
                "status": "error",
                "message": f"Document '{doc_name}' not found.",
                "available_documents": list(KNOWLEDGE_DOCS.keys())
            }

        elif tool_name == "query_customer":
            query = str(arguments.get("query", "")).lower().strip()
            results = []
            for cid, cdata in CUSTOMERS_DB.items():
                if query in cid.lower() or query in cdata["name"].lower():
                    results.append(cdata)
            if results:
                return {"status": "success", "count": len(results), "customers": results}
            return {"status": "not_found", "message": f"No customer matching '{query}' found. Total registered: {len(CUSTOMERS_DB)}"}

        elif tool_name == "process_refund":
            cid = str(arguments.get("customer_id", "")).strip()
            amount = float(arguments.get("amount", 0.0))
            reason = arguments.get("reason", "Customer requested balance adjustment")
            
            if cid in CUSTOMERS_DB:
                c = CUSTOMERS_DB[cid]
                c["balance"] = max(0.0, c["balance"] - amount)
                save_customers_db(CUSTOMERS_DB)
                return {
                    "status": "success",
                    "customer_id": cid,
                    "customer_name": c["name"],
                    "refunded_amount": amount,
                    "new_balance": c["balance"],
                    "reason": reason
                }
            return {"status": "error", "message": f"Customer ID {cid} not found"}

        elif tool_name == "send_customer_email":
            rec = arguments.get("recipient_email", "")
            sub = arguments.get("subject", "")
            body = arguments.get("body", "")
            record = {
                "timestamp": time.time(),
                "recipient": rec,
                "subject": sub,
                "body_preview": body[:120],
                "status": "DISPATCHED"
            }
            record_outbound_email(record)
            OUTBOUND_MESSAGES_LOG = load_outbound_audit()
            return {
                "status": "success",
                "message": f"Notification email sent to {rec}",
                "subject": sub
            }

        elif tool_name == "write_file":
            path = arguments.get("path", "").strip()
            content = arguments.get("content", "")
            
            # Security guard: prevent writing to windows root system files
            if any(forbidden in path.lower() for forbidden in ["system32", "windows", "boot.ini"]):
                return {"status": "blocked", "message": "Access Denied: Protected OS system file write attempt blocked by Zvy MDM."}
            
            # Relative to project workspace if not absolute
            target_path = path if os.path.isabs(path) else os.path.join(os.path.dirname(BASE_DIR), path)
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return {
                "status": "success",
                "message": f"File successfully written and saved to device: {path}",
                "bytes_written": len(content.encode("utf-8")),
                "path": target_path
            }

        elif tool_name == "read_file":
            path = arguments.get("path", "").strip()
            target_path = path if os.path.isabs(path) else os.path.join(os.path.dirname(BASE_DIR), path)
            if not os.path.exists(target_path):
                return {"status": "error", "message": f"File not found on device: {path}"}
            
            with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read(4000) # read up to 4000 chars
            return {
                "status": "success",
                "path": path,
                "content": data
            }

        elif tool_name == "open_application":
            app_name = arguments.get("app_name", "").strip().lower()
            # Common application mappings on Windows
            app_map = {
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "calc": "calc.exe",
                "chrome": "start chrome",
                "browser": "start msedge",
                "edge": "start msedge",
                "terminal": "start wt",
                "cmd": "start cmd",
                "powershell": "start powershell",
                "explorer": "explorer.exe",
                "settings": "start ms-settings:",
                "paint": "mspaint.exe",
                "task manager": "taskmgr.exe"
            }
            cmd = app_map.get(app_name, f"start {app_name}")
            try:
                subprocess.Popen(cmd, shell=True)
                return {"status": "success", "message": f"Opened {app_name.capitalize()} for you."}
            except Exception as e:
                return {"status": "error", "message": f"Could not launch {app_name}: {str(e)}"}

        elif tool_name == "set_timer_alarm":
            duration_str = arguments.get("duration", "5 minutes")
            label = arguments.get("label", "Alarm / Reminder")
            # Quick Windows timer / toast notification
            cmd = f'powershell -Command "[System.Reflection.Assembly]::LoadWithPartialName(\'System.Windows.Forms\'); [System.Windows.Forms.MessageBox]::Show(\'Reminder: {label}\', \'Zvy Assistant\')"'
            try:
                # Launch asynchronously
                subprocess.Popen(f'powershell -Command "Start-Sleep -Seconds 3; {cmd}"', shell=True)
                return {"status": "success", "duration": duration_str, "label": label}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        elif tool_name == "get_device_status":
            import platform
            from datetime import datetime
            now_str = datetime.now().strftime("%I:%M %p, %A, %B %d, %Y")
            try:
                import psutil
                battery_sensor = psutil.sensors_battery()
                if battery_sensor:
                    plugged = "charging" if battery_sensor.power_plugged else "on battery"
                    battery = f"{battery_sensor.percent}% ({plugged})"
                else:
                    battery = "Desktop (no battery)"
                cpu = f"{psutil.cpu_percent(interval=0.1)}%"
                ram = f"{psutil.virtual_memory().percent}% used ({round(psutil.virtual_memory().total / (1024**3), 1)} GB total)"
            except Exception:
                battery = "Normal"
                cpu = "Normal"
                ram = "Normal"

            return {
                "status": "success",
                "current_time": now_str,
                "battery": battery,
                "cpu": cpu,
                "ram": ram,
                "device_name": platform.node(),
                "os": f"{platform.system()} {platform.release()}"
            }

        elif tool_name == "find_device_files":
            query = arguments.get("query", "").strip()
            search_dirs = [
                os.path.expanduser("~"),
                os.path.join(os.path.expanduser("~"), "Downloads"),
                os.path.join(os.path.expanduser("~"), "Documents"),
                os.path.join(os.path.expanduser("~"), "Desktop"),
                BASE_DIR,
                os.path.dirname(BASE_DIR)
            ]
            matches = []
            seen = set()
            terms = [t.lower() for t in query.split() if t.strip()]

            for sdir in search_dirs:
                if not os.path.exists(sdir):
                    continue
                try:
                    for root, dirs, files in os.walk(sdir):
                        # Avoid deeply nested node_modules, git, and venvs
                        dirs[:] = [d for d in dirs if d not in [".git", "node_modules", ".venv", "venv", "__pycache__", "AppData"]]
                        for f in files:
                            fname_lower = f.lower()
                            if all(t in fname_lower for t in terms) or (query.lower() in fname_lower):
                                full_p = os.path.join(root, f)
                                if full_p not in seen:
                                    seen.add(full_p)
                                    size_kb = round(os.path.getsize(full_p) / 1024, 1)
                                    matches.append({"name": f, "path": full_p, "size_kb": size_kb})
                                    if len(matches) >= 8:
                                        break
                        if len(matches) >= 8:
                            break
                except Exception:
                    continue

            return {
                "status": "success",
                "query": query,
                "count": len(matches),
                "matches": matches
            }

        elif tool_name == "run_system_command":
            cmd = arguments.get("command", "")
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            output = res.stdout if res.stdout else res.stderr
            return {
                "status": "success" if res.returncode == 0 else "failed",
                "command": cmd,
                "exit_code": res.returncode,
                "output": output.strip() or "(Command completed with no output)"
            }
        elif tool_name == "latis_plugin":
            action = arguments.get("action", "alerts")
            target_id = arguments.get("target_id") or arguments.get("target_case_id") or ""
            import urllib.request
            import json

            LATIS_HOST = "http://localhost:8000"

            try:
                if action in ["alerts", "status", "list_alerts"]:
                    url = f"{LATIS_HOST}/api/alerts"
                    req = urllib.request.Request(url, headers={"User-Agent": "Zvy-Plugin"})
                    with urllib.request.urlopen(req, timeout=4) as response:
                        data = json.loads(response.read().decode())
                    return {
                        "status": "success",
                        "connected_to": LATIS_HOST,
                        "count": data.get("count", 0),
                        "alerts": data.get("alerts", [])
                    }

                elif action in ["next_event", "step_stream", "trigger_incident"]:
                    url = f"{LATIS_HOST}/api/stream/next"
                    req = urllib.request.Request(url, headers={"User-Agent": "Zvy-Plugin"})
                    with urllib.request.urlopen(req, timeout=4) as response:
                        data = json.loads(response.read().decode())
                    return {
                        "status": "success",
                        "source": "LATIS Live Threat Stream",
                        "data": data
                    }

                elif action in ["investigate", "forensic_report", "case"]:
                    # Fetch case state or trigger investigation on alert
                    if target_id.startswith("ALT-"):
                        url = f"{LATIS_HOST}/api/alerts/{target_id}/investigate"
                        req = urllib.request.Request(url, method="POST", headers={"User-Agent": "Zvy-Plugin"})
                    else:
                        case_id = target_id if target_id else "CASE-PV2-001"
                        url = f"{LATIS_HOST}/api/cases/{case_id}/state"
                        req = urllib.request.Request(url, headers={"User-Agent": "Zvy-Plugin"})

                    with urllib.request.urlopen(req, timeout=5) as response:
                        data = json.loads(response.read().decode())
                    return {
                        "status": "success",
                        "investigation": data
                    }

                else:
                    return {"status": "error", "message": f"Unknown LATIS action: {action}"}

            except Exception as ex:
                return {
                    "status": "error",
                    "plugin": "LATIS Core",
                    "error": f"Failed to connect to live LATIS engine at {LATIS_HOST}: {str(ex)}"
                }

        else:
            return {"status": "unknown_tool", "tool": tool_name}

    except Exception as e:
        return {"status": "execution_error", "error": str(e)}
