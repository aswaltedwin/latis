"""
Zvy - Enterprise Operations & Support Copilot
Hybrid Aesthetic:
- Claude (Warm Editorial Serif typography, clean breathing space, human tone)
- Raycast (âŒ˜K Command Palette, hotkey badges, sub-millisecond keyboard shortcuts, frosted blur glass)
- Notion (Document page hierarchy, collapsible block aesthetics, clean minimal layout)
Features:
- Real Raycast âŒ˜K Spotlight Command Palette (Type anything to instant search accounts, policies, or switch chats)
- ChatGPT-style '+' attachment popup (LATIS Security plugin, documents, image receipts)
- Full Markdown parsing with marked.js (real bold, code highlights, bullet points)
- Clean Dark Mode & Light Mode (persisted in localStorage)
- Settings modal
- Multi-Chat conversation history with quick keyboard shortcuts
"""

import time
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from zvy_agent.model.company_model import ZvyCompanyModel
from zvy_agent.tools import CUSTOMERS_DB, KNOWLEDGE_DOCS, OUTBOUND_MESSAGES_LOG, execute_tool

app = FastAPI(title="Zvy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ZvyCompanyModel()

class CommandRequest(BaseModel):
    command: str
    plugins_enabled: list = []

@app.post("/api/command")
def execute_command(req: CommandRequest):
    cmd = req.command
    plugins = req.plugins_enabled or []
    latis_active = "latis" in plugins

    if latis_active and any(w in cmd.lower() for w in ["latis", "security", "telemetry", "investigat", "anomaly", "forensic", "case"]):
        action = "status"
        if "case" in cmd.lower() or "list" in cmd.lower():
            action = "list_cases"
        elif "investigat" in cmd.lower() or "experiment" in cmd.lower():
            action = "investigate"
        elif "forensic" in cmd.lower() or "report" in cmd.lower():
            action = "forensic_report"

        latis_res = execute_tool("latis_plugin", {"action": action, "target_case_id": "PV2-CASE-01"})
        
        if action == "status":
            reply = (
                f"### **LATIS Security Status**\n\n"
                f"- **Plugin Version**: `{latis_res.get('version', 'v2.4-active')}`\n"
                f"- **State**: `{latis_res.get('operational_state', 'ACTIVE')}`\n"
                f"- **Monitored Telemetry Events**: **{latis_res.get('monitored_events', 0)}**\n"
                f"- **Active Alerts**: **{latis_res.get('active_alerts', 0)}**\n"
                f"- **Benchmark Cases**: `{', '.join(latis_res.get('available_cases', []))}`"
            )
        elif action == "list_cases":
            cases_md = "\n".join([f"- **{c['id']}**: {c['name']} (Severity: `{c['severity']}`)" for c in latis_res.get("cases", [])])
            reply = (
                f"### **LATIS Incident Cases** ({latis_res.get('total_cases', 0)} loaded)\n\n"
                f"{cases_md}\n\n"
                f"Tip: Ask me to *'Investigate PV2-CASE-01'* to run causal analysis."
            )
        else:
            reply = (
                f"### **LATIS Investigation: {latis_res.get('case_id')}**\n\n"
                f"- **Incident**: {latis_res.get('incident_name')}\n"
                f"- **Root Cause**: **{latis_res.get('causal_root')}**\n"
                f"- **Confidence**: `{latis_res.get('confidence_score') * 100:.1f}%`\n"
                f"- **Mitigation**: {latis_res.get('recommended_mitigation')}"
            )

        return JSONResponse({
            "model": "Zvy_Model",
            "response": reply,
            "tools_called": [{"tool": "latis_plugin", "arguments": {"action": action}, "status": "COMPLETED"}],
            "latency_ms": 16.0,
            "plugin_used": "latis"
        })

    # Standard concise model prediction & tool execution
    res = model.predict_and_act(cmd)
    res["model"] = "Zvy_Model"
    return JSONResponse(res)

@app.post("/api/reset")
def reset_conversation():
    model.reset()
    return JSONResponse({"status": "reset_complete"})

@app.get("/api/customers")
def get_customers():
    return JSONResponse(list(CUSTOMERS_DB.values()))

@app.get("/api/documents")
def get_documents():
    return JSONResponse(list(KNOWLEDGE_DOCS.keys()))

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zvy</title>
    <!-- Fonts: Claude Editorial (Newsreader) + Raycast Inter + Mono (JetBrains Mono) -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <!-- Marked.js for markdown rendering -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            /* Raycast + Claude Dark Theme */
            --bg-canvas: #0B0C0E;
            --bg-sidebar: #111216;
            --bg-card: #17181F;
            --bg-card-hover: #1E202A;
            --bg-input: #14151B;
            --bg-popover: rgba(22, 23, 30, 0.94);
            --bg-command: rgba(18, 19, 25, 0.96);
            
            --border-hairline: rgba(255, 255, 255, 0.08);
            --border-subtle: rgba(255, 255, 255, 0.15);
            --border-highlight: rgba(255, 255, 255, 0.25);
            
            --text-heading: #F5F5F0;
            --text-body: #D4D4DE;
            --text-secondary: #8E929E;
            --text-muted: #5D606C;
            
            --accent-warm: #D97757;
            --accent-glow: rgba(217, 119, 87, 0.22);
            --raycast-red: #FF6363;
            --accent-violet: #6366F1;
            --green-active: #22C55E;
            
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-serif: 'Newsreader', Georgia, serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        [data-theme="light"] {
            /* Notion Clean Paper Light Theme */
            --bg-canvas: #FFFFFF;
            --bg-sidebar: #F7F7F5;
            --bg-card: #F1F1EF;
            --bg-card-hover: #E9E9E7;
            --bg-input: #FFFFFF;
            --bg-popover: rgba(255, 255, 255, 0.96);
            --bg-command: rgba(255, 255, 255, 0.98);
            
            --border-hairline: rgba(0, 0, 0, 0.07);
            --border-subtle: rgba(0, 0, 0, 0.14);
            --border-highlight: rgba(0, 0, 0, 0.22);
            
            --text-heading: #191919;
            --text-body: #37352F;
            --text-secondary: #787774;
            --text-muted: #9B9A97;
            
            --accent-warm: #C96645;
            --accent-glow: rgba(201, 102, 69, 0.16);
            --accent-violet: #4F46E5;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-sans);
            background: var(--bg-canvas);
            color: var(--text-body);
            height: 100vh;
            display: flex;
            overflow: hidden;
            font-size: 13.5px;
            letter-spacing: -0.01em;
            -webkit-font-smoothing: antialiased;
            transition: background 0.2s ease, color 0.2s ease;
        }

        /* 1. Left Notion-Style Navigation Sidebar */
        .sidebar {
            width: 260px;
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border-hairline);
            display: flex;
            flex-direction: column;
            user-select: none;
            transition: background 0.2s ease;
        }
        .sidebar-brand {
            padding: 14px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand-title-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .brand-symbol {
            width: 24px;
            height: 24px;
            border-radius: 6px;
            background: linear-gradient(135deg, var(--accent-warm), #C15F3E);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-serif);
            font-weight: 500;
            font-size: 15px;
            color: #FFF;
        }
        .brand-name {
            font-size: 13.5px;
            font-weight: 600;
            color: var(--text-heading);
            letter-spacing: -0.02em;
        }

        /* Raycast Quick Search Bar in Sidebar */
        .search-trigger {
            margin: 6px 12px 10px 12px;
            padding: 7px 10px;
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            border-radius: 7px;
            color: var(--text-secondary);
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.15s;
        }
        .search-trigger:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-subtle);
            color: var(--text-heading);
        }
        .kbd-pill {
            font-family: var(--font-mono);
            font-size: 10px;
            background: var(--bg-canvas);
            border: 1px solid var(--border-hairline);
            padding: 1px 5px;
            border-radius: 4px;
            color: var(--text-muted);
        }

        /* Notion-style Page/Chat Sections */
        .sidebar-section-title {
            padding: 12px 16px 4px;
            font-size: 10.5px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .chat-history-list {
            list-style: none;
            padding: 0 8px;
            overflow-y: auto;
            flex: 1;
        }
        .chat-history-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 8px;
            border-radius: 6px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 12.5px;
            transition: all 0.12s ease;
            margin-bottom: 2px;
        }
        .chat-history-item:hover {
            background: var(--bg-card-hover);
            color: var(--text-heading);
        }
        .chat-history-item.active {
            background: var(--bg-card);
            color: var(--text-heading);
            font-weight: 500;
        }
        .chat-history-item .chat-title {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 185px;
            display: flex;
            align-items: center;
            gap: 7px;
        }
        .chat-history-item .btn-del {
            opacity: 0;
            color: var(--text-muted);
            font-size: 11px;
            padding: 2px 4px;
            border-radius: 4px;
            transition: all 0.15s;
        }
        .chat-history-item:hover .btn-del {
            opacity: 0.7;
        }
        .chat-history-item .btn-del:hover {
            color: var(--raycast-red);
        }

        /* Sidebar Bottom Actions */
        .sidebar-bottom {
            padding: 10px 12px;
            border-top: 1px solid var(--border-hairline);
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .bottom-action-btn {
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 6px 8px;
            border-radius: 6px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 12.5px;
            transition: all 0.12s;
        }
        .bottom-action-btn:hover {
            background: var(--bg-card-hover);
            color: var(--text-heading);
        }

        /* 2. Main Stage (Claude Reading Canvas) */
        .main-stage {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-canvas);
            overflow: hidden;
            position: relative;
        }

        /* Top Command Header */
        .top-command-bar {
            height: 48px;
            border-bottom: 1px solid var(--border-hairline);
            padding: 0 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-canvas);
        }
        .chat-header-title {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .chat-header-title span.current {
            color: var(--text-heading);
            font-weight: 600;
        }

        .model-badge {
            font-size: 11.5px;
            font-weight: 600;
            font-family: var(--font-mono);
            color: var(--text-heading);
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            padding: 3px 9px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .model-badge .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--green-active);
        }

        /* Conversational Container (Claude Warm Typography) */
        .chat-flow {
            flex: 1;
            overflow-y: auto;
            padding: 36px 28px 24px 28px;
            display: flex;
            flex-direction: column;
            gap: 26px;
            max-width: 820px;
            width: 100%;
            margin: 0 auto;
        }
        
        .message-row {
            display: flex;
            flex-direction: column;
            gap: 6px;
            animation: fadeIn 0.12s ease-out;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }

        .user-message-row {
            align-items: flex-end;
        }
        .agent-message-row {
            align-items: flex-start;
        }

        .user-bubble {
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            color: var(--text-heading);
            padding: 10px 16px;
            border-radius: 12px 12px 2px 12px;
            font-size: 14px;
            line-height: 1.5;
            max-width: 82%;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
        }

        /* Claude Editorial Agent Output */
        .agent-response-content {
            font-size: 14.5px;
            line-height: 1.68;
            color: var(--text-body);
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .agent-response-content strong {
            color: var(--text-heading);
            font-weight: 600;
        }
        .agent-response-content ul, .agent-response-content ol {
            padding-left: 20px;
        }
        .agent-response-content code {
            font-family: var(--font-mono);
            font-size: 12px;
            background: var(--bg-card);
            padding: 2px 5px;
            border-radius: 4px;
            border: 1px solid var(--border-hairline);
        }
        .agent-response-content pre {
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: var(--font-mono);
            font-size: 12px;
        }

        /* Tool Execution Cards (Notion Callout Style) */
        .tool-card {
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            border-radius: 6px;
            padding: 5px 9px;
            display: flex;
            align-items: center;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
            width: fit-content;
        }
        .tool-card .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent-violet);
            display: inline-block;
            margin-right: 6px;
        }

        /* Floating Composer (Raycast Frosted Blur) */
        .composer-dock {
            padding: 12px 28px 24px 28px;
            max-width: 820px;
            width: 100%;
            margin: 0 auto;
            position: relative;
        }
        .composer-box {
            background: var(--bg-input);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
            transition: all 0.15s;
        }
        .composer-box:focus-within {
            border-color: var(--accent-warm);
            box-shadow: 0 4px 24px var(--accent-glow);
        }

        /* '+' Attachment Button */
        .btn-plus-attach {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            color: var(--text-secondary);
            font-size: 17px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s;
            flex-shrink: 0;
        }
        .btn-plus-attach:hover {
            color: var(--text-heading);
            background: var(--bg-card-hover);
        }

        /* Attachment Popover Menu */
        .attachment-menu {
            position: absolute;
            bottom: 72px;
            left: 28px;
            background: var(--bg-popover);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 6px;
            width: 240px;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
            display: none;
            flex-direction: column;
            gap: 2px;
            z-index: 100;
        }
        .attachment-menu.open {
            display: flex;
        }
        .attachment-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 7px 10px;
            border-radius: 6px;
            color: var(--text-heading);
            font-size: 12.5px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.12s;
        }
        .attachment-item:hover {
            background: var(--bg-card-hover);
        }
        .attachment-item .tag {
            margin-left: auto;
            font-size: 10px;
            font-family: var(--font-mono);
            color: var(--accent-warm);
        }

        .composer-textarea {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-heading);
            font-family: var(--font-sans);
            font-size: 14px;
            line-height: 1.45;
            resize: none;
            height: 24px;
            max-height: 120px;
        }
        .composer-textarea::placeholder {
            color: var(--text-muted);
        }

        .btn-dispatch {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: var(--accent-warm);
            border: none;
            color: #FFF;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s;
            flex-shrink: 0;
        }
        .btn-dispatch:hover {
            opacity: 0.9;
            transform: scale(1.05);
        }

        /* 3. Raycast âŒ˜K Spotlight Command Palette */
        .raycast-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(8px);
            display: none;
            align-items: flex-start;
            justify-content: center;
            padding-top: 15vh;
            z-index: 500;
        }
        .raycast-overlay.open {
            display: flex;
        }
        .raycast-modal {
            background: var(--bg-command);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-highlight);
            border-radius: 12px;
            width: 580px;
            max-width: 90%;
            box-shadow: 0 16px 50px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            animation: raycastScale 0.15s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes raycastScale { from { opacity: 0; transform: scale(0.96); } to { opacity: 1; transform: scale(1); } }

        .raycast-search-wrap {
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-hairline);
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .raycast-search-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-heading);
            font-size: 15px;
            font-family: var(--font-sans);
        }
        .raycast-results-list {
            max-height: 340px;
            overflow-y: auto;
            padding: 8px;
            list-style: none;
        }
        .raycast-section-title {
            padding: 6px 12px 4px;
            font-size: 10.5px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .raycast-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 9px 12px;
            border-radius: 7px;
            cursor: pointer;
            color: var(--text-heading);
            font-size: 13px;
            transition: all 0.1s;
        }
        .raycast-item:hover, .raycast-item.selected {
            background: var(--bg-card-hover);
        }
        .raycast-item .left {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .raycast-item .sub {
            font-size: 11.5px;
            color: var(--text-muted);
        }
        .raycast-footer {
            padding: 9px 16px;
            border-top: 1px solid var(--border-hairline);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
            background: rgba(0, 0, 0, 0.1);
        }

        /* Settings Modal */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 300;
        }
        .modal-overlay.open { display: flex; }
        .settings-card {
            background: var(--bg-sidebar);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            width: 440px;
            max-width: 90%;
            padding: 22px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .settings-title {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-heading);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .settings-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid var(--border-hairline);
            font-size: 13px;
        }
        .settings-label { font-weight: 500; color: var(--text-heading); }
        .settings-desc { font-size: 11.5px; color: var(--text-muted); margin-top: 2px; }
        .theme-select {
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            color: var(--text-heading);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            outline: none;
        }
    </style>
</head>
<body>

    <!-- Left Sidebar (Notion-Style) -->
    <aside class="sidebar">
        <div class="sidebar-brand">
            <div class="brand-title-wrap">
                <div class="brand-symbol">Z</div>
                <div class="brand-name">Zvy</div>
            </div>
            <button class="kbd-pill" onclick="createNewChat()" title="New Chat (Ctrl+N)">+</button>
        </div>

        <!-- Raycast âŒ˜K Spotlight Trigger -->
        <div class="search-trigger" onclick="openRaycast()">
            <span style="display: flex; align-items: center; gap: 6px;">
                <span>ðŸ”</span>
                <span>Quick search...</span>
            </span>
            <span class="kbd-pill">âŒ˜K</span>
        </div>

        <div class="sidebar-section-title">Conversations</div>
        <ul class="chat-history-list" id="chatHistoryList">
            <!-- Rendered from localStorage -->
        </ul>

        <!-- Bottom Actions (Theme & Settings) -->
        <div class="sidebar-bottom">
            <div class="bottom-action-btn" onclick="toggleTheme()">
                <span id="themeIcon">â˜€ï¸</span>
                <span id="themeLabel">Light mode</span>
            </div>
            <div class="bottom-action-btn" onclick="openSettings()">
                <span>âš™ï¸</span>
                <span>Settings</span>
            </div>
        </div>
    </aside>

    <!-- Main Canvas (Claude Editorial Feel) -->
    <main class="main-stage">
        <!-- Top Command Bar -->
        <div class="top-command-bar">
            <div class="chat-header-title">
                <span class="current" id="activeChatTitle">New conversation</span>
            </div>
            <div class="model-badge">
                <span class="dot"></span>
                <span>Zvy_Model</span>
            </div>
        </div>

        <!-- Chat Stream -->
        <div class="chat-flow" id="chatFlow">
            <!-- Messages rendered dynamically -->
        </div>

        <!-- Floating Composer with '+' Attachment Menu (Raycast / ChatGPT Style) -->
        <div class="composer-dock">
            <!-- Attachment Menu -->
            <div class="attachment-menu" id="attachmentMenu">
                <div class="attachment-item" onclick="selectAttachment('latis')">
                    <span style="font-size: 14px;">ðŸ›¡ï¸</span>
                    <span>LATIS Security</span>
                    <span class="tag" id="latisTag">ACTIVE</span>
                </div>
                <div class="attachment-item" onclick="selectAttachment('doc')">
                    <span style="font-size: 14px;">ðŸ“„</span>
                    <span>Add Runbook</span>
                    <span class="tag">10 Docs</span>
                </div>
                <div class="attachment-item" onclick="selectAttachment('raycast')">
                    <span style="font-size: 14px;">âš¡</span>
                    <span>Command Palette</span>
                    <span class="tag">âŒ˜K</span>
                </div>
            </div>

            <div class="composer-box">
                <button class="btn-plus-attach" onclick="toggleAttachmentMenu(event)" title="Add plugins, documents, or utilities">
                    +
                </button>
                <textarea id="composerInput" class="composer-textarea" rows="1" placeholder="Message Zvy..." onkeydown="handleKeyDown(event)"></textarea>
                <button class="btn-dispatch" id="btnDispatch" onclick="submitMessage()" title="Send">âž¤</button>
            </div>
        </div>
    </main>

    <!-- Raycast âŒ˜K Spotlight Command Palette -->
    <div class="raycast-overlay" id="raycastOverlay" onclick="closeRaycast(event)">
        <div class="raycast-modal" onclick="event.stopPropagation()">
            <div class="raycast-search-wrap">
                <span style="font-size: 16px;">âš¡</span>
                <input type="text" id="raycastInput" class="raycast-search-input" placeholder="Type a command, account name, or policy..." oninput="filterRaycast(this.value)">
                <span class="kbd-pill">ESC</span>
            </div>
            <div class="raycast-section-title">Commands & Shortcuts</div>
            <ul class="raycast-results-list" id="raycastResults">
                <!-- Populated dynamically -->
            </ul>
            <div class="raycast-footer">
                <span>Navigation: â†‘ â†“ &nbsp;&bull;&nbsp; Select: â†µ</span>
                <span style="font-family: var(--font-mono);">Raycast Spotlight</span>
            </div>
        </div>
    </div>

    <!-- Settings Modal -->
    <div class="modal-overlay" id="settingsModal" onclick="closeSettings(event)">
        <div class="settings-card" onclick="event.stopPropagation()">
            <div class="settings-title">
                <span>Settings</span>
                <span style="cursor: pointer; color: var(--text-muted);" onclick="closeSettings()">âœ•</span>
            </div>
            <div class="settings-row">
                <div>
                    <div class="settings-label">Appearance</div>
                    <div class="settings-desc">Switch between Dark and Light mode</div>
                </div>
                <select class="theme-select" id="themeSelect" onchange="setTheme(this.value)">
                    <option value="dark">Dark</option>
                    <option value="light">Light</option>
                </select>
            </div>
            <div class="settings-row">
                <div>
                    <div class="settings-label">Inference Model</div>
                    <div class="settings-desc">In-house neural architecture</div>
                </div>
                <span style="font-family: var(--font-mono); font-size: 12px; color: var(--text-heading);">Zvy_Model</span>
            </div>
            <div class="settings-row">
                <div>
                    <div class="settings-label">LATIS Security Plugin</div>
                    <div class="settings-desc">Autonomous telemetry and causal sandbox</div>
                </div>
                <input type="checkbox" id="latisSettingCheck" checked onchange="toggleLatisFromSettings(this.checked)" style="cursor: pointer;">
            </div>
        </div>
    </div>

    <script>
        let enabledPlugins = ['latis'];
        let sessions = JSON.parse(localStorage.getItem('zvy_sessions') || '[]');
        let currentSessionId = localStorage.getItem('zvy_active_session_id') || null;

        const RAYCAST_COMMANDS = [
            { title: "Query Customer Profile", desc: "Lookup balance, tier and invoices (e.g. Acme Corp #101)", prompt: "Look up customer Acme Corp (#101)" },
            { title: "Enterprise Service Level Agreements", desc: "Read guaranteed 15-min SLA terms", prompt: "What is our SLA policy?" },
            { title: "Authorize $50 Customer Refund", desc: "Process autonomous refund under $500 threshold", prompt: "Process a $50 refund for customer 101" },
            { title: "LATIS Security Plugin Status", desc: "Inspect active telemetry events and sandbox cases", prompt: "What is LATIS status?" },
            { title: "List Benchmark Incident Cases", desc: "Query SEASIS causal investigation registry", prompt: "List LATIS cases" },
            { title: "Corporate Return & Refund Policy", desc: "Verify agent refund authority boundaries", prompt: "What is our return and refund policy?" }
        ];

        function initApp() {
            // Theme setup
            const savedTheme = localStorage.getItem('zvy_theme') || 'dark';
            setTheme(savedTheme);

            // Hotkeys (âŒ˜K or Ctrl+K for Raycast, Ctrl+N for new chat)
            document.addEventListener('keydown', (e) => {
                if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
                    e.preventDefault();
                    openRaycast();
                } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'n') {
                    e.preventDefault();
                    createNewChat();
                } else if (e.key === 'Escape') {
                    closeRaycast();
                    closeSettings();
                }
            });

            if (!sessions.length) {
                const defaultSession = {
                    id: 'session_' + Date.now(),
                    title: 'New conversation',
                    messages: []
                };
                sessions.push(defaultSession);
                currentSessionId = defaultSession.id;
                saveSessions();
            } else if (!currentSessionId || !sessions.find(s => s.id === currentSessionId)) {
                currentSessionId = sessions[0].id;
            }
            renderHistoryList();
            loadCurrentSession();
        }

        // Raycast Spotlight Functions
        function openRaycast() {
            const overlay = document.getElementById('raycastOverlay');
            overlay.classList.add('open');
            const input = document.getElementById('raycastInput');
            input.value = '';
            filterRaycast('');
            setTimeout(() => input.focus(), 50);
        }

        function closeRaycast(e) {
            const overlay = document.getElementById('raycastOverlay');
            overlay.classList.remove('open');
        }

        function filterRaycast(query) {
            const q = query.toLowerCase();
            const list = document.getElementById('raycastResults');
            list.innerHTML = '';
            
            const matches = RAYCAST_COMMANDS.filter(c => c.title.toLowerCase().includes(q) || c.desc.toLowerCase().includes(q));
            matches.forEach((c, idx) => {
                const li = document.createElement('li');
                li.className = 'raycast-item' + (idx === 0 ? ' selected' : '');
                li.innerHTML = `
                    <div class="left">
                        <span>âš¡</span>
                        <div>
                            <div>${c.title}</div>
                            <div class="sub">${c.desc}</div>
                        </div>
                    </div>
                    <span class="kbd-pill">â†µ</span>
                `;
                li.onclick = () => {
                    closeRaycast();
                    const input = document.getElementById('composerInput');
                    input.value = c.prompt;
                    submitMessage();
                };
                list.appendChild(li);
            });
        }

        // Theme management
        function setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('zvy_theme', theme);
            const icon = document.getElementById('themeIcon');
            const label = document.getElementById('themeLabel');
            const select = document.getElementById('themeSelect');
            if (theme === 'light') {
                if (icon) icon.innerText = 'ðŸŒ™';
                if (label) label.innerText = 'Dark mode';
                if (select) select.value = 'light';
            } else {
                if (icon) icon.innerText = 'â˜€ï¸';
                if (label) label.innerText = 'Light mode';
                if (select) select.value = 'dark';
            }
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            setTheme(current === 'dark' ? 'light' : 'dark');
        }

        function openSettings() {
            document.getElementById('settingsModal').classList.add('open');
        }

        function closeSettings() {
            document.getElementById('settingsModal').classList.remove('open');
        }

        function toggleLatisFromSettings(checked) {
            if (checked && !enabledPlugins.includes('latis')) {
                enabledPlugins.push('latis');
            } else if (!checked) {
                enabledPlugins = enabledPlugins.filter(p => p !== 'latis');
            }
            updateLatisTag();
        }

        // '+' Attachment Menu
        function toggleAttachmentMenu(e) {
            e.stopPropagation();
            const menu = document.getElementById('attachmentMenu');
            menu.classList.toggle('open');
        }

        document.addEventListener('click', () => {
            const menu = document.getElementById('attachmentMenu');
            if (menu) menu.classList.remove('open');
        });

        function selectAttachment(type) {
            const menu = document.getElementById('attachmentMenu');
            menu.classList.remove('open');
            const input = document.getElementById('composerInput');

            if (type === 'latis') {
                if (enabledPlugins.includes('latis')) {
                    enabledPlugins = enabledPlugins.filter(p => p !== 'latis');
                } else {
                    enabledPlugins.push('latis');
                }
                updateLatisTag();
            } else if (type === 'doc') {
                input.value = "What is our corporate return and refund policy?";
                input.focus();
            } else if (type === 'raycast') {
                openRaycast();
            }
        }

        function updateLatisTag() {
            const tag = document.getElementById('latisTag');
            const check = document.getElementById('latisSettingCheck');
            const active = enabledPlugins.includes('latis');
            if (tag) tag.innerText = active ? 'ACTIVE' : 'OFF';
            if (check) check.checked = active;
        }

        // Chat & Session storage
        function saveSessions() {
            localStorage.setItem('zvy_sessions', JSON.stringify(sessions));
            localStorage.setItem('zvy_active_session_id', currentSessionId);
        }

        function renderHistoryList() {
            const list = document.getElementById('chatHistoryList');
            list.innerHTML = '';
            sessions.forEach(s => {
                const li = document.createElement('li');
                li.className = 'chat-history-item' + (s.id === currentSessionId ? ' active' : '');
                li.innerHTML = `
                    <span class="chat-title" onclick="switchSession('${s.id}')">
                        <span style="opacity: 0.6;">ðŸ’¬</span>
                        <span>${s.title}</span>
                    </span>
                    <span class="btn-del" onclick="deleteSession(event, '${s.id}')" title="Delete">âœ•</span>
                `;
                list.appendChild(li);
            });
        }

        function switchSession(sessionId) {
            currentSessionId = sessionId;
            saveSessions();
            renderHistoryList();
            loadCurrentSession();
        }

        function createNewChat() {
            fetch('/api/reset', { method: 'POST' });
            const newSession = {
                id: 'session_' + Date.now(),
                title: 'New conversation',
                messages: []
            };
            sessions.unshift(newSession);
            currentSessionId = newSession.id;
            saveSessions();
            renderHistoryList();
            loadCurrentSession();
            document.getElementById('composerInput').focus();
        }

        function deleteSession(e, sessionId) {
            e.stopPropagation();
            if (sessions.length <= 1) {
                sessions[0].messages = [];
                sessions[0].title = 'New conversation';
                saveSessions();
                loadCurrentSession();
                renderHistoryList();
                return;
            }
            sessions = sessions.filter(s => s.id !== sessionId);
            if (currentSessionId === sessionId) {
                currentSessionId = sessions[0].id;
            }
            saveSessions();
            renderHistoryList();
            loadCurrentSession();
        }

        function loadCurrentSession() {
            const current = sessions.find(s => s.id === currentSessionId);
            const flow = document.getElementById('chatFlow');
            const titleEl = document.getElementById('activeChatTitle');

            if (!current || !current.messages.length) {
                flow.innerHTML = `
                    <div class="message-row agent-message-row">
                        <div class="agent-response-content">${marked.parse("Hello. How can I help you with accounts, policies, or operations today?")}</div>
                    </div>
                `;
                if (titleEl) titleEl.innerText = 'New conversation';
                return;
            }

            if (titleEl) titleEl.innerText = current.title;
            flow.innerHTML = '';
            current.messages.forEach(m => {
                if (m.role === 'user') {
                    renderUserBubble(m.text, false);
                } else {
                    renderAgentResponse(m.text, m.toolCalls || [], false);
                }
            });
            flow.scrollTop = flow.scrollHeight;
        }

        function appendUserMessage(text) {
            renderUserBubble(text, true);
            const current = sessions.find(s => s.id === currentSessionId);
            if (current) {
                if (current.messages.length === 0) {
                    current.title = text.length > 25 ? text.substring(0, 25) + '...' : text;
                    document.getElementById('activeChatTitle').innerText = current.title;
                    renderHistoryList();
                }
                current.messages.push({ role: 'user', text: text });
                saveSessions();
            }
        }

        function renderUserBubble(text, scroll = true) {
            const flow = document.getElementById('chatFlow');
            const row = document.createElement('div');
            row.className = 'message-row user-message-row';
            row.innerHTML = `<div class="user-bubble">${text.replace(/\\n/g, '<br>')}</div>`;
            flow.appendChild(row);
            if (scroll) flow.scrollTop = flow.scrollHeight;
        }

        function appendAgentResponse(text, toolCalls = []) {
            renderAgentResponse(text, toolCalls, true);
            const current = sessions.find(s => s.id === currentSessionId);
            if (current) {
                current.messages.push({ role: 'assistant', text: text, toolCalls: toolCalls });
                saveSessions();
            }
        }

        function renderAgentResponse(text, toolCalls = [], scroll = true) {
            const flow = document.getElementById('chatFlow');
            const row = document.createElement('div');
            row.className = 'message-row agent-message-row';

            let toolHtml = '';
            if (toolCalls && toolCalls.length > 0) {
                toolCalls.forEach(t => {
                    toolHtml += `
                        <div class="tool-card">
                            <span><span class="dot"></span>${t.tool}</span>
                        </div>
                    `;
                });
            }

            row.innerHTML = `
                ${toolHtml}
                <div class="agent-response-content">
                    ${marked.parse(text)}
                </div>
            `;
            flow.appendChild(row);
            if (scroll) flow.scrollTop = flow.scrollHeight;
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitMessage();
            }
        }

        async function submitMessage() {
            const input = document.getElementById('composerInput');
            const text = input.value.trim();
            if (!text) return;

            input.value = '';
            appendUserMessage(text);

            const btn = document.getElementById('btnDispatch');
            btn.disabled = true;
            btn.style.opacity = '0.5';

            try {
                const res = await fetch('/api/command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        command: text,
                        plugins_enabled: enabledPlugins
                    })
                });
                const data = await res.json();
                appendAgentResponse(data.response || '', data.tools_called || []);
            } catch (err) {
                appendAgentResponse('Error communicating with Zvy_Model.');
            } finally {
                btn.disabled = false;
                btn.style.opacity = '1';
            }
        }

        initApp();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")

if __name__ == "__main__":
    run_server()

