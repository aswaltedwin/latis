"""
Zvy Native Desktop Launcher (Auto-Server Startup)
Design: Claude + Linear Inspired Hybrid Workspace
"""

import sys
import time
import threading
import webview
import requests
import uvicorn
from zvy_agent.web_app import app

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")

def is_server_running():
    try:
        r = requests.get("http://127.0.0.1:8080/api/customers", timeout=0.8)
        return r.status_code == 200
    except Exception:
        return False

def main():
    # If the backend server isn't running, start it in a background thread
    if not is_server_running():
        t = threading.Thread(target=start_server, daemon=True)
        t.start()
        time.sleep(1.2)

    # Launch native desktop app window
    window = webview.create_window(
        title="Zvy Copilot",
        url="http://127.0.0.1:8080",
        width=1240,
        height=780,
        min_size=(1020, 640),
        background_color="#0D0E12",
        text_select=True
    )
    webview.start()

if __name__ == "__main__":
    main()
