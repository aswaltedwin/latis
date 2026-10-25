# LATIS — AI Security Investigation Platform

LATIS is a tool that finds out **which part of an AI system caused a security problem**.

When an AI application (like a chatbot with document search or tool access) behaves unsafely — leaking data, running unauthorized commands, or following injected instructions — LATIS investigates the failure, tests each component, and tells you exactly what went wrong and why.

---

## What Problem Does It Solve?

Modern AI applications are built from many parts working together:

- A **document search** (retrieves context from files or a knowledge base)
- A **system instruction** (tells the AI how to behave)
- A **memory store** (saves past conversation or session data)
- **Tools** (actions the AI can take, like calling an API or running a command)
- The **AI model** itself

When something goes wrong, it is not obvious which part is at fault. LATIS tests each part individually and in combinations, measures the results, and gives a clear answer.

---

## How It Works

1. **Load an incident** — describe what happened (the user request, the failure, and which parts were active).
2. **Run tests** — LATIS temporarily removes or disables one part at a time, replays the same request, and checks whether the failure still occurs.
3. **Compare results** — if removing a part makes the failure disappear, that part was responsible.
4. **Get a report** — LATIS outputs a diagnosis with confidence scores and the role each part played.

---

## What LATIS Can Identify

| Result | Meaning |
|---|---|
| Single Necessary Cause | One part alone caused the failure |
| Enabling Structure | One part allowed another to cause the failure |
| Joint Interaction | Two parts together caused the failure (neither alone is enough) |
| Redundant Causes | Either part alone is enough; both paths lead to failure |
| Correlated Artifact | A part was present but was not the cause |

---

## Project Structure

```
latis/
│
├── backend/                  # Core server and investigation logic
│   ├── server.py             # API server — handles all requests from the UI
│   ├── planner.py            # Decides which test to run next based on current uncertainty
│   ├── causal_matrix.py      # Reads test results and identifies the failure pattern
│   ├── case_registry.py      # Pre-built incident cases for testing and demos
│   ├── governance.py         # Checks whether enough tests have been done before closing a case
│   ├── expansion_policy.py   # Controls which combinations of parts to test together
│   ├── rag_analyzer.py       # Inspects retrieved document chunks for injection or tampering
│   ├── system_prompt_analyzer.py  # Checks system instructions for security issues
│   ├── memory_analyzer.py    # Checks saved memory records for malicious content
│   ├── tool_analyzer.py      # Checks tool calls for dangerous or unauthorized actions
│   ├── owasp_engine.py       # Checks events against the OWASP Top 10 for AI threats
│   ├── detector.py           # Watches live events and raises alerts in real time
│   ├── telemetry.py          # Stores runtime events and security alerts
│   ├── remediation.py        # Applies fixes: removes bad documents, revokes tools, clears memory
│   └── bridge.py             # Connects live telemetry alerts to full investigations
│
├── real_sandbox/             # A test AI pipeline you can run investigations against
│   ├── pipeline.py           # Runs a full AI request: retrieval → prompt → model → tool
│   ├── model.py              # Local AI model connection (Ollama or deterministic fallback)
│   ├── rag.py                # Document retrieval with selectable corpus
│   ├── tools.py              # Tool definitions and execution sandbox
│   ├── memory.py             # Session memory store
│   └── context.py            # Builds and masks context passed to the model
│
├── replay/                   # Runs the same request many times under different conditions
│   ├── runner.py             # Executes N repeated runs and counts outcomes
│   ├── experiment.py         # Defines what to test (mask a chunk, revoke a tool, etc.)
│   ├── outcomes.py           # Tracks SAFE / UNSAFE / BLOCKED / ERROR counts
│   └── statistics.py         # Converts counts to confidence scores
│
├── behavior_model/           # A small neural model that learns safe vs unsafe action patterns
│   ├── model_arch.py         # Model architecture
│   ├── dataset.py            # Dataset preparation
│   └── train.py              # Training script
│
├── zvy_agent/                # A demo enterprise AI agent that LATIS investigates
│   ├── agent_core.py         # AI agent powered by Groq (Qwen 3.8-27b model)
│   ├── tools.py              # Agent tools: CRM lookup, refunds, emails, diagnostics
│   ├── web_app.py            # Web interface for the Zvy agent
│   ├── app.py                # Flask app entry point
│   └── model/                # Trained behavior model for the Zvy agent
│
├── testbed/                  # Automated tests for every part of the system
│
├── ui/                       # Browser-based investigation console
│   ├── index.html            # Main investigation dashboard
│   ├── app.js                # Frontend logic
│   └── style.css             # Styles
│
├── data/                     # Sample documents and audit data
├── zvy_desktop.py            # Desktop launcher for the Zvy agent
├── Launch_Zvy_Desktop.bat    # Windows quick-start for the Zvy agent
└── requirements.txt          # Python dependencies
```

---

## The Zvy Agent (Demo Target)

The project includes **Zvy**, a demo enterprise AI assistant that handles tasks like customer lookups, refunds, and system diagnostics. Zvy is the AI system that LATIS investigates.

- Zvy is powered by **Groq** (using the Qwen 3.8-27b model) for fast responses and real tool calling.
- It has tools for: looking up customers, processing refunds, sending emails, running diagnostics, and searching the knowledge base.
- LATIS monitors Zvy's actions in real time and investigates whenever something unsafe happens.

---

## Getting Started

### Requirements
- Python 3.10 or higher
- (Optional) [Ollama](https://ollama.com/) running locally for live model inference in the sandbox
- A [Groq API key](https://console.groq.com/) if you want to run the Zvy agent

### Install

```bash
git clone https://github.com/aswaltedwin/latis.git
cd latis

python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Run the Investigation Server

```bash
python -m backend.server
```

The server starts at `http://127.0.0.1:8000`.

### Open the Investigation Console

Open `ui/index.html` directly in your browser, or serve it:

```bash
python -m http.server 3000 --directory ui
```

Then go to `http://localhost:3000`.

### Launch the Zvy Agent (Optional)

```bash
# Windows
Launch_Zvy_Desktop.bat

# Or directly
python zvy_desktop.py
```

---

## API Reference

| Method | Endpoint | What It Does |
|---|---|---|
| `POST` | `/api/cases/ingest` | Load a new incident into the system |
| `GET` | `/api/cases/{id}/state` | Get current hypotheses, uncertainty level, and next recommended test |
| `POST` | `/api/cases/{id}/experiment` | Run a test (mask a part, toggle a tool, isolate a component) |
| `GET` | `/api/cases/{id}/report` | Get the final investigation report with confidence scores |
| `POST` | `/api/telemetry/event` | Submit a live runtime event for real-time detection |
| `GET` | `/api/alerts` | List security alerts raised by the online detector |
| `POST` | `/api/cases/{id}/remediate` | Apply a fix (quarantine document, revoke tool, clear memory) |
| `POST` | `/api/sandbox/multiagent/run` | Run a request through the multi-agent sandbox |
| `POST` | `/api/sandbox/multiagent/escalate` | Escalate a verified sandbox threat to a full investigation case |

---

## Known Limitations

- Results are more reliable with more repeated test runs. Very few runs (under 30) with a non-deterministic model may give noisy results.
- Components that are hidden inside a model (internal weights, activations) cannot be tested directly without model instrumentation.
- Three-way or higher combinations of parts are not fully covered in the current test budget.
- Running tests against a live local model (via Ollama) is limited by the speed of your hardware.

---

## Dependencies

```
fastapi
uvicorn
pydantic
requests
numpy
scipy
scikit-learn
```

---

## License

MIT License
