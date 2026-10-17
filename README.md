# LATIS: AI Security Incident Investigation & Causal Attribution Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**LATIS** (**L**earned **A**ttribution & **T**opology **I**dentification for AI **S**ecurity) is an active causal forensic platform designed to diagnose, isolate, and disambiguate failures in compound AI systems (e.g., RAG pipelines, autonomous tool agents, prompt injections, and multi-component LLM workflows).

---

## 🌟 Key Features

- **Active Counterfactual Interventions**: Systematically tests single component masks (`CONTEXT_PRESERVING_MASK`), tool authorization toggles, joint multi-component interventions, and component isolations.
- **SEASIS Cognitive Engine Adapter**: Integrates entropy-driven Bayesian hypothesis updates, adaptive next-best-experiment planning, and reflective learning.
- **Causal Identification Matrix**: Disambiguates complex failure topologies, including:
  - *Single Necessary Cause*
  - *Enabling Structures*
  - *Joint Compositional Interactions*
  - *Redundant Failure Paths*
  - *Correlated / Spurious Artifacts*
- **Forensic Evidence Reporting**: Generates confidence scores and complete audit logs for security compliance.
- **Interactive Web Console**: Real-time interface with live telemetry, causal graph visualization, entropy monitoring, and intervention controls.
- **Real Sandbox & Local LLM Integration**: Run interventions on simulated pipelines or real local models (e.g., Ollama `qwen2.5:3b`, OpenAI-compatible servers).

---

## 📁 Repository Structure

```
latis/
├── backend/
│   ├── server.py                     # FastAPI REST server & runtime investigator
│   ├── planner.py                    # Bayesian active-learning investigation planner
│   ├── rag_analyzer.py               # Context-aware RAG Cyber Evidence Engine
│   ├── system_prompt_analyzer.py     # System & Developer Prompt Cyber Evidence Engine
│   ├── case_registry.py              # Preset incident case definitions
│   ├── causal_matrix.py              # Pearlian Causal Identification Matrix
│   ├── governance.py                 # Causal sufficiency & closure admissibility
│   └── expansion_policy.py           # Adaptive joint-candidate expansion policy

├── real_sandbox/                     # Real LLM / RAG / Tool sandbox with Ollama & local fallback support
├── replay/                           # Deterministic replay and outcome distribution engine
├── ui/                               # Interactive HTML5/CSS3 Investigation Console

├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

### 🏷️ Component Classification: Production vs Validation

| Component | Nature / Type | Description |
|---|---|---|
| `backend/` | **Core Production Architecture** | REST API server, Bayesian hypothesis planner, security analyzers (RAG, System Prompt, Memory, Tool/Auth), and Pearlian causal identification matrix. |
| `backend/case_registry.py` | **Validation Test Cases** | Canonical multi-component incident benchmarks (PV2) for automated regression testing and live investigation. |
| `real_sandbox/` | **Hybrid Execution Sandbox** | Connects to live local LLMs via Ollama; automatically provides deterministic fallback when no external model daemon is active. |
| `replay/` | **Controlled Replay Engine** | Statistical replay runner and counterfactual experiment execution engine ($do(-X)$, $do(-X,-Y)$, isolations). |
| `ui/` | **Interactive Investigation Console** | Production browser console providing real-time forensic telemetry, causal DAG rendering, and intervention controls. |

---

## 🔬 Scientific & Architectural Foundations

LATIS consolidates systematic empirical attribution capabilities into a unified modular framework:

- **Causal DAG Modeling**: Formal counterfactual intervention framework ($P(\text{unsafe} \mid \text{baseline})$ vs $P(\text{unsafe} \mid do(-X))$).
- **Pearlian Causal Identification Matrix**: Formal engine mathematically separating Single Necessary Causes, Enabling Structures, Joint Compositional Interactions, Redundant Paths, and Unrepresented External Causes.
- **Adaptive Joint-Candidate Expansion Policy**: Dynamic pair-selection engine reducing the combinatorial $O(N^2)$ joint search space during interaction investigations.
- **Causal Sufficiency Governance**: Evidentiary sufficiency guardrails preventing premature case closure before candidate interactions are resolved.
- **Multi-Layer Cybersecurity Analyzers**: Static and empirical analyzers spanning RAG corpora, System Prompts, Persistent Agent Memory, and Tool/Authorization boundaries.

---

## ⚠️ Current Status / Limitations

- **Stochastic Variability**: Like all empirical causal testing frameworks, low repetition sample sizes ($N < 30$) under high non-deterministic model temperature can increase variance in outcome probability distributions.
- **Unobserved Confounders & Hidden State**: Interventions operate on observable pipeline nodes (prompts, retrieved chunks, tool permissions, fine-tuned adapters); hidden internal activations or unlogged system states outside the DAG cannot be directly counterfactually intervened upon without instrumentation hooks.
- **High-Order Interactions ($k \ge 3$)**: While single ($k=1$) and pairwise ($k=2$) joint interactions are adaptively covered, $3$-way or higher simultaneous compositional dependencies require higher intervention budgets.
- **Local Model Rate Limits / Latency**: Real-time evaluation with local models (e.g. via Ollama) is bound by local GPU/CPU inference latency during repeated replay runs.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- (Optional) [Ollama](https://ollama.com/) running locally for live LLM execution in `real_sandbox`

### 2. Installation
Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/your-org/latis.git
cd latis

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Running the Platform

### Option A: Launch Web API & UI Console

1. **Start the FastAPI Backend Server:**
   ```bash
   python -m backend.server
   ```
   The backend API will start at `http://127.0.0.1:8000`.


2. **Open the Investigation Console:**
   - Open [`ui/index.html`](ui/index.html) directly in your browser, or serve it using any static server:
   ```bash
   python -m http.server 3000 --directory ui
   ```
   - Navigate to `http://localhost:3000` to interact with the forensic console, run interventions, and generate reports.

---

### Option B: Run Replay Acceptance Demo

You can run an automated controlled replay intervention check directly via CLI:

```bash
# Run Controlled Replay Orchestrator Acceptance Demo
python -m replay.demo
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/cases/ingest` | Ingests a new security incident case |
| `GET` | `/api/cases/{id}/state` | Fetches dependency graph, hypotheses, entropy, and recommended next experiment |
| `POST` | `/api/cases/{id}/experiment` | Executes an active intervention (single mask, joint mask, toggle) and returns outcome distribution |
| `GET` | `/api/cases/{id}/report` | Generates a complete forensic causal report with confidence score |

---

## 🔬 Scientific & Causal Methodology

1. **Prior Formation**: Ingested components are mapped to competing causal hypotheses ($H_1, H_2, H_3, H_4$).
2. **Optimal Active Intervention**: SEASIS selects the counterfactual action yielding maximum Information Gain ($IG = H(S) - \mathbb{E}[H(S \mid A)]$).
3. **Execution & Replay**: Context-preserving interventions replace tokens or mask retrieval chunks while preserving non-intervened context.
4. **Bayesian Posterior Update**: Distributions from repeated runs update hypothesis posteriors and reduce Shannon Entropy.
5. **Causal Disambiguation**: The Causal Identification Matrix identifies root causes, distinguishing direct causes from enabling conditions and spurious artifacts.

---

## 📜 License

This project is licensed under the MIT License.
