# LitAgent

LitAgent is a course prototype for automated academic literature retrieval and synthesis. It uses a deterministic LangGraph finite state machine (FSM) to control retrieval, prompt construction, LLM invocation, JSON validation, retry behavior, and dashboard output.

The project is demo-ready with mock data by default. It can also call Semantic Scholar for live metadata and an OpenAI-compatible local model server such as vLLM, Ollama, or LM Studio when those services are available.

## Current Architecture

- `litagent_backend.py` is the canonical implementation. It contains the Semantic Scholar client, rate limiter, LangGraph graph builder, mock LLM, OpenAI-compatible LLM call, Pydantic schema validation, pipeline runner, and benchmark helper.
- `streamlit_app.py` is the dashboard UI for generating dossiers and running the 5x benchmark.
- `litagent_fsm.py` is a small CLI demo that calls the canonical backend in mock LangGraph mode.
- `api_contracts.py` preserves the original single-paper AI contract used during earlier parallel development.
- `smoke_test.py` verifies the mock LangGraph path and benchmark acceptance checks.

## Requirements Mapping

| Requirement | Implementation |
|---|---|
| Req01: Retrieve up to 10 papers | `SemanticScholarClient.search_papers(... limit=10)` in live mode |
| Req02: FSM orchestration | `build_litagent_graph()` with LangGraph nodes |
| Req03: Prompt AI after retrieval/chunking | `Build_Prompt` then `Call_LLM` graph transition |
| Req04: Strict JSON schema | `ResearchDossier` Pydantic validation |
| Req05: Dashboard dossier | `streamlit_app.py` dossier tab and metrics |
| Req06: Stop after 3 validation retries | conditional edge after `Validate_JSON` |
| NfReq01: 100 requests / 5 minutes | `SemanticScholarRateLimiter` |
| NfReq02: Python + LangGraph | Python implementation using LangGraph |
| NfReq03: 60-second LLM latency | local model HTTP timeout and latency metric |
| NfReq04: Extraction fidelity target | supported through structured output mode; full fidelity evaluation requires a real model |
| NfReq05: AI disclaimer | `AI_DISCLAIMER` rendered in every validated dossier |

## Setup

Python 3.10+ is recommended for the prototype. The local smoke checks in this workspace passed on Python 3.13; optional vLLM deployment may require a narrower Python/CUDA combination.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run the Prototype

Run the smoke test:

```bash
python smoke_test.py
```

Run the CLI demo:

```bash
python litagent_fsm.py "graph neural networks for molecular property prediction"
python litagent_fsm.py --benchmark
```

Start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8501
```

## Live Retrieval and Local LLM Options

Mock papers are enabled by default so the course prototype works without external services.

For live Semantic Scholar retrieval, turn off **Use mock papers** in the sidebar. If the API fails, the app reports the error by default. Turn on **Fallback to mock papers** only when you explicitly want demo fallback behavior.

For local model inference, choose an OpenAI-compatible provider in the sidebar and provide the base URL, API key, model name, and JSON mode. vLLM is the architecture target, but it is intentionally not included in `requirements.txt` because installation is CUDA/Linux/WSL-specific.

Example vLLM command on a compatible machine:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --api-key token-abc123
```

Then use:

```text
Model provider: vLLM local
Base URL: http://localhost:8000/v1
API key: token-abc123
JSON mode: vllm
```

## Current Limitations

- The default demo synthesizes abstracts, not full downloaded PDFs.
- The 95% extraction fidelity requirement can only be evaluated with a real model and a labeled benchmark set.
- Local vLLM deployment is optional and environment-dependent.
- Assignment PDF files are intentionally ignored and should remain local unless a private course repository requires them.
