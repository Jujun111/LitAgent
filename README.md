# LitAgent

LitAgent is a course prototype for automated academic literature retrieval and synthesis. It uses a deterministic LangGraph finite state machine (FSM) to control retrieval, prompt construction, LLM invocation, JSON validation, retry behavior, and dashboard output.

The project is demo-ready with mock data by default. It can also call Semantic Scholar for live metadata and an OpenAI-compatible local model server such as vLLM, Ollama, or LM Studio when those services are available.

## Current Architecture

- `litagent_backend.py` is the canonical implementation. It contains the Semantic Scholar client, rate limiter, LangGraph graph builder, mock LLM, OpenAI-compatible LLM call, Pydantic schema validation, pipeline runner, and benchmark helper.
- `streamlit_app.py` is the dashboard UI for generating dossiers and running the 5x benchmark.
- `litagent_fsm.py` is a small CLI demo that calls the canonical backend in mock LangGraph mode.
- `api_contracts.py` preserves the original single-paper AI contract used during earlier parallel development.
- `smoke_test.py` verifies the mock LangGraph path and benchmark acceptance checks.
- `llm_service/` contains the optional vLLM AI microservice launch script, config example, health check, and schema-constrained smoke test.

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
| NfReq03: 60-second LLM latency | local model HTTP timeout, latency metric, and vLLM smoke test |
| NfReq04: Extraction fidelity target | vLLM structured output mode; full fidelity evaluation requires a labeled benchmark set |
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

## vLLM AI Microservice

The `feat/llm-service` branch includes a reproducible vLLM service wrapper for the AI component. It serves an OpenAI-compatible `/v1/chat/completions` endpoint, and the controller sends vLLM structured output parameters in this shape:

```json
{
  "structured_outputs": {
    "json": "<ResearchDossier JSON schema>"
  }
}
```

Install the optional AI service dependencies in a Linux/WSL CUDA environment:

```bash
python -m venv .venv-llm
source .venv-llm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-llm.txt
```

Copy and edit the service config if needed:

```bash
cp llm_service/config.example.env llm_service/.env
```

Start vLLM:

```bash
bash llm_service/serve_vllm.sh llm_service/.env
```

Default model settings:

```text
LITAGENT_VLLM_MODEL=jc-builds/Qwen3.5-9B-Q4_K_M-GGUF:Q4_K_M
LITAGENT_VLLM_TOKENIZER=Qwen/Qwen3.5-9B
LITAGENT_VLLM_MAX_MODEL_LEN=4096
LITAGENT_VLLM_GPU_MEMORY_UTILIZATION=0.85
```

vLLM supports GGUF quantization, but its GGUF support is still described as experimental and under-optimized by the vLLM docs. vLLM also recommends using the base tokenizer when serving GGUF models. The local development machine has an RTX 4060 Laptop GPU with 8GB VRAM, so Qwen3.5-9B Q4_K_M is plausible but tight; if startup fails due to memory pressure, keep the same scripts and override `LITAGENT_VLLM_MODEL` / `LITAGENT_VLLM_TOKENIZER` with a smaller GGUF model.

Check the running service:

```bash
python llm_service/check_vllm.py
python llm_service/smoke_vllm_schema.py
```

Streamlit preset:

```text
Model provider: vLLM local
Base URL: http://localhost:8000/v1
API key: token-abc123
Model: jc-builds/Qwen3.5-9B-Q4_K_M-GGUF:Q4_K_M
JSON mode: vllm
```

## Current Limitations

- The default demo synthesizes abstracts, not full downloaded PDFs.
- The 95% extraction fidelity requirement can only be evaluated with a real model and a labeled benchmark set.
- Local vLLM deployment is optional and environment-dependent; Windows users should prefer WSL/Linux CUDA for vLLM.
- Assignment PDF files are intentionally ignored and should remain local unless a private course repository requires them.
