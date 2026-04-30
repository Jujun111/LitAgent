# LitAgent: Automated Information Retrieval & Academic Synthesis

LitAgent is a multi-agent, deterministic AI orchestration system designed to retrieve, analyze, and synthesize academic literature.

Unlike traditional LLM wrappers that rely on brittle prompt-loops, LitAgent strictly decouples **Planning/Execution** from **AI Generation**. It utilizes a Finite State Machine (FSM) to ensure deterministic control flow while leveraging a locally hosted, quantized Large Language Model (Qwen 9B) via constrained decoding to guarantee hallucination-free, structurally valid JSON outputs.

##  System Architecture

The system is strictly divided into two decoupled components communicating via a formal API contract:

1. **The Deterministic Controller (LangGraph):** Orchestrates the FSM, handles external API rate limits (Semantic Scholar), manages state memory, and compiles the final dossier.
    
2. **The AI Microservice (vLLM):** A local inference server running Qwen 9B. It executes constrained decoding (via Pydantic) to mathematically guarantee the JSON structure of the synthesis.
    

##  Repository Structure

Development is divided across specific feature branches to maintain the Separation of Concerns:

- `main`: Contains the formal data contract (`api_contracts.py`) and production-ready code.
    
- `feat/langgraph-fsm`: Development branch for the orchestration logic, external API integrations, and user interface.
    
- `feat/llm-service`: Development branch for the local vLLM deployment, prompt engineering, and hardware optimization.
    

##  Getting Started

### Prerequisites

- Python 3.10+
    
- Local GPU hardware capable of running a language model.
    

### 1. The FSM Orchestrator Setup (Software Engineering)

Ensure you are on the `feat/langgraph-fsm` branch.

```
# Install dependencies
pip install langgraph pydantic requests
```

To run the orchestrator in isolation (using the mocked AI response):

```
python litagent_fsm.py
```

### 2. The AI Microservice Setup (AI Engineering)

Ensure you are on the `feat/llm-service` branch.

```
# Install vLLM (Linux/WSL recommended)
pip install vllm
```

_Note: The exact launch command for the vLLM server will be documented here once the local port and constrained decoding parameters are finalized by the AI Engineering team._

##  The API Contract

All communication between the FSM and the AI Microservice is strictly governed by `api_contracts.py`.

- **The FSM** guarantees it will send a `SynthesisRequestPayload` containing the `paper_id`, `chunked_text`, and a latency threshold.
    
- **The AI Microservice** guarantees it will return a `ResearchSynthesis` JSON object containing the `paper_title`, `executive_summary`, `methodology_used`, and `key_findings`.
    

Do not alter `api_contracts.py` without mutual agreement between both team members.

##  Team & Responsibilities

- **Software/Systems Engineering:** FSM orchestration, UI development (Streamlit/Gradio), Semantic Scholar API integration, and error fallback logic.
    
- **Artificial Intelligence Engineering:** Model deployment (vLLM), hardware latency optimization, prompt engineering, and constrained decoding enforcement.
    

_Disclaimer: LitAgent is developed as part of the "Engineering of AI-Intensive Systems" course. All generated dossiers contain explicit transparency warnings indicating AI involvement._