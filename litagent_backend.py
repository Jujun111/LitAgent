from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, TypedDict

import requests
from pydantic import BaseModel, Field, ValidationError


AI_DISCLAIMER = (
    "This dossier was generated automatically by a local AI pipeline. "
    "It should be reviewed by a human before being used for academic decisions."
)


@dataclass
class AuthorMetadata:
    author_id: Optional[str]
    name: str


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    venue: Optional[str]
    citation_count: Optional[int]
    reference_count: Optional[int]
    influential_citation_count: Optional[int]
    authors: List[AuthorMetadata] = field(default_factory=list)
    fields_of_study: List[str] = field(default_factory=list)
    url: Optional[str] = None
    open_access_pdf_url: Optional[str] = None
    external_ids: Dict[str, str] = field(default_factory=dict)


class KeyPaperItem(BaseModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    reason: str
    key_findings: List[str] = Field(default_factory=list)


class ResearchDossier(BaseModel):
    query: str
    topic: str
    summary: str
    key_papers: List[KeyPaperItem]
    limitations: List[str] = Field(default_factory=list)
    disclaimer: str


def research_dossier_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "topic", "summary", "key_papers", "limitations", "disclaimer"],
        "properties": {
            "query": {"type": "string"},
            "topic": {"type": "string"},
            "summary": {"type": "string"},
            "key_papers": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["paper_id", "title", "year", "venue", "reason", "key_findings"],
                    "properties": {
                        "paper_id": {"type": "string"},
                        "title": {"type": "string"},
                        "year": {"type": "integer"},
                        "venue": {"type": "string"},
                        "reason": {"type": "string"},
                        "key_findings": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "disclaimer": {"type": "string"},
        },
    }


class LitAgentState(TypedDict, total=False):
    query: str
    raw_papers: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    prompt: str
    llm_raw_output: str
    dossier: Dict[str, Any]
    is_schema_valid: bool
    error: str
    validation_error: str
    validation_attempts: int
    fetch_mode: str
    llm_mode: str
    orchestrator: str
    llm_latency_seconds: float
    end_to_end_latency_seconds: float


class SemanticScholarError(Exception):
    pass


class SemanticScholarRateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 300) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: Deque[float] = deque()

    def wait_if_needed(self) -> None:
        now = time.monotonic()

        while self.request_times and now - self.request_times[0] >= self.window_seconds:
            self.request_times.popleft()

        if len(self.request_times) < self.max_requests:
            self.request_times.append(now)
            return

        sleep_for = self.window_seconds - (now - self.request_times[0])
        if sleep_for > 0:
            time.sleep(sleep_for)

        now = time.monotonic()
        while self.request_times and now - self.request_times[0] >= self.window_seconds:
            self.request_times.popleft()
        self.request_times.append(now)


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    DEFAULT_FIELDS = (
        "paperId,title,abstract,year,venue,citationCount,referenceCount,"
        "influentialCitationCount,authors,fieldsOfStudy,url,openAccessPdf,externalIds"
    )

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.rate_limiter = SemanticScholarRateLimiter()
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def search_papers(self, query: str, limit: int = 10, retries: int = 5) -> List[PaperMetadata]:
        endpoint = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": self.DEFAULT_FIELDS,
        }
        last_error: Optional[Exception] = None

        for attempt in range(retries):
            self.rate_limiter.wait_if_needed()
            response = self.session.get(endpoint, params=params, timeout=self.timeout)

            if response.status_code == 200:
                papers = response.json().get("data", [])
                return [self.parse_paper_metadata(paper) for paper in papers]

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else min(5 * (attempt + 1), 60)
                time.sleep(sleep_for)
                last_error = SemanticScholarError(f"429 Too Many Requests: {response.text}")
                continue

            if response.status_code >= 500:
                time.sleep(min(2**attempt, 30))
                last_error = SemanticScholarError(f"Server error {response.status_code}: {response.text}")
                continue

            raise SemanticScholarError(f"Request failed with status {response.status_code}: {response.text}")

        raise last_error or SemanticScholarError("Unknown Semantic Scholar request failure.")

    @staticmethod
    def parse_paper_metadata(raw: Dict[str, Any]) -> PaperMetadata:
        authors = [
            AuthorMetadata(author_id=author.get("authorId"), name=author.get("name", "Unknown Author"))
            for author in raw.get("authors", [])
        ]
        open_access_pdf = raw.get("openAccessPdf") or {}

        return PaperMetadata(
            paper_id=raw.get("paperId", ""),
            title=raw.get("title", "Untitled"),
            abstract=raw.get("abstract"),
            year=raw.get("year"),
            venue=raw.get("venue"),
            citation_count=raw.get("citationCount"),
            reference_count=raw.get("referenceCount"),
            influential_citation_count=raw.get("influentialCitationCount"),
            authors=authors,
            fields_of_study=raw.get("fieldsOfStudy") or [],
            url=raw.get("url"),
            open_access_pdf_url=open_access_pdf.get("url"),
            external_ids=raw.get("externalIds") or {},
        )


def make_mock_papers(query: str) -> List[Dict[str, Any]]:
    return [
        {
            "paper_id": "demo-001",
            "title": f"Demo Paper 1 for: {query}",
            "abstract": (
                "Graph neural networks are widely used for molecular property prediction. "
                "They encode atoms and bonds as graphs and learn representations through message passing."
            ),
            "year": 2024,
            "venue": "Demo Venue A",
            "citation_count": 42,
            "reference_count": 10,
            "influential_citation_count": 5,
            "authors": [{"author_id": "a1", "name": "Demo Author A"}],
            "fields_of_study": ["Computer Science"],
            "url": None,
            "open_access_pdf_url": None,
            "external_ids": {},
        },
        {
            "paper_id": "demo-002",
            "title": f"Demo Paper 2 for: {query}",
            "abstract": (
                "Molecular property prediction commonly compares graph convolutions, attention, "
                "and descriptor-based baselines across benchmark datasets."
            ),
            "year": 2023,
            "venue": "Demo Venue B",
            "citation_count": 21,
            "reference_count": 8,
            "influential_citation_count": 2,
            "authors": [{"author_id": "a2", "name": "Demo Author B"}],
            "fields_of_study": ["Artificial Intelligence"],
            "url": None,
            "open_access_pdf_url": None,
            "external_ids": {},
        },
    ]


def fetch_data_node(
    state: LitAgentState,
    api_key: Optional[str] = None,
    use_mock_fetch: bool = True,
    fallback_to_mock_on_error: bool = False,
) -> LitAgentState:
    query = state["query"]
    if use_mock_fetch:
        return {"raw_papers": make_mock_papers(query), "fetch_mode": "mock"}

    try:
        papers = SemanticScholarClient(api_key=api_key).search_papers(query, limit=10)
        return {"raw_papers": [asdict(paper) for paper in papers], "fetch_mode": "live"}
    except Exception as exc:
        if not fallback_to_mock_on_error:
            raise
        return {
            "raw_papers": make_mock_papers(query),
            "fetch_mode": "mock_fallback",
            "error": str(exc),
        }


def chunk_text_node(state: LitAgentState, chunk_size: int = 1200) -> LitAgentState:
    chunks: List[Dict[str, Any]] = []
    for paper in state.get("raw_papers", []):
        abstract = paper.get("abstract") or ""
        for i in range(0, len(abstract), chunk_size):
            chunks.append(
                {
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", "Untitled"),
                    "year": paper.get("year"),
                    "venue": paper.get("venue"),
                    "text": abstract[i : i + chunk_size],
                }
            )
    return {"chunks": chunks}


def build_prompt(query: str, chunks: List[Dict[str, Any]], max_chunks: int = 8) -> str:
    joined_context = "\n\n".join(
        (
            f"[Chunk {i + 1}]\n"
            f"paper_id: {chunk.get('paper_id', '')}\n"
            f"title: {chunk.get('title', '')}\n"
            f"year: {chunk.get('year', '')}\n"
            f"venue: {chunk.get('venue', '')}\n"
            f"text: {chunk.get('text', '')}"
        )
        for i, chunk in enumerate(chunks[:max_chunks])
    )

    return f"""
User query:
{query}

Context:
{joined_context}

Create a concise research dossier from the context.
Return JSON only. No markdown. No extra commentary.

Required JSON fields:
- query
- topic
- summary
- key_papers: paper_id, title, year, venue, reason, key_findings
- limitations
- disclaimer

Use this disclaimer exactly:
{AI_DISCLAIMER}
""".strip()


def build_prompt_node(state: LitAgentState, max_chunks: int = 8) -> LitAgentState:
    return {"prompt": build_prompt(state["query"], state.get("chunks", []), max_chunks=max_chunks)}


def mock_call_llm_node(state: LitAgentState) -> LitAgentState:
    query = state.get("query", "unknown query")
    papers = state.get("raw_papers", [])
    payload = {
        "query": query,
        "topic": query,
        "summary": "Mock dossier summary generated without calling a local model server.",
        "key_papers": [
            {
                "paper_id": paper.get("paper_id", "unknown-id"),
                "title": paper.get("title", "Untitled"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "reason": "Included because its abstract matches the query in the current context.",
                "key_findings": [
                    "This is a placeholder finding.",
                    "Switch provider from Mock to a local server when one is running.",
                ],
            }
            for paper in papers[:3]
        ],
        "limitations": [
            "This mock output is for pipeline testing only.",
            "Use a real local model before evaluating extraction quality.",
        ],
        "disclaimer": AI_DISCLAIMER,
    }
    return {
        "llm_raw_output": json.dumps(payload, ensure_ascii=False),
        "llm_mode": "mock",
        "llm_latency_seconds": 0.0,
    }


def call_openai_compatible_node(
    state: LitAgentState,
    base_url: str,
    api_key: str,
    model: str,
    structured_mode: str = "vllm",
    timeout: int = 60,
) -> LitAgentState:
    prompt = state.get("prompt") or build_prompt(state["query"], state.get("chunks", []))
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a research dossier extraction assistant. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    if structured_mode == "vllm":
        payload["structured_outputs"] = {"json": ResearchDossier.model_json_schema()}
    elif structured_mode == "json_schema":
        payload["response_format"] = {
            "type": "json_object",
            "schema": research_dossier_json_schema(),
        }
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif structured_mode == "json_object":
        payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoint = base_url.rstrip("/") + "/chat/completions"
    start = time.perf_counter()
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"]

    return {
        "llm_raw_output": text.strip(),
        "llm_mode": structured_mode,
        "llm_latency_seconds": elapsed,
    }


def validate_json_node(state: LitAgentState) -> LitAgentState:
    raw = state.get("llm_raw_output", "")
    attempts = state.get("validation_attempts", 0) + 1
    try:
        model = ResearchDossier.model_validate_json(raw)
        return {
            "dossier": model.model_dump(),
            "is_schema_valid": True,
            "validation_attempts": attempts,
            "validation_error": "",
        }
    except ValidationError as exc:
        return {
            "is_schema_valid": False,
            "validation_attempts": attempts,
            "validation_error": str(exc),
            "error": str(exc),
        }


def build_litagent_graph(
    provider: str = "mock",
    use_mock_fetch: bool = True,
    fallback_to_mock_on_error: bool = False,
    semantic_scholar_api_key: Optional[str] = None,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "token-abc123",
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    structured_mode: str = "vllm",
    max_context_chunks: int = 8,
    max_validation_attempts: int = 3,
):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(LitAgentState)

    graph.add_node(
        "Fetch_Data",
        lambda state: fetch_data_node(
            state,
            api_key=semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            use_mock_fetch=use_mock_fetch,
            fallback_to_mock_on_error=fallback_to_mock_on_error,
        ),
    )
    graph.add_node("Chunk_Text", chunk_text_node)
    graph.add_node("Build_Prompt", lambda state: build_prompt_node(state, max_chunks=max_context_chunks))

    def call_llm(state: LitAgentState) -> LitAgentState:
        if provider == "mock":
            return mock_call_llm_node(state)
        return call_openai_compatible_node(
            state,
            base_url=base_url,
            api_key=api_key,
            model=model,
            structured_mode=structured_mode,
            timeout=60,
        )

    graph.add_node("Call_LLM", call_llm)
    graph.add_node("Validate_JSON", validate_json_node)

    def route_after_validation(state: LitAgentState) -> str:
        if state.get("is_schema_valid"):
            return "done"
        if state.get("validation_attempts", 0) >= max_validation_attempts:
            return "done"
        return "retry"

    graph.add_edge(START, "Fetch_Data")
    graph.add_edge("Fetch_Data", "Chunk_Text")
    graph.add_edge("Chunk_Text", "Build_Prompt")
    graph.add_edge("Build_Prompt", "Call_LLM")
    graph.add_edge("Call_LLM", "Validate_JSON")
    graph.add_conditional_edges(
        "Validate_JSON",
        route_after_validation,
        {
            "retry": "Call_LLM",
            "done": END,
        },
    )

    return graph.compile()


def run_pipeline(
    query: str,
    provider: str = "mock",
    use_mock_fetch: bool = True,
    fallback_to_mock_on_error: bool = False,
    semantic_scholar_api_key: Optional[str] = None,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "token-abc123",
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    structured_mode: str = "vllm",
    max_context_chunks: int = 8,
    use_langgraph: bool = True,
) -> Dict[str, Any]:
    pipeline_start = time.perf_counter()
    state: LitAgentState = {"query": query, "validation_attempts": 0}

    if use_langgraph:
        app = build_litagent_graph(
            provider=provider,
            use_mock_fetch=use_mock_fetch,
            fallback_to_mock_on_error=fallback_to_mock_on_error,
            semantic_scholar_api_key=semantic_scholar_api_key,
            base_url=base_url,
            api_key=api_key,
            model=model,
            structured_mode=structured_mode,
            max_context_chunks=max_context_chunks,
        )
        state = app.invoke(state)
        state["orchestrator"] = "langgraph"
    else:
        state.update(
            fetch_data_node(
                state,
                api_key=semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
                use_mock_fetch=use_mock_fetch,
                fallback_to_mock_on_error=fallback_to_mock_on_error,
            )
        )
        state.update(chunk_text_node(state))
        state.update(build_prompt_node(state, max_chunks=max_context_chunks))
        if provider == "mock":
            state.update(mock_call_llm_node(state))
        else:
            state.update(
                call_openai_compatible_node(
                    state,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    structured_mode=structured_mode,
                    timeout=60,
                )
            )
        state.update(validate_json_node(state))
        state["orchestrator"] = "sequential"

    if not state.get("is_schema_valid") and state.get("validation_error"):
        state.update(
            {
                "error": (
                    "JSON validation failed after "
                    f"{state.get('validation_attempts', 0)} attempt(s): {state['validation_error']}"
                )
            }
        )
    state["end_to_end_latency_seconds"] = time.perf_counter() - pipeline_start
    return state


def benchmark_pipeline(query: str, runs: int = 5, **kwargs: Any) -> Dict[str, Any]:
    measurements: List[Dict[str, Any]] = []
    for i in range(runs):
        result = run_pipeline(query, **kwargs)
        measurements.append(
            {
                "run": i + 1,
                "end_to_end_latency_seconds": result.get("end_to_end_latency_seconds"),
                "llm_latency_seconds": result.get("llm_latency_seconds"),
                "is_schema_valid": result.get("is_schema_valid", False),
                "fetch_mode": result.get("fetch_mode"),
                "llm_mode": result.get("llm_mode"),
                "orchestrator": result.get("orchestrator"),
                "validation_attempts": result.get("validation_attempts"),
            }
        )

    latencies = [m["end_to_end_latency_seconds"] for m in measurements if m["end_to_end_latency_seconds"] is not None]
    valid_count = sum(1 for m in measurements if m["is_schema_valid"])
    return {
        "runs": runs,
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "max_latency_seconds": max(latencies) if latencies else None,
        "valid_json_success_rate": valid_count / runs if runs else 0,
        "acceptance_under_60s": all(t < 60 for t in latencies),
        "measurements": measurements,
    }
