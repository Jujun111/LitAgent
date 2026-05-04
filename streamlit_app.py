import json

import streamlit as st

from litagent_backend import benchmark_pipeline, run_pipeline


PROVIDER_PRESETS = {
    "Mock": {
        "provider": "mock",
        "base_url": "",
        "api_key": "",
        "model": "",
        "structured_mode": "none",
    },
    "vLLM local": {
        "provider": "openai_compatible",
        "base_url": "http://localhost:8000/v1",
        "api_key": "token-abc123",
        "model": "jc-builds/Qwen3.5-9B-Q4_K_M-GGUF:Q4_K_M",
        "structured_mode": "vllm",
    },
    "Ollama local": {
        "provider": "openai_compatible",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen3:8b",
        "structured_mode": "json_object",
    },
    "LM Studio local": {
        "provider": "openai_compatible",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": "local-model",
        "structured_mode": "json_object",
    },
    "Custom OpenAI-compatible": {
        "provider": "openai_compatible",
        "base_url": "http://localhost:8000/v1",
        "api_key": "token-abc123",
        "model": "local-model",
        "structured_mode": "json_object",
    },
}


st.set_page_config(page_title="LitAgent", layout="wide")

st.title("LitAgent Research Dossier")

with st.sidebar:
    provider_label = st.selectbox("Model provider", list(PROVIDER_PRESETS.keys()))
    preset = PROVIDER_PRESETS[provider_label]

    use_mock_fetch = st.toggle("Use mock papers", value=True)
    use_langgraph = st.toggle("Use LangGraph FSM", value=True)
    fallback_to_mock = st.toggle("Fallback to mock papers", value=False)
    semantic_scholar_api_key = st.text_input("Semantic Scholar API key", type="password")
    max_context_chunks = st.slider("Context chunks", min_value=1, max_value=12, value=8)

    base_url = preset["base_url"]
    api_key = preset["api_key"]
    model = preset["model"]
    structured_mode = preset["structured_mode"]

    if preset["provider"] != "mock":
        base_url = st.text_input("Base URL", value=base_url)
        api_key = st.text_input("API key", value=api_key, type="password")
        model = st.text_input("Model", value=model)
        structured_mode = st.selectbox(
            "JSON mode",
            ["vllm", "json_object", "none"],
            index=["vllm", "json_object", "none"].index(structured_mode),
        )

query = st.text_input(
    "Research topic",
    value="graph neural networks for molecular property prediction",
)

run_col, bench_col = st.columns([1, 1])
run_clicked = run_col.button("Generate dossier", type="primary", use_container_width=True)
bench_clicked = bench_col.button("Run 5x benchmark", use_container_width=True)

if run_clicked:
    with st.status("Running LitAgent pipeline...", expanded=False):
        try:
            result = run_pipeline(
                query=query,
                provider=preset["provider"],
                use_mock_fetch=use_mock_fetch,
                fallback_to_mock_on_error=fallback_to_mock,
                semantic_scholar_api_key=semantic_scholar_api_key or None,
                base_url=base_url,
                api_key=api_key,
                model=model,
                structured_mode=structured_mode,
                max_context_chunks=max_context_chunks,
                use_langgraph=use_langgraph,
            )
            st.session_state["last_result"] = result
        except Exception as exc:
            st.session_state["last_result"] = {"is_schema_valid": False, "error": str(exc)}

if bench_clicked:
    with st.status("Benchmarking pipeline...", expanded=False):
        try:
            benchmark = benchmark_pipeline(
                query=query,
                runs=5,
                provider=preset["provider"],
                use_mock_fetch=use_mock_fetch,
                fallback_to_mock_on_error=fallback_to_mock,
                semantic_scholar_api_key=semantic_scholar_api_key or None,
                base_url=base_url,
                api_key=api_key,
                model=model,
                structured_mode=structured_mode,
                max_context_chunks=max_context_chunks,
                use_langgraph=use_langgraph,
            )
            st.session_state["last_benchmark"] = benchmark
        except Exception as exc:
            st.session_state["last_benchmark"] = {"error": str(exc)}

result = st.session_state.get("last_result")
benchmark = st.session_state.get("last_benchmark")

if result:
    metrics = st.columns(6)
    metrics[0].metric("Fetch", result.get("fetch_mode", "-"))
    metrics[1].metric("LLM", result.get("llm_mode", "-"))
    metrics[2].metric("FSM", result.get("orchestrator", "-"))
    metrics[3].metric("Valid JSON", str(result.get("is_schema_valid", False)))
    metrics[4].metric("Attempts", result.get("validation_attempts", 0))
    metrics[5].metric("Total sec", f"{result.get('end_to_end_latency_seconds', 0):.2f}")

    if result.get("error"):
        st.error(result["error"])

    tabs = st.tabs(["Dossier", "Raw JSON", "Prompt", "Papers"])
    with tabs[0]:
        dossier = result.get("dossier")
        if dossier:
            st.subheader(dossier.get("topic", "Research dossier"))
            st.write(dossier.get("summary", ""))
            st.divider()
            for paper in dossier.get("key_papers", []):
                with st.container(border=True):
                    st.markdown(f"**{paper.get('title', 'Untitled')}**")
                    st.caption(f"{paper.get('year') or 'Unknown year'} | {paper.get('venue') or 'Unknown venue'}")
                    st.write(paper.get("reason", ""))
                    findings = paper.get("key_findings", [])
                    if findings:
                        st.markdown("**Key findings**")
                        for finding in findings:
                            st.write(f"- {finding}")
            limitations = dossier.get("limitations", [])
            if limitations:
                st.markdown("**Limitations**")
                for limitation in limitations:
                    st.write(f"- {limitation}")
            disclaimer = dossier.get("disclaimer")
            if disclaimer:
                st.warning(disclaimer)
        else:
            st.info("No validated dossier yet. Check Raw JSON for model output.")

    with tabs[1]:
        st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")

    with tabs[2]:
        st.code(result.get("prompt", ""), language="text")

    with tabs[3]:
        st.json(result.get("raw_papers", []))

if benchmark:
    st.subheader("Benchmark")
    if benchmark.get("error"):
        st.error(benchmark["error"])
    else:
        cols = st.columns(4)
        cols[0].metric("Runs", benchmark["runs"])
        cols[1].metric("Avg sec", f"{benchmark['avg_latency_seconds']:.2f}")
        cols[2].metric("Max sec", f"{benchmark['max_latency_seconds']:.2f}")
        cols[3].metric("Valid rate", f"{benchmark['valid_json_success_rate']:.0%}")
        st.json(benchmark["measurements"])
