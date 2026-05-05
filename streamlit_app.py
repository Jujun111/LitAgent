import json
from pathlib import Path

import requests
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
    "llama.cpp local": {
        "provider": "openai_compatible",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "llama-cpp",
        "model": "qwen3.5-9b-vlm",
        "structured_mode": "json_schema",
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


def format_demo_error(exc: Exception, provider_label: str, base_url: str) -> str:
    if isinstance(exc, requests.ConnectionError) and "127.0.0.1" in base_url:
        return (
            f"Could not connect to the local model server at {base_url}. "
            "Start llama.cpp in another terminal before using the llama.cpp local provider:\n\n"
            "powershell -NoProfile -ExecutionPolicy Bypass -File "
            ".\\llm_service\\serve_llamacpp_vision.ps1 "
            "llm_service\\config.llamacpp-vision.example.env\n\n"
            "For a no-model UI demo, choose Model provider = Mock and turn on Use mock papers."
        )
    if isinstance(exc, requests.ConnectionError):
        return (
            f"Could not connect to the selected provider '{provider_label}' at {base_url}. "
            "Start that OpenAI-compatible server first, or switch to Mock for an offline demo."
        )
    return str(exc)


st.set_page_config(page_title="LitAgent", layout="wide")

st.title("LitAgent Research Dossier")

with st.sidebar:
    provider_label = st.selectbox("Model provider", list(PROVIDER_PRESETS.keys()))
    preset = PROVIDER_PRESETS[provider_label]

    use_mock_fetch = st.toggle("Use mock papers", value=True)
    use_pdf_input = st.toggle("Use PDF paper", value=False)
    use_vision = st.toggle("Use pixel vision", value=False, disabled=not use_pdf_input)
    use_langgraph = st.toggle("Use LangGraph FSM", value=True)
    fallback_to_mock = st.toggle("Fallback to mock papers", value=False)
    semantic_scholar_api_key = st.text_input("Semantic Scholar API key", type="password")
    max_context_chunks = st.slider("Context chunks", min_value=1, max_value=12, value=8)
    uploaded_pdf = st.file_uploader("PDF upload", type=["pdf"], disabled=not use_pdf_input)
    local_pdf_path = st.text_input("Local PDF path", value="", disabled=not use_pdf_input)

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
            ["json_schema", "vllm", "json_object", "none"],
            index=["json_schema", "vllm", "json_object", "none"].index(structured_mode),
        )

    vision_base_url = base_url
    vision_api_key = api_key
    vision_model = model
    if use_pdf_input and use_vision:
        vision_base_url = st.text_input("Vision Base URL", value=vision_base_url)
        vision_api_key = st.text_input("Vision API key", value=vision_api_key, type="password")
        vision_model = st.text_input("Vision model", value=vision_model)

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
            fixed_papers = None
            fixed_fetch_mode = "fixed"
            effective_use_mock_fetch = use_mock_fetch
            if use_pdf_input:
                upload_dir = Path("evidence/streamlit_uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)
                if uploaded_pdf is not None:
                    pdf_path = upload_dir / uploaded_pdf.name
                    pdf_path.write_bytes(uploaded_pdf.getbuffer())
                elif local_pdf_path:
                    pdf_path = Path(local_pdf_path)
                else:
                    raise ValueError("PDF input is enabled but no PDF was provided.")
                from pdf_ingest import parse_pdf_to_paper_record

                fixed_papers = [
                    parse_pdf_to_paper_record(
                        pdf_path,
                        export_visuals=use_vision,
                        visual_output_dir=Path("evidence/streamlit_visual_crops") / Path(pdf_path).stem,
                        include_page_images=use_vision,
                    )
                ]
                fixed_fetch_mode = "pdf_vision_fixed" if use_vision else "pdf_fixed"
                effective_use_mock_fetch = False

            result = run_pipeline(
                query=query,
                provider=preset["provider"],
                use_mock_fetch=effective_use_mock_fetch,
                fallback_to_mock_on_error=fallback_to_mock,
                fixed_papers=fixed_papers,
                fixed_fetch_mode=fixed_fetch_mode,
                semantic_scholar_api_key=semantic_scholar_api_key or None,
                base_url=base_url,
                api_key=api_key,
                model=model,
                structured_mode=structured_mode,
                max_context_chunks=max_context_chunks,
                use_langgraph=use_langgraph,
                use_vision=use_vision,
                vision_base_url=vision_base_url,
                vision_api_key=vision_api_key,
                vision_model=vision_model,
            )
            st.session_state["last_result"] = result
        except Exception as exc:
            st.session_state["last_result"] = {
                "is_schema_valid": False,
                "error": format_demo_error(exc, provider_label, base_url),
            }

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
            st.session_state["last_benchmark"] = {"error": format_demo_error(exc, provider_label, base_url)}

result = st.session_state.get("last_result")
benchmark = st.session_state.get("last_benchmark")

if result:
    metrics = st.columns(7)
    metrics[0].metric("Fetch", result.get("fetch_mode", "-"))
    metrics[1].metric("LLM", result.get("llm_mode", "-"))
    metrics[2].metric("FSM", result.get("orchestrator", "-"))
    metrics[3].metric("Valid JSON", str(result.get("is_schema_valid", False)))
    metrics[4].metric("Attempts", result.get("validation_attempts", 0))
    metrics[5].metric("Total sec", f"{result.get('end_to_end_latency_seconds', 0):.2f}")
    metrics[6].metric("Vision assets", result.get("vision_asset_count", 0))

    if result.get("error"):
        st.error(result["error"])

    tabs = st.tabs(["Dossier", "Raw JSON", "Prompt", "Papers", "Chunks", "Visuals", "Sources"])
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

    with tabs[4]:
        chunks = result.get("chunks", [])
        if chunks:
            st.dataframe(
                [
                    {
                        "type": chunk.get("chunk_type"),
                        "page": chunk.get("page_number"),
                        "source": chunk.get("source_ref"),
                        "paper_id": chunk.get("paper_id"),
                        "text": chunk.get("text", "")[:500],
                    }
                    for chunk in chunks
                ],
                use_container_width=True,
            )
        else:
            st.info("No chunks available.")

    with tabs[5]:
        visual_assets = [
            asset
            for paper in result.get("raw_papers", [])
            for asset in paper.get("visual_assets", []) or []
        ]
        observations = result.get("vision_observations", [])
        if visual_assets:
            for asset in visual_assets[:12]:
                st.markdown(f"**{asset.get('source_ref', 'visual asset')}**")
                image_path = asset.get("image_path")
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption=asset.get("visual_type", "image"))
                st.json(
                    {
                        "page_number": asset.get("page_number"),
                        "visual_type": asset.get("visual_type"),
                        "bbox": asset.get("bbox"),
                    }
                )
        else:
            st.info("No visual assets available.")
        if observations:
            st.markdown("**Vision observations**")
            st.json(observations)

    with tabs[6]:
        finding_sources = result.get("finding_sources", [])
        if finding_sources:
            st.dataframe(finding_sources, use_container_width=True)
        else:
            st.info("No finding sources available.")

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
