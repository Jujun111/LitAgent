from pydantic import BaseModel, Field
from typing import List

# ----------------------------------------------------------------------
# 1. THE RESPONSE CONTRACT (What the AI guarantees to return)
# ----------------------------------------------------------------------
# Your teammate must configure vLLM's guided decoding to mathematically 
# guarantee that the LLM output exactly matches this Python class.
# You (the FSM builder) will write your LangGraph code assuming this 
# object is ALWAYS returned perfectly.

class ResearchSynthesis(BaseModel):
    """
    The formal data contract for the AI Summarization output.
    """
    paper_title: str = Field(
        ..., 
        description="The exact title of the academic paper."
    )
    executive_summary: str = Field(
        ..., 
        description="A concise, factual 3-4 sentence summary of the entire paper."
    )
    methodology_used: str = Field(
        ..., 
        description="A brief description of the research methods (e.g., RCT, Literature Review)."
    )
    key_findings: List[str] = Field(
        ..., 
        description="An array of 3 to 5 strictly factual bullet points extracting the core results."
    )
    is_hallucination_free: bool = Field(
        default=True,
        description="System flag to assert extraction fidelity."
    )

# ----------------------------------------------------------------------
# 2. THE REQUEST CONTRACT (What the FSM guarantees to send)
# ----------------------------------------------------------------------
# You must guarantee that your LangGraph FSM will always format the 
# HTTP POST request to your teammate's vLLM server in this exact way.

class SynthesisRequestPayload(BaseModel):
    """
    The formal data contract for the payload sent TO the AI microservice.
    """
    paper_id: str = Field(
        ..., 
        description="The unique Semantic Scholar ID."
    )
    chunked_text: str = Field(
        ..., 
        description="The sanitized, pre-processed text chunk from the paper (max 4000 tokens)."
    )
    max_latency_seconds: int = Field(
        default=60, 
        description="The timeout threshold (NfReq03) the AI service must respect."
    )

# ----------------------------------------------------------------------
# 3. MOCKING FOR PARALLEL DEVELOPMENT
# ----------------------------------------------------------------------
def mock_ai_response() -> ResearchSynthesis:
    """
    You (Software Engineer) will use this function in your LangGraph node 
    to simulate the AI while your teammate builds the actual vLLM server.
    """
    return ResearchSynthesis(
        paper_title="Attention Is All You Need",
        executive_summary="The paper proposes the Transformer architecture, dispensing with recurrence entirely.",
        methodology_used="Machine Learning Architecture Design",
        key_findings=[
            "Replaces RNNs and CNNs with self-attention mechanisms.",
            "Achieves state-of-the-art BLEU scores on translation tasks.",
            "Significantly reduces training time compared to sequential models."
        ]
    )