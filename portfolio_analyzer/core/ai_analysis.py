"""
AI-powered portfolio commentary generator.
Three distinct analysis blocks for the PDF tear sheet:
1. metrics_commentary   — portfolio manager style analysis of risk/return metrics
2. allocation_commentary — comment on holdings concentration and sector/country mix
3. overall_recommendation — concise BUY / HOLD / REVIEW recommendation

Uses OpenAI gpt-4o-mini. Gracefully returns "" if no API key is set.
"""
from __future__ import annotations
import streamlit as st


def _get_client():
    """Return an OpenAI client, or None if key is not set."""
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key or api_key.strip().startswith("sk-..."):
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _call(client, prompt: str, max_tokens: int = 160) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[AI] Error: {e}")
        return ""


def generate_metrics_commentary(
    metrics: dict, 
    benchmark_name: str = "benchmark",
    tracking_error: float = None,
    information_ratio: float = None,
) -> str:
    """
    Senior portfolio manager style commentary on risk/return metrics.
    Structured in 3 paragraphs with market context. Uses \n\n to separate paragraphs.
    Placed below the metrics table on page 1.
    """
    client = _get_client()
    if not client:
        return ""

    import datetime
    today = datetime.date.today().strftime("%B %Y")
    
    # Flatten metrics
    m = ", ".join(f"{k}: {v}" for k, v in metrics.items())
    
    # Optional active metrics
    active_ctx = ""
    if tracking_error is not None and information_ratio is not None:
        active_ctx = f"Active Risk (Tracking Error): {tracking_error:.2%}, Information Ratio: {information_ratio:.2f}\n"

    prompt = f"""You are a senior portfolio manager at a major asset management firm writing the commentary section of a monthly tear sheet as of {today}.

Benchmark: {benchmark_name}
Portfolio Baseline Metrics: {m}
{active_ctx}

Write exactly 3 paragraphs separated by a blank line (\n\n). Each paragraph MUST be exactly 1 to 2 sentences max. Be extremely concise to fit a strict vertical page limit. Structure as follows:

Paragraph 1: Performance review. Comment on the CAGR and absolute return profile. CRITICAL: You must heavily contextualize this performance by referencing specific and recent macroeconomic events (e.g. recent central bank rate decisions, inflation trends, geopolitical news, or sector rallies of 2024-2025). Critically evaluate if the returns justify the Active Risk taken relative to the benchmark.

Paragraph 2: Risk analysis. Analyse the Sharpe ratio, maximum drawdown, and volatility. Contextualise these figures against what investors experienced during recent market stress periods or shock events. Draw a conclusion about the portfolio's drawdown protection and risk discipline.

Paragraph 3: Consistency and outlook. Comment on the consistency of returns (Information Ratio, Calmar, Sortino). Offer a forward-looking sentence framing the portfolio's expected behavior in the current macro and geopolitical environment.

Tone: authoritative, institutional, highly analytical, no hedging language. Write as if published in a factsheet distributed to institutional LPs.
Constraints: Plain text only. No bullet points. No markdown. No headers. DO NOT use the em-dash character ('—') anywhere in your response."""
    return _call(client, prompt, max_tokens=350)
