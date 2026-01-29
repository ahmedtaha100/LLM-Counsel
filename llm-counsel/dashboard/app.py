import os

import httpx
import streamlit as st

from dashboard.components.cost_tracker import render_cost_tracker
from dashboard.components.deliberation_view import render_deliberation_view
from dashboard.components.dissent_report import render_dissent_report
from dashboard.components.latency_charts import render_latency_charts

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="LLM Counsel", layout="wide")

st.title("LLM Counsel Dashboard")


def fetch_analytics():
    try:
        resp = httpx.get(f"{API_URL}/analytics", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_recent_queries(n: int = 50):
    try:
        resp = httpx.get(f"{API_URL}/analytics/recent?n={n}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


tab1, tab2, tab3, tab4 = st.tabs(["Query", "Costs", "Latency", "Dissent"])

with tab1:
    st.header("Submit Query")

    query = st.text_area("Enter your query", height=100)
    col1, col2 = st.columns(2)

    with col1:
        mode = st.selectbox("Mode", ["auto", "panel", "single"])
        budget = st.checkbox("Budget mode")

    with col2:
        max_tokens = st.slider("Max tokens", 100, 8000, 4096)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)

    if st.button("Submit", type="primary") and query:
        with st.spinner("Processing..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/query",
                    json={
                        "query": query,
                        "mode": mode,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "budget_mode": budget,
                    },
                    timeout=120,
                )
                result = resp.json()

                st.success(
                    f"Model: {result.get('model_used')} | Confidence: {result.get('confidence', 0):.2f}"
                )
                st.write(result.get("response", ""))

                if result.get("cache_hit"):
                    st.info("Served from cache")

                if result.get("all_responses"):
                    st.subheader("All Model Responses")
                    cols = st.columns(len(result["all_responses"]))
                    for i, (model, response) in enumerate(result["all_responses"].items()):
                        with cols[i]:
                            selected = " [selected]" if model == result.get("model_used") else ""
                            st.markdown(f"**{model}{selected}**")
                            st.text_area(
                                f"Response from {model}",
                                value=response,
                                height=200,
                                key=f"resp_{model}_{i}",
                                disabled=True,
                            )

                if result.get("dissent"):
                    dissent = result["dissent"]
                    st.warning(f"Dissent detected: {dissent.get('summary', '')}")

                    if dissent.get("pairs"):
                        st.subheader("Dissent Pairs")
                        for pair in dissent["pairs"]:
                            with st.expander(
                                f"{pair['model_a']} vs {pair['model_b']} "
                                f"(similarity: {pair['similarity']:.2f})"
                            ):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.markdown(f"**{pair['model_a']} unique claims:**")
                                    for claim in pair.get("divergent_claims_a", []):
                                        st.write(f"- {claim}")
                                with col_b:
                                    st.markdown(f"**{pair['model_b']} unique claims:**")
                                    for claim in pair.get("divergent_claims_b", []):
                                        st.write(f"- {claim}")

                    if dissent.get("consensus_claims"):
                        st.subheader("Consensus Claims")
                        for claim in dissent["consensus_claims"]:
                            st.write(f"[agreed] {claim}")

            except Exception as e:
                st.error(f"Error: {e}")

    recent = fetch_recent_queries()
    render_deliberation_view(recent)

with tab2:
    analytics = fetch_analytics()
    render_cost_tracker(analytics)

with tab3:
    analytics = fetch_analytics()
    render_latency_charts(analytics)

with tab4:
    recent = fetch_recent_queries(100)
    render_dissent_report(recent)
