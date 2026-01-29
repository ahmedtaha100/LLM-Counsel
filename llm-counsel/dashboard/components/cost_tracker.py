from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def render_cost_tracker(analytics: dict[str, Any] | None) -> None:
    st.header("Cost Analysis")

    if not analytics:
        st.info("No analytics data available. Make some queries first.")
        return

    cost = analytics.get("cost", {})
    total = cost.get("total_cost", 0)
    gpt4_baseline = cost.get("estimated_gpt4o_cost", 0)
    savings = cost.get("cost_savings", 0)
    total_queries = analytics.get("total_queries", 0)
    avg = total / total_queries if total_queries > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost", f"${total:.4f}")
    c2.metric("Avg per Query", f"${avg:.4f}")
    c3.metric("GPT-4o Baseline", f"${gpt4_baseline:.4f}")
    c4.metric(
        "Savings",
        f"${savings:.4f}",
        delta=f"{(savings / gpt4_baseline * 100):.1f}%" if gpt4_baseline > 0 else None,
    )

    cost_by_model = cost.get("cost_by_model", {})
    if cost_by_model:
        df = pd.DataFrame([{"Model": k, "Cost": v} for k, v in cost_by_model.items()])
        fig = px.pie(df, values="Cost", names="Model", title="Cost by Model")
        st.plotly_chart(fig, use_container_width=True)

    queries_by_model = analytics.get("queries_by_model", {})
    selections_by_model = analytics.get("selections_by_model", {})

    if queries_by_model:
        st.subheader("Model Usage")

        col1, col2 = st.columns(2)

        with col1:
            df_queries = pd.DataFrame(
                [{"Model": k, "Queries": v} for k, v in queries_by_model.items()]
            )
            fig_q = px.bar(df_queries, x="Model", y="Queries", title="Total Queries by Model")
            st.plotly_chart(fig_q, use_container_width=True)

        with col2:
            if selections_by_model:
                df_sel = pd.DataFrame(
                    [{"Model": k, "Selections": v} for k, v in selections_by_model.items()]
                )
                fig_s = px.bar(df_sel, x="Model", y="Selections", title="Times Selected as Best")
                st.plotly_chart(fig_s, use_container_width=True)

    panel_rate = analytics.get("panel_usage_rate", 0)
    cache_hit_rate = analytics.get("cache", {}).get("hit_rate", 0)

    st.subheader("Efficiency Metrics")
    e1, e2 = st.columns(2)
    e1.metric("Panel Usage Rate", f"{panel_rate * 100:.1f}%")
    e2.metric("Cache Hit Rate", f"{cache_hit_rate * 100:.1f}%")
