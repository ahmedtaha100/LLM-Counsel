from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_latency_charts(analytics: dict[str, Any] | None) -> None:
    st.header("Latency Analysis")

    if not analytics:
        st.info("No latency data available. Make some queries first.")
        return

    latency = analytics.get("latency", {})
    p50 = latency.get("p50_ms", 0)
    p95 = latency.get("p95_ms", 0)
    p99 = latency.get("p99_ms", 0)
    avg = latency.get("avg_ms", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("p50", f"{p50:.0f}ms")
    c2.metric("p95", f"{p95:.0f}ms")
    c3.metric("p99", f"{p99:.0f}ms")
    c4.metric("Avg", f"{avg:.0f}ms")

    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Average Latency (ms)"},
            gauge={
                "axis": {"range": [None, max(p99 * 1.2, 1000)]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, p50], "color": "lightgreen"},
                    {"range": [p50, p95], "color": "yellow"},
                    {"range": [p95, p99], "color": "orange"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": p99,
                },
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latency Percentiles")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=["p50", "p95", "p99", "avg"],
            y=[p50, p95, p99, avg],
            marker_color=["green", "yellow", "orange", "blue"],
        )
    )
    fig2.update_layout(
        title="Latency Breakdown",
        xaxis_title="Percentile",
        yaxis_title="Latency (ms)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    total_queries = analytics.get("total_queries", 0)
    panel_rate = analytics.get("panel_usage_rate", 0)

    st.subheader("Query Statistics")
    q1, q2 = st.columns(2)
    q1.metric("Total Queries", total_queries)
    q2.metric("Panel Mode Rate", f"{panel_rate * 100:.1f}%")
