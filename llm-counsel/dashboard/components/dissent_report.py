from typing import Any

import streamlit as st


def render_dissent_report(queries: list[dict[str, Any]]) -> None:
    st.header("Panel Analysis")

    panel_queries = [q for q in queries if q.get("panel_mode")]

    if not panel_queries:
        st.info("No panel queries recorded yet")
        return

    total = len(queries)
    panel_count = len(panel_queries)

    c1, c2 = st.columns(2)
    c1.metric("Panel Queries", panel_count)
    c2.metric("Panel Rate", f"{panel_count / total * 100:.1f}%" if total > 0 else "0%")

    st.subheader("Recent Panel Queries")
    for i, q in enumerate(reversed(panel_queries[-10:])):
        models = q.get("models", [])
        selected = q.get("selected_model", "N/A")
        query_text = q.get("query", "")
        display_query = query_text[:50] + "..." if len(query_text) > 50 else query_text

        with st.expander(f"Panel Query #{len(panel_queries) - i}: {display_query}"):
            st.write(f"**Models consulted:** {', '.join(models)}")
            st.write(f"**Selected:** {selected}")

            cost_by_model = q.get("cost_by_model", {})
            if cost_by_model:
                st.markdown("**Cost per Model:**")
                for model, cost in cost_by_model.items():
                    marker = " [selected]" if model == selected else ""
                    st.write(f"  - {model}{marker}: ${cost:.6f}")

            st.write(f"**Total Cost:** ${q.get('total_cost_usd', 0):.6f}")
            st.write(f"**Latency:** {q.get('latency_ms', 0):.0f}ms")

    st.markdown("---")
    st.info(
        "Tip: Dissent details are shown in the Query tab when you submit a panel query. "
        "They include divergent claims, consensus claims, and unique claims per model."
    )
