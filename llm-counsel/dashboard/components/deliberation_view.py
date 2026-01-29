from typing import Any

import streamlit as st


def render_deliberation_view(queries: list[dict[str, Any]]) -> None:
    st.subheader("Recent Queries")

    if not queries:
        st.info("No queries submitted yet")
        return

    for i, q in enumerate(reversed(queries[-10:])):
        query_text = q.get("query", "")
        display_query = query_text[:50] + "..." if len(query_text) > 50 else query_text

        with st.expander(f"Query {len(queries) - i}: {display_query}"):
            models = q.get("models", [])
            selected = q.get("selected_model", "N/A")

            st.write(f"**Selected Model:** {selected}")
            if len(models) > 1:
                st.write(f"**All Models:** {', '.join(models)}")
            st.write(f"**Latency:** {q.get('latency_ms', 0):.0f}ms")
            st.write(f"**Total Cost:** ${q.get('total_cost_usd', 0):.6f}")

            if q.get("cache_hit"):
                st.success("Cache Hit")

            if q.get("panel_mode"):
                st.info("Panel Mode")

                cost_by_model = q.get("cost_by_model", {})
                if cost_by_model:
                    st.markdown("**Cost Breakdown:**")
                    for model, cost in cost_by_model.items():
                        marker = " [selected]" if model == selected else ""
                        st.write(f"  - {model}{marker}: ${cost:.6f}")
