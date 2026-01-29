import pytest

from core.metrics import MetricsTracker


class TestMetricsTracker:
    @pytest.fixture
    def tracker(self):
        return MetricsTracker()

    def test_record_single_success(self, tracker):
        tracker.record_single(
            query="test query",
            model="gpt-4o",
            latency_ms=100,
            cost_usd=0.01,
            tokens_in=50,
            tokens_out=100,
            cache_hit=False,
            failed=False,
        )

        summary = tracker.summary()
        assert summary.total_queries == 1
        assert summary.total_cost == 0.01
        assert "gpt-4o" in summary.queries_by_model
        assert summary.queries_by_model["gpt-4o"] == 1
        assert "gpt-4o" in summary.selections_by_model
        assert summary.selections_by_model["gpt-4o"] == 1
        assert summary.failures_by_model.get("gpt-4o", 0) == 0

    def test_record_single_failure(self, tracker):
        tracker.record_single(
            query="test query",
            model="gpt-4o",
            latency_ms=0,
            cost_usd=0,
            tokens_in=0,
            tokens_out=0,
            cache_hit=False,
            failed=True,
        )

        summary = tracker.summary()
        assert summary.total_queries == 1
        assert summary.total_cost == 0
        assert "gpt-4o" in summary.queries_by_model
        assert summary.queries_by_model["gpt-4o"] == 1
        assert "gpt-4o" in summary.failures_by_model
        assert summary.failures_by_model["gpt-4o"] == 1
        assert summary.selections_by_model.get("gpt-4o", 0) == 0

    def test_record_panel_with_failures(self, tracker):
        tracker.record(
            query="panel query",
            models=["gpt-4o", "claude-3-opus", "llama-3"],
            selected_model="gpt-4o",
            latency_ms=500,
            total_cost_usd=0.05,
            cost_by_model={"gpt-4o": 0.03, "claude-3-opus": 0.02},
            tokens_by_model={"gpt-4o": (100, 200), "claude-3-opus": (100, 150)},
            latency_by_model={"gpt-4o": 400, "claude-3-opus": 350},
            cache_hit=False,
            panel_mode=True,
            failed_models=["llama-3"],
        )

        summary = tracker.summary()
        assert summary.total_queries == 1
        assert "llama-3" in summary.failures_by_model
        assert summary.failures_by_model["llama-3"] == 1
        assert "gpt-4o" in summary.selections_by_model
        assert summary.selections_by_model["gpt-4o"] == 1

    def test_failures_by_model_accumulate(self, tracker):
        for _ in range(3):
            tracker.record_single(
                query="failing query",
                model="unstable-model",
                latency_ms=0,
                cost_usd=0,
                tokens_in=0,
                tokens_out=0,
                cache_hit=False,
                failed=True,
            )

        summary = tracker.summary()
        assert summary.failures_by_model["unstable-model"] == 3

    def test_mixed_success_and_failure(self, tracker):
        tracker.record_single(
            query="success",
            model="gpt-4o",
            latency_ms=100,
            cost_usd=0.01,
            tokens_in=50,
            tokens_out=100,
            cache_hit=False,
            failed=False,
        )

        tracker.record_single(
            query="failure",
            model="gpt-4o",
            latency_ms=0,
            cost_usd=0,
            tokens_in=0,
            tokens_out=0,
            cache_hit=False,
            failed=True,
        )

        summary = tracker.summary()
        assert summary.queries_by_model["gpt-4o"] == 2
        assert summary.selections_by_model["gpt-4o"] == 1
        assert summary.failures_by_model["gpt-4o"] == 1

    def test_none_selected_not_in_selections(self, tracker):
        tracker.record(
            query="all failed",
            models=["model-a", "model-b"],
            selected_model="none",
            latency_ms=100,
            total_cost_usd=0,
            cost_by_model={},
            tokens_by_model={},
            latency_by_model={},
            cache_hit=False,
            panel_mode=True,
            failed_models=["model-a", "model-b"],
        )

        summary = tracker.summary()
        assert "none" not in summary.selections_by_model

    def test_recent_queries_include_failures(self, tracker):
        tracker.record_single(
            query="failed query",
            model="gpt-4o",
            latency_ms=0,
            cost_usd=0,
            tokens_in=0,
            tokens_out=0,
            cache_hit=False,
            failed=True,
        )

        recent = tracker.recent(10)
        assert len(recent) == 1
        assert recent[0].failed_models == ["gpt-4o"]
        assert recent[0].selected_model == "none"

    def test_cache_hit_tracking(self, tracker):
        tracker.record_single(
            query="cached",
            model="gpt-4o",
            latency_ms=0,
            cost_usd=0,
            tokens_in=0,
            tokens_out=0,
            cache_hit=True,
        )

        summary = tracker.summary()
        assert summary.cache_hit_rate == 1.0

    def test_panel_usage_rate(self, tracker):
        tracker.record_single(
            query="single",
            model="gpt-4o",
            latency_ms=100,
            cost_usd=0.01,
            tokens_in=50,
            tokens_out=100,
            cache_hit=False,
        )

        tracker.record(
            query="panel",
            models=["gpt-4o", "claude-3-opus"],
            selected_model="gpt-4o",
            latency_ms=200,
            total_cost_usd=0.05,
            cost_by_model={"gpt-4o": 0.03, "claude-3-opus": 0.02},
            tokens_by_model={"gpt-4o": (100, 200), "claude-3-opus": (100, 150)},
            latency_by_model={"gpt-4o": 180, "claude-3-opus": 160},
            cache_hit=False,
            panel_mode=True,
        )

        summary = tracker.summary()
        assert summary.panel_usage_rate == 0.5

    def test_clear_resets_all(self, tracker):
        tracker.record_single(
            query="test",
            model="gpt-4o",
            latency_ms=100,
            cost_usd=0.01,
            tokens_in=50,
            tokens_out=100,
            cache_hit=False,
        )

        tracker.clear()
        summary = tracker.summary()

        assert summary.total_queries == 0
        assert summary.total_cost == 0
        assert len(summary.queries_by_model) == 0
        assert len(summary.failures_by_model) == 0
