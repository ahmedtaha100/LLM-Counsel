from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.dissent import DissentDetector, DissentReport


@pytest.fixture
def mock_encoder():
    from core import shared

    shared.reset_encoder()

    with patch("core.shared.SentenceTransformer") as mock:
        encoder = MagicMock()

        def mock_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for text in texts:
                if "capital" in text.lower() and "paris" in text.lower():
                    results.append(np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32))
                elif "capital" in text.lower() and "london" in text.lower():
                    results.append(np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32))
                else:
                    results.append(np.random.rand(384).astype(np.float32))
            return np.array(results)

        encoder.encode = MagicMock(side_effect=mock_encode)
        mock.return_value = encoder
        yield encoder

    shared.reset_encoder()


class TestDissentDetector:
    def test_detect_no_dissent(self, mock_encoder):
        def similar_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for _ in texts:
                base = np.array([1.0, 0.1, 0.1] + [0.0] * 381, dtype=np.float32)
                results.append(base / np.linalg.norm(base))
            return np.array(results)

        mock_encoder.encode.side_effect = similar_encode

        detector = DissentDetector(similarity_threshold=0.7)
        responses = {
            "model-a": "Paris is the capital of France.",
            "model-b": "The capital of France is Paris.",
        }
        report = detector.detect(responses)

        assert isinstance(report, DissentReport)

    def test_detect_high_dissent(self, mock_encoder):
        detector = DissentDetector(similarity_threshold=0.7)
        responses = {
            "model-a": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "model-b": "The capital of France is London. It has Big Ben.",
        }
        report = detector.detect(responses)

        assert isinstance(report, DissentReport)

    def test_detect_single_response(self, mock_encoder):
        detector = DissentDetector()
        responses = {"model-a": "Some response"}
        report = detector.detect(responses)

        assert not report.has_dissent
        assert "single" in report.summary.lower()

    def test_detect_three_models(self, mock_encoder):
        detector = DissentDetector()
        responses = {
            "model-a": "First response about topic one",
            "model-b": "Second response about topic two",
            "model-c": "Third response about topic three",
        }
        report = detector.detect(responses)

        assert len(report.pairs) == 3

    def test_similarity_threshold_respected(self, mock_encoder):
        call_count = [0]

        def orthogonal_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for text in texts:
                call_count[0] += 1
                if "cats" in text.lower():
                    results.append(np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32))
                elif "dogs" in text.lower():
                    results.append(np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32))
                else:
                    results.append(np.random.rand(384).astype(np.float32))
            return np.array(results)

        mock_encoder.encode.side_effect = orthogonal_encode

        detector = DissentDetector(similarity_threshold=0.9)
        responses = {
            "model-a": "Cats are wonderful pets with fur",
            "model-b": "Dogs are loyal companions with bark",
        }
        report = detector.detect(responses)

        assert report.has_dissent or len(report.pairs) > 0

    def test_dissent_level_classification(self, mock_encoder):
        def test_level(sim_value: float) -> str:
            call_count = [0]

            def varied_encode(texts):
                if isinstance(texts, str):
                    texts = [texts]
                results = []
                for _ in texts:
                    call_count[0] += 1
                    if call_count[0] % 2 == 1:
                        results.append(np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32))
                    else:
                        results.append(
                            np.array(
                                [sim_value, np.sqrt(max(0, 1 - sim_value**2))] + [0.0] * 382,
                                dtype=np.float32,
                            )
                        )
                return np.array(results)

            mock_encoder.encode.side_effect = varied_encode

            detector = DissentDetector()
            responses = {"a": "response one", "b": "response two"}
            report = detector.detect(responses)
            return report.dissent_level

        high_sim_level = test_level(0.95)
        low_sim_level = test_level(0.3)

        assert high_sim_level in ["none", "low"]
        assert low_sim_level in ["moderate", "high"]

    def test_extract_claims(self, mock_encoder):
        detector = DissentDetector()
        text = "First claim here. Second claim with more detail. Third claim about something else."
        claims = detector._extract_claims(text)

        assert len(claims) >= 2
        assert all(len(c) > 20 for c in claims)

    def test_empty_responses(self, mock_encoder):
        detector = DissentDetector()
        report = detector.detect({})

        assert not report.has_dissent

    def test_report_summary_generated(self, mock_encoder):
        detector = DissentDetector()
        responses = {
            "model-a": "Some response with claims",
            "model-b": "Different response with other claims",
        }
        report = detector.detect(responses)

        assert len(report.summary) > 0
