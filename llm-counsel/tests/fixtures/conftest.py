import pytest

from tests.fixtures.mock_responses import MOCK_RESPONSES
from tests.fixtures.sample_queries import SAMPLE_QUERIES


@pytest.fixture
def mock_responses():
    return MOCK_RESPONSES


@pytest.fixture
def sample_queries():
    return SAMPLE_QUERIES
