.PHONY: install test test-all test-cov lint format typecheck ci run dashboard docker clean

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/unit -v

test-all:
	pytest tests/ -v

test-cov:
	pytest tests/unit --cov=. --cov-report=html --cov-fail-under=85

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

typecheck:
	mypy .

ci: lint typecheck test-cov

run:
	uvicorn api.main:app --reload

dashboard:
	streamlit run dashboard/app.py

docker:
	docker-compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
