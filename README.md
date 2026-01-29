# LLM Counsel

LLM Counsel is a FastAPI service that routes queries to one or more language models, aggregates responses, and returns a final answer with confidence and metrics.

## How it works

- Classifies query complexity and capabilities.
- Routes to a single model or a multi-model panel.
- Optionally detects dissent between model responses.
- Uses a semantic cache to speed up repeated queries.
- Tracks cost and latency metrics for analytics.

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
TOGETHER_API_KEY=...
```

## Run

```bash
uvicorn api.main:app --reload
```

## Use

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

Response includes:

- `response`
- `model_used`
- `confidence`
- `latency_ms`
- `cost_usd`
- `dissent` when panel mode is used

## Analytics

- `GET /analytics` for cost and latency summaries
- `GET /analytics/recent` for recent query history
