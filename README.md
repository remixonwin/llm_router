# Intelligent LLM Router

A small, reliable backend that routes requests to multiple LLM providers.
It handles quota tracking, discovery of available models, and response caching.

Getting started

- Copy the environment template and add your provider keys:

  ```bash
  cp .env.example .env
  # edit .env and add API keys (e.g. GROQ_API_KEY)
  ```

- Start locally:

  ```bash
  python -m llm_router.server
  # or: python -m uvicorn llm_router.server:app --port 7544
  ```

Run in Docker

- Build and run:

  ```bash
  docker build -t llm-router .
  docker run -p 7544:7544 --env-file .env llm-router
  ```

API endpoints

- `POST /v1/chat/completions` — chat/text completions
- `POST /v1/embeddings` — embeddings
- `POST /v1/vision` — vision tasks
- `GET /v1/models` — list discovered models
- `GET /providers` — provider quota & latency stats

Tests

- Run the test suite locally with `pytest -q`

Project layout

- `src/llm_router/` — backend implementation
- `tests/` — unit and integration tests
