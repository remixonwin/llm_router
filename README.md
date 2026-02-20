# Intelligent LLM Router

Backend service that routes LLM requests across providers with quota
management, discovery, and caching.

Quick start

- Copy env template and add provider API keys: `cp .env.example .env`
- Run the server (it listens on port 7544 by default).

APIs (minimal)

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/vision`
- `GET /v1/models`
- `GET /providers`

Run locally

- Load env: `cp .env.example .env && edit .env` (set provider API keys)
- Start locally: `python -m llm_router.server` or `python -m uvicorn llm_router.server:app --port 7544`

Run with Docker

- Build: `docker build -t llm-router .`
- Run: `docker run -p 7544:7544 --env-file .env llm-router`

Run tests

- `pytest -q`

Project layout

- `src/llm_router/` — backend implementation
- `tests/` — unit and integration tests
