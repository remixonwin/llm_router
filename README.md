# Intelligent LLM Router (backend)

Quick start (local)

- Copy environment template and add provider API keys:

  ```bash
  cp .env.example .env
  # edit .env and add API keys
  ```

- Start locally:

  ```bash
  python -m llm_router.server
  # or with uvicorn:
  python -m uvicorn llm_router.server:app --port 7544
  ```

Run with Docker

- Build image:

  ```bash
  docker build -t llm-router .
  ```

- Run container:

  ```bash
  docker run -p 7544:7544 --env-file .env llm-router
  ```

Tests

- Run tests: `pytest -q`

This README provides only basic commands to run the backend.
