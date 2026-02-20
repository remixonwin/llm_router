# Workspace Instructions for Intelligent LLM Router

These guidelines help an AI-powered agent or developer quickly understand and contribute to this repository.

---
## 🚦 Quick Start for Agents

1. **Environment setup**
   - Copy `.env.example` to `.env` and populate provider API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`, etc.).
   - The Python services use [uv](https://docs.astral.sh/uv/) as the task runner; install it and use `uv` commands instead of raw `python`/`pip` where documented.
   - The CLI lives under `cli/` and is a separate Node project (uses Ink/React).

2. **Common commands**
   ```bash
   # install Python deps (via uv/poetry internally)
   uv run install  # or `uv run pytest` to run tests

   # run the router server locally
   uv run llm-router

   # run Python test suite
   uv run pytest

   # run CLI
   cd cli && npm install && npm start
   ```

3. **Architecture overview**
   - `src/llm_router/`: core Python logic (router.py, discovery.py, quota.py, models.py, cache.py, etc.)
   - `tests/`: Python unit/integration tests; run with `uv run pytest`.
   - `cli/`: JavaScript CLI client demonstrating usage of the router API.
   - `scripts/`: small utility Python scripts (e.g. inspect_models, manage_port).

4. **API details**
   - The router exposes an OpenAI-compatible HTTP API on port `7544` by default.
   - Main endpoints: `/v1/chat/completions`, `/v1/embeddings`, `/v1/vision`, `/v1/models`, `/providers`.
   - Routing strategies include `auto`, `cost_optimized`, `quality_first`, `latency_first`.

5. **Testing and CI**
   - All tests live in `tests/` and may be executed by the `pytest` runner.
   - Integration tests assume a running router server on localhost.
   - Node tests (in `cli/test`) run via `npm test`.

6. **Conventions & patterns**
   - Follow PEP8 and use type hints; the project uses `python.analysis` settings configured by Pylance.
   - Use `uv` tasks defined in `pyproject.toml`; avoid manual `python -m` invocations unless necessary.
   - Python modules use absolute imports from `llm_router` package (see `src/llm_router/__init__.py`).
   - CLI code is a minimal Ink application; it talks to the router backend via `LLM_ROUTER_URL` env var (default `http://localhost:8080`).

7. **Common pitfalls**
   - Forgetting to populate `.env` causes provider discovery failures.
   - Ensure the router is running before executing integration tests.
   - When editing the CLI, `npm install` is required after adding dependencies.
   - Some tests rely on cached models; running `tests/test_discovery_remove_model.py` may require cleanup of local cache file.

8. **Where to explore for agent-specific tasks**
   - `router.py`: how requests are routed and failover logic.
   - `discovery.py`: model discovery from providers.
   - `quota.py`: quota tracking implementation.
   - `cache.py`: disk and semantic caching details.

---

Feel free to ask me for help with generating new features, debugging failing tests, or navigating the code!
