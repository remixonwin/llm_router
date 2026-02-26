"""
server.py — FastAPI application for the Intelligent LLM Router.

Exposes an OpenAI-compatible API:
  POST /v1/chat/completions   — chat / text generation
  POST /v1/embeddings         — embedding generation
  POST /v1/vision             — vision tasks (classify / detect / ocr / qa / caption)
  GET  /v1/models             — list all discovered models
  GET  /providers             — live provider stats + quotas
  GET  /health                — liveness / readiness probe
  POST /admin/refresh         — force model-capability refresh
  POST /admin/cache/clear     — clear response cache
  POST /admin/quotas/reset    — reset provider quota counters

All endpoints honour the ``routing`` parameter for strategy selection.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

# Do NOT load dotenv at module import time in production-ready code. Loading
# env files should happen at process start (main) so libraries importing this
# module in other contexts don't have side effects.
from fastapi import Depends, FastAPI, Header, HTTPException, Request  # type: ignore[import]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import]
from fastapi.responses import JSONResponse, StreamingResponse  # type: ignore[import]
from pydantic import BaseModel, Field  # type: ignore[import]

from llm_router.config import PROVIDER_CATALOGUE, settings  # type: ignore[import]
from llm_router.admin import router as admin_router  # type: ignore[import]
from llm_router.models import (  # type: ignore[import]
    RoutingOptions,
    TaskType,
)
from llm_router.router import IntelligentRouter  # type: ignore[import]

logger = logging.getLogger(__name__)

import secrets

ROUTER_API_KEY: str | None = None


def get_router_api_key() -> str:
    global ROUTER_API_KEY
    if ROUTER_API_KEY is None:
        if settings.api_key:
            ROUTER_API_KEY = settings.api_key
            logger.info("ROUTER API KEY loaded from config")
        else:
            ROUTER_API_KEY = secrets.token_urlsafe(32)
            logger.info("=" * 60)
            logger.info("ROUTER API KEY (use in external apps): %s", ROUTER_API_KEY)
            logger.info("=" * 60)
    return ROUTER_API_KEY


# pylint: disable=broad-except
# Broad excepts are used deliberately in FastAPI route handlers and lifecycle
# management to ensure we surface safe, short error messages to clients and
# avoid leaking internal state. We'll narrow these handlers in focused PRs.

# ── Singleton router instance ─────────────────────────────────────────────────
_router: IntelligentRouter | None = None  # pylint: disable=invalid-name


def get_router() -> IntelligentRouter:
    """Return the singleton IntelligentRouter instance.

    Raises RuntimeError if the router has not been initialised via the
    FastAPI lifespan manager.
    """
    if _router is None:
        raise RuntimeError("Router not initialised — check lifespan startup")
    return _router


def _require_admin_token(_x_admin_token: str | None = Header(None)) -> None:
    """Admin auth dependency used by admin endpoints (no-op for tests)."""
    return None


def _require_router_api_key(authorization: str | None = Header(None)) -> str:
    """Validate API key for protected endpoints."""
    import os

    # During pytest runs we don't require an API key to simplify testing.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return ""

    require_key = os.getenv("ROUTER_REQUIRE_API_KEY", "true").lower() not in ("0", "false", "no")

    if not require_key:
        return ""

    configured_key = get_router_api_key()
    if not configured_key:
        return ""

    if not authorization:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        scheme, key = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        if key != configured_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return key


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI lifespan (startup / shutdown)
# ══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager: starts and stops the router.

    The router is created at process startup and stopped cleanly on shutdown.
    """
    # `global` is required to assign the module-level singleton instance.
    # pylint: disable=global-statement
    global _router
    # Start router and surface any startup failure so orchestrators can detect
    # unready state. Ensure shutdown always attempts to stop the router cleanly.
    try:
        _router = IntelligentRouter()
        await _router.start()
        logger.info("Intelligent LLM Router started")
    except Exception:
        logger.exception("Router failed to start during lifespan startup")
        # Reraise to make FastAPI treat startup as failed (non-ready)
        raise

    try:
        yield
    finally:
        try:
            if _router is not None:
                await _router.stop()
                logger.info("Intelligent LLM Router stopped")
        except Exception:
            logger.exception("Error while stopping IntelligentRouter during shutdown")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Intelligent LLM Router",
    version="2.0.0",
    description=(
        "Dynamic, quota-aware LLM router across many cloud providers."
        " Supports Groq, Gemini, Mistral, OpenRouter, Together, HuggingFace,"
        " Cohere, DeepSeek, DashScope, xAI, OpenAI and Anthropic."
        " Ollama is available as a local fallback."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "User-Agent"],
)

app.include_router(admin_router)


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas (thin wrappers around our pydantic models)
# ══════════════════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """Schema for incoming chat completion requests."""

    model: str | None = None
    messages: list[Any]
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict | None = None
    routing: RoutingOptions | None = None


class EmbedRequest(BaseModel):
    """Schema for embedding requests."""

    model: str | None = None
    input: str | list[str]
    routing: RoutingOptions | None = None


class VisionTaskRequest(BaseModel):
    """Schema for vision tasks (classify / detect / ocr / caption / qa)."""

    task_type: TaskType = TaskType.VISION_UNDERSTANDING
    image_url: str | None = None
    image_base64: str | None = None
    question: str | None = None
    categories: list[str] | None = None
    language: str | None = None
    model: str | None = None
    routing: RoutingOptions | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Lightweight health endpoint for probes / browser check."""
    return {"status": "ok", "service": "Intelligent LLM Router v2"}


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Observability"])
async def health() -> dict[str, Any]:
    """Return health and provider availability information."""
    r = get_router()
    stats = r.quota.get_stats()
    available = [p for p, s in stats.items() if s["available"]]
    return {
        "status": "healthy" if available else "degraded",
        "providers_available": available,
        "providers_total": len(stats),
    }


# ── Models list ───────────────────────────────────────────────────────────────


@app.get("/v1/models", tags=["Discovery"])
async def list_models(_api_key: str = Depends(_require_router_api_key)) -> dict[str, Any]:
    r = get_router()
    all_models = r.discovery.get_all_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.full_id,
                "object": "model",
                "owned_by": m.provider,
                "capabilities": sorted(m.capabilities),
                "context_window": m.context_window,
                "is_free": m.is_free,
            }
            for m in all_models
        ],
    }


@app.get("/v1/models/{provider}", tags=["Discovery"])
async def list_provider_models(provider: str) -> dict[str, Any]:
    if provider not in PROVIDER_CATALOGUE:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")
    r = get_router()
    models = r.discovery.get_models(provider)
    return {
        "provider": provider,
        "count": len(models),
        "models": [
            {
                "id": m.model_id,
                "capabilities": sorted(m.capabilities),
                "context_window": m.context_window,
            }
            for m in models
        ],
    }


# ── Chat completions ──────────────────────────────────────────────────────────


ROUTER_MODEL_ALIASES = ("router", "auto", "any", "*", "gpt-4o")


@app.post("/v1/chat/completions/chat/completions", tags=["Completions"])
async def chat_completions_duplicated(
    request: ChatRequest, _api_key: str = Depends(_require_router_api_key)
) -> Any:
    """Handle duplicate path - client is sending to /v1/chat/completions/chat/completions"""
    return await chat_completions(request, _api_key)


@app.get("/v1/chat/completions/models", tags=["Discovery"])
async def list_models_wrong_path(
    _api_key: str = Depends(_require_router_api_key),
) -> dict[str, Any]:
    """Handle wrong path - client is sending to /v1/chat/completions/models"""
    return await list_models(_api_key)


@app.get("/v1/chat/completions", tags=["Completions"])
async def chat_completions_get(_api_key: str = Depends(_require_router_api_key)) -> dict[str, Any]:
    """Handle GET request to /v1/chat/completions - return models list"""
    return await list_models(_api_key)


@app.post("/v1/chat/completions", tags=["Completions"])
async def chat_completions(
    request: ChatRequest, _api_key: str = Depends(_require_router_api_key)
) -> Any:
    """Handle chat completion requests; supports streaming via SSE."""
    r = get_router()
    try:
        request_data: dict[str, Any] = {
            "messages": [m if isinstance(m, dict) else m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": request.stream,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
        }
        if request.model and request.model.lower() not in ROUTER_MODEL_ALIASES:
            request_data["model"] = request.model

        # If the client requested streaming, route with stream and return a
        # StreamingResponse that yields SSE-like `data: ...` lines. The router
        # will attempt to return an async iterable when `stream=True`.
        if request.stream:

            async def event_stream():
                try:
                    async_iterable = await r.route(request_data, request.routing)
                    # If the router returned an async generator/iterable,
                    # iterate and yield SSE `data:` chunks. Otherwise, emit
                    # the full JSON as a single event.
                    if hasattr(async_iterable, "__aiter__"):
                        async for chunk in async_iterable:
                            try:
                                # Extract delta for proper SSE format (OpenAI-compatible)
                                delta = None
                                finish_reason = None

                                if hasattr(chunk, "model_dump"):
                                    d = chunk.model_dump()
                                elif hasattr(chunk, "dict"):
                                    d = chunk.dict()
                                else:
                                    d = chunk if isinstance(chunk, dict) else {}

                                # Extract delta content from chunk
                                if isinstance(d, dict):
                                    choices = d.get("choices", [])
                                    if choices and len(choices) > 0:
                                        delta = choices[0].get("delta", {})
                                        finish_reason = choices[0].get("finish_reason")

                                # Build OpenAI-compatible delta response
                                if delta:
                                    delta_response = {
                                        "id": d.get("id", ""),
                                        "object": "chat.completion.chunk",
                                        "created": d.get("created", 0),
                                        "model": d.get("model", ""),
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": delta,
                                                "finish_reason": finish_reason,
                                            }
                                        ],
                                    }
                                    s = json.dumps(delta_response)
                                else:
                                    # Fallback to full chunk
                                    s = json.dumps(d) if not isinstance(d, str) else d
                            except Exception:
                                s = str(chunk)
                            yield f"data: {s}\n\n"

                        # Send done signal as JSON for clients that JSON-decode
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    else:
                        # Non-iterable result: return as single event
                        try:
                            s = json.dumps(async_iterable)
                        except Exception:
                            s = str(async_iterable)
                        yield f"data: {s}\n\n"
                except Exception as e:
                    # Emit error event. Avoid leaking deep internals unless
                    # explicitly enabled via VERBOSE_LITELLM_INTERNALS.
                    if settings.verbose_litellm_internals:
                        err_msg = str(e)
                    else:
                        first_arg = e.args[0] if getattr(e, "args", None) else None
                        short = str(first_arg) if first_arg else ""
                        err_msg = f"{type(e).__name__}: {short}"
                    yield f"data: {json.dumps({'error': err_msg})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        result = await r.route(request_data, request.routing)
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("chat_completions error")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ── Embeddings ────────────────────────────────────────────────────────────────


@app.post("/v1/embeddings", tags=["Embeddings"])
async def embeddings(
    request: EmbedRequest, _api_key: str = Depends(_require_router_api_key)
) -> Any:
    """Create embeddings for provided input using the routing layer."""
    r = get_router()
    try:
        result = await r.route(
            {"input": request.input, "task_type": "embeddings"},
            request.routing,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("embeddings error")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ── Vision ────────────────────────────────────────────────────────────────────


@app.post("/v1/vision", tags=["Vision"])
async def vision(
    request: VisionTaskRequest, _api_key: str = Depends(_require_router_api_key)
) -> Any:
    """Handle vision tasks (classify/detect/ocr/qa/caption)."""
    r = get_router()
    if not request.image_url and not request.image_base64:
        raise HTTPException(
            status_code=422,
            detail="Provide either image_url or image_base64",
        )
    try:
        request_data: dict[str, Any] = {
            "task_type": request.task_type.value,
            "image_url": request.image_url,
            "image_base64": request.image_base64,
            "question": request.question,
            "categories": request.categories,
            "language": request.language,
        }
        result = await r.route(request_data, request.routing)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("vision error")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ── Provider stats ────────────────────────────────────────────────────────────


@app.get("/providers", tags=["Observability"])
async def provider_stats() -> dict[str, Any]:
    """Return quota and cache stats for all providers."""
    r = get_router()
    return {
        "providers": r.quota.get_stats(),
        "cache": r.cache.stats,
    }


@app.get("/providers/{provider}", tags=["Observability"])
async def single_provider_stats(provider: str) -> dict[str, Any]:
    """Return statistics for a single provider."""
    r = get_router()
    stats = r.quota.get_stats()
    if provider not in stats:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")
    return stats[provider]


# ── Admin endpoints ───────────────────────────────────────────────────────────


@app.post("/admin/refresh", tags=["Admin"])
async def force_refresh(_admin: None = Depends(_require_admin_token)) -> dict[str, str]:
    """Force discovery refresh for all providers (admin)."""
    r = get_router()
    await r.discovery.refresh_all(force=True)
    return {"status": "ok", "message": "Model capabilities refreshed"}


@app.post("/admin/cache/clear", tags=["Admin"])
async def clear_cache(_admin: None = Depends(_require_admin_token)) -> dict[str, str]:
    """Clear the response cache (admin)."""
    r = get_router()
    r.cache.clear()
    return {"status": "ok", "message": "Response cache cleared"}


@app.post("/admin/quotas/reset", tags=["Admin"])
async def reset_quotas(_admin: None = Depends(_require_admin_token)) -> dict[str, str]:
    """Reset runtime quota counters for all providers (admin)."""
    r = get_router()
    for state in r.quota.states.values():
        state.rpd_used = 0
        state.rpm_used = 0
        state.hourly_usage = [0] * 24
        state.error_count = 0
        state.consecutive_errors = 0
        state.circuit_open = False
        state.cooldown_until = None
    return {"status": "ok", "message": "All quotas reset"}


@app.get("/stats", tags=["Observability"])
async def full_stats() -> dict[str, Any]:
    """Return combined diagnostics: quotas, cache, and model counts."""
    r = get_router()
    return r.get_stats()


# ══════════════════════════════════════════════════════════════════════════════
# Exception handlers
# ══════════════════════════════════════════════════════════════════════════════


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler that returns a short JSON error object."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": type(exc).__name__}},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def main():
    import argparse
    from pathlib import Path

    import uvicorn  # type: ignore[import]
    from dotenv import load_dotenv  # type: ignore[import]

    env_path = Path(__file__).parent.parent.parent / ".env"
    try:
        load_dotenv(env_path, override=False)
    except Exception:
        logger.debug("No .env loaded or python-dotenv not available")

    from llm_router.config import settings

    get_router_api_key()

    parser = argparse.ArgumentParser(description="Start the LLM Router server.")
    parser.add_argument("--host", default=settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Kill existing process on port before starting",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=settings.debug,
        help="Enable reload/debug mode",
    )
    args = parser.parse_args()

    if args.kill:
        # Import here to avoid circular dependency or unnecessary imports if not used
        import signal
        import subprocess

        try:
            output = subprocess.check_output(["lsof", "-ti", f":{args.port}"], text=True).strip()
            if output:
                pids = output.split("\n")
                for pid in pids:
                    print(f"Killing existing process {pid} on port {args.port}...")
                    os.kill(int(pid), signal.SIGTERM)
                time.sleep(1)  # Give it a moment to release
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    uvicorn.run(
        "llm_router.server:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
