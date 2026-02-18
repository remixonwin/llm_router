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

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from dotenv import load_dotenv  # type: ignore[import]
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, status  # type: ignore[import]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import]
from fastapi.responses import JSONResponse, StreamingResponse  # type: ignore[import]
from pydantic import BaseModel, Field  # type: ignore[import]

from llm_router.config import PROVIDER_CATALOGUE, settings  # type: ignore[import]
from llm_router.models import (  # type: ignore[import]
    CachePolicy,
    ChatCompletionRequest,
    EmbeddingRequest,
    RoutingOptions,
    RoutingStrategy,
    TaskType,
    VisionRequest,
)
from llm_router.router import IntelligentRouter  # type: ignore[import]

logger = logging.getLogger(__name__)

# ── Singleton router instance ─────────────────────────────────────────────────
_router: Optional[IntelligentRouter] = None


def get_router() -> IntelligentRouter:
    if _router is None:
        raise RuntimeError("Router not initialised — check lifespan startup")
    return _router


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI lifespan (startup / shutdown)
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _router
    _router = IntelligentRouter()
    await _router.start()
    logger.info("Intelligent LLM Router started")
    yield
    await _router.stop()
    logger.info("Intelligent LLM Router stopped")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Intelligent LLM Router",
    version="2.0.0",
    description=(
        "Dynamic, quota-aware LLM router across Groq, Gemini, Mistral, OpenRouter, "
        "Together, HuggingFace, Cohere, DeepSeek, DashScope, xAI, OpenAI, Anthropic — "
        "with Ollama as strict local fallback."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas (thin wrappers around our pydantic models)
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    model:       Optional[str]                      = None
    messages:    List[Any]
    temperature: Optional[float]                    = Field(None, ge=0.0, le=2.0)
    max_tokens:  Optional[int]                      = Field(None, gt=0)
    top_p:       Optional[float]                    = Field(None, ge=0.0, le=1.0)
    stream:      bool                               = False
    tools:       Optional[List[Dict[str, Any]]]     = None
    tool_choice: Optional[Union[str, Dict]]         = None
    routing:     Optional[RoutingOptions]           = None


class EmbedRequest(BaseModel):
    model:   Optional[str]              = None
    input:   Union[str, List[str]]
    routing: Optional[RoutingOptions]  = None


class VisionTaskRequest(BaseModel):
    task_type:     TaskType                 = TaskType.VISION_UNDERSTANDING
    image_url:     Optional[str]            = None
    image_base64:  Optional[str]            = None
    question:      Optional[str]            = None
    categories:    Optional[List[str]]      = None
    language:      Optional[str]            = None
    model:         Optional[str]            = None
    routing:       Optional[RoutingOptions] = None


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:
    return {"status": "ok", "service": "Intelligent LLM Router v2"}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Observability"])
async def health() -> Dict[str, Any]:
    r = get_router()
    stats = r.quota.get_stats()
    available = [p for p, s in stats.items() if s["available"]]
    return {
        "status":             "healthy" if available else "degraded",
        "providers_available": available,
        "providers_total":     len(stats),
    }


# ── Models list ───────────────────────────────────────────────────────────────

@app.get("/v1/models", tags=["Discovery"])
async def list_models() -> Dict[str, Any]:
    r = get_router()
    all_models = r.discovery.get_all_models()
    return {
        "object": "list",
        "data": [
            {
                "id":         m.full_id,
                "object":     "model",
                "owned_by":   m.provider,
                "capabilities": sorted(m.capabilities),
                "context_window": m.context_window,
                "is_free":    m.is_free,
            }
            for m in all_models
        ],
    }


@app.get("/v1/models/{provider}", tags=["Discovery"])
async def list_provider_models(provider: str) -> Dict[str, Any]:
    if provider not in PROVIDER_CATALOGUE:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")
    r = get_router()
    models = r.discovery.get_models(provider)
    return {
        "provider": provider,
        "count":    len(models),
        "models":   [{"id": m.model_id, "capabilities": sorted(m.capabilities), "context_window": m.context_window} for m in models],
    }


# ── Chat completions ──────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", tags=["Completions"])
async def chat_completions(request: ChatRequest) -> Any:
    r = get_router()
    try:
        request_data: Dict[str, Any] = {
            "messages":    [m if isinstance(m, dict) else m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens":  request.max_tokens,
            "top_p":       request.top_p,
            "stream":      request.stream,
            "tools":       request.tools,
            "tool_choice": request.tool_choice,
        }
        if request.model:
            request_data["model"] = request.model

        result = await r.route(request_data, request.routing)
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("chat_completions error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Embeddings ────────────────────────────────────────────────────────────────

@app.post("/v1/embeddings", tags=["Embeddings"])
async def embeddings(request: EmbedRequest) -> Any:
    r = get_router()
    try:
        result = await r.route(
            {"input": request.input, "task_type": "embeddings"},
            request.routing,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("embeddings error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Vision ────────────────────────────────────────────────────────────────────

@app.post("/v1/vision", tags=["Vision"])
async def vision(request: VisionTaskRequest) -> Any:
    r = get_router()
    if not request.image_url and not request.image_base64:
        raise HTTPException(
            status_code=422,
            detail="Provide either image_url or image_base64",
        )
    try:
        request_data: Dict[str, Any] = {
            "task_type":    request.task_type.value,
            "image_url":    request.image_url,
            "image_base64": request.image_base64,
            "question":     request.question,
            "categories":   request.categories,
            "language":     request.language,
        }
        result = await r.route(request_data, request.routing)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("vision error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Provider stats ────────────────────────────────────────────────────────────

@app.get("/providers", tags=["Observability"])
async def provider_stats() -> Dict[str, Any]:
    r = get_router()
    return {
        "providers": r.quota.get_stats(),
        "cache":     r.cache.stats,
    }


@app.get("/providers/{provider}", tags=["Observability"])
async def single_provider_stats(provider: str) -> Dict[str, Any]:
    r = get_router()
    stats = r.quota.get_stats()
    if provider not in stats:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")
    return stats[provider]


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.post("/admin/refresh", tags=["Admin"])
async def force_refresh() -> Dict[str, str]:
    r = get_router()
    await r.discovery.refresh_all(force=True)
    return {"status": "ok", "message": "Model capabilities refreshed"}


@app.post("/admin/cache/clear", tags=["Admin"])
async def clear_cache() -> Dict[str, str]:
    r = get_router()
    r.cache.clear()
    return {"status": "ok", "message": "Response cache cleared"}


@app.post("/admin/quotas/reset", tags=["Admin"])
async def reset_quotas() -> Dict[str, str]:
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
async def full_stats() -> Dict[str, Any]:
    """Combined stats: quotas + cache + model counts."""
    r = get_router()
    return r.get_stats()


# ══════════════════════════════════════════════════════════════════════════════
# Exception handlers
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": type(exc).__name__}},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import uvicorn  # type: ignore[import]
    import argparse
    from llm_router.config import settings  # type: ignore[import]

    parser = argparse.ArgumentParser(description="Start the LLM Router server.")
    parser.add_argument("--host", default=settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    parser.add_argument("--kill", action="store_true", help="Kill existing process on port before starting")
    parser.add_argument("--debug", action="store_true", default=settings.debug, help="Enable reload/debug mode")
    args = parser.parse_args()

    if args.kill:
        # Import here to avoid circular dependency or unnecessary imports if not used
        import subprocess
        import signal
        try:
            output = subprocess.check_output(["lsof", "-ti", f":{args.port}"], text=True).strip()
            if output:
                pids = output.split("\n")
                for pid in pids:
                    print(f"Killing existing process {pid} on port {args.port}...")
                    os.kill(int(pid), signal.SIGTERM)
                time.sleep(1) # Give it a moment to release
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
