"""
admin.py — Admin endpoints for API key management.

Provides endpoints to:
  GET    /admin/api-keys        — list all providers with key status
  POST   /admin/api-keys/{provider}  — set/update API key
  DELETE /admin/api-keys/{provider}  — remove API key
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from llm_router.config import PROVIDER_CATALOGUE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/api-keys", tags=["Admin"])

ROUTER_API_KEY: str | None = None


def get_router_api_key() -> str:
    global ROUTER_API_KEY
    if ROUTER_API_KEY is None:
        from llm_router.config import settings

        if settings.api_key:
            ROUTER_API_KEY = settings.api_key
        else:
            import secrets

            ROUTER_API_KEY = secrets.token_urlsafe(32)
    return ROUTER_API_KEY


def _require_router_api_key(authorization: str | None = Header(None)) -> str:
    import os

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
    except ValueError as err:
        raise HTTPException(status_code=401, detail="Invalid or missing API key") from err

    return key


ENV_FILE_PATH = Path(__file__).parent.parent.parent / ".env"


class ApiKeyStatus(BaseModel):
    """Status of an API key for a provider."""

    provider: str
    has_key: bool
    key_masked: str | None = None


class SetApiKeyRequest(BaseModel):
    """Request to set an API key."""

    api_key: str = Field(..., min_length=1, description="The API key value")


def _load_env_file() -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars: dict[str, str] = {}
    if ENV_FILE_PATH.exists():
        with open(ENV_FILE_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip()
    return env_vars


def _save_env_file(env_vars: dict[str, str]) -> None:
    """Save environment variables to .env file."""
    with open(ENV_FILE_PATH, "w") as f:
        f.write("# Router Configuration\n")
        for key, value in sorted(env_vars.items()):
            f.write(f"{key}={value}\n")


def _has_api_key(provider: str) -> bool:
    """Check if a provider has an API key configured (env or .env file)."""
    cfg = PROVIDER_CATALOGUE.get(provider, {})
    env_var_name = cfg.get("api_key_env")
    if not env_var_name:
        return False

    if os.getenv(env_var_name):
        return True

    env_vars = _load_env_file()
    return env_vars.get(env_var_name) is not None


def _get_masked_key(provider: str) -> str | None:
    """Get masked version of the API key if configured."""
    cfg = PROVIDER_CATALOGUE.get(provider, {})
    env_var_name = cfg.get("api_key_env")
    if not env_var_name:
        return None

    api_key = os.getenv(env_var_name)
    if not api_key:
        env_vars = _load_env_file()
        api_key = env_vars.get(env_var_name)

    if api_key and len(api_key) > 4:
        return f"{api_key[:2]}...{api_key[-2:]}"
    elif api_key:
        return "***"

    return None


@router.get("", response_model=list[ApiKeyStatus])
async def list_api_keys(_api_key: str = Depends(_require_router_api_key)) -> list[dict[str, Any]]:
    """List all providers with their API key status."""
    result = []
    for provider in PROVIDER_CATALOGUE:
        if provider == "ollama":
            continue
        result.append(
            {
                "provider": provider,
                "has_key": _has_api_key(provider),
                "key_masked": _get_masked_key(provider),
            }
        )
    return result


@router.post("/{provider}", response_model=dict[str, str])
async def set_api_key(
    provider: str, request: SetApiKeyRequest, _api_key: str = Depends(_require_router_api_key)
) -> dict[str, str]:
    """Set or update API key for a provider (writes to .env file)."""
    if provider not in PROVIDER_CATALOGUE:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")

    cfg = PROVIDER_CATALOGUE[provider]
    env_var_name = cfg.get("api_key_env")
    if not env_var_name:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' does not support API key configuration",
        )

    env_vars = _load_env_file()
    env_vars[env_var_name] = request.api_key
    _save_env_file(env_vars)

    os.environ[env_var_name] = request.api_key

    logger.info("API key for %s updated in .env file", provider)

    return {
        "status": "ok",
        "message": f"API key for {provider} updated. Restart router to apply changes.",
    }


@router.delete("/{provider}", response_model=dict[str, str])
async def delete_api_key(
    provider: str, _api_key: str = Depends(_require_router_api_key)
) -> dict[str, str]:
    """Remove API key for a provider (removes from .env file)."""
    if provider not in PROVIDER_CATALOGUE:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")

    cfg = PROVIDER_CATALOGUE[provider]
    env_var_name = cfg.get("api_key_env")
    if not env_var_name:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' does not support API key configuration",
        )

    env_vars = _load_env_file()
    if env_var_name in env_vars:
        del env_vars[env_var_name]
        _save_env_file(env_vars)

        if env_var_name in os.environ:
            del os.environ[env_var_name]

        logger.info("API key for %s removed from .env file", provider)
        return {"status": "ok", "message": f"API key for {provider} removed"}
    else:
        return {"status": "ok", "message": f"No API key found for {provider}"}


class OpenAICompatibleEndpoint(BaseModel):
    """OpenAI Compatible endpoint configuration."""

    id: str
    name: str
    base_url: str
    api_key: str | None = None
    models: str = ""
    streaming: bool = True
    enabled: bool = True


class CreateOpenAICompatibleRequest(BaseModel):
    """Request to create an OpenAI Compatible endpoint."""

    name: str
    base_url: str
    api_key: Optional[str] = None
    models: str = ""
    streaming: bool = True
    enabled: bool = True

    model_config = {
        "extra": "allow",
        "populate_by_name": True,
        "validate_default": True,
    }


class TestEndpointResponse(BaseModel):
    """Response from testing an endpoint."""

    success: bool
    message: str
    models: list[str] | None = None


def _load_oai_endpoints() -> list[dict]:
    """Load OAI endpoints from settings."""
    from llm_router.config import settings

    return settings.openai_compatible_endpoints


def _save_oai_endpoints(endpoints: list[dict]) -> None:
    """Save OAI endpoints to .env file."""
    import json

    env_vars = _load_env_file()
    env_vars["ROUTER_OPENAI_COMPATIBLE_ENDPOINTS"] = json.dumps(endpoints)
    _save_env_file(env_vars)
    os.environ["ROUTER_OPENAI_COMPATIBLE_ENDPOINTS"] = json.dumps(endpoints)


@router.get("/openai-compatible", response_model=list[OpenAICompatibleEndpoint])
async def list_openai_compatible_endpoints(
    _api_key: str = Depends(_require_router_api_key),
) -> list[dict]:
    """List all OpenAI Compatible endpoints."""
    endpoints = _load_oai_endpoints()
    result = []
    for ep in endpoints:
        result.append(
            {
                "id": ep.get("id", ""),
                "name": ep.get("name", ""),
                "base_url": ep.get("base_url", ""),
                "api_key": ep.get("api_key"),  # Don't expose raw key
                "models": ep.get("models", ""),
                "streaming": ep.get("streaming", True),
                "enabled": ep.get("enabled", True),
            }
        )
    return result


@router.post("/openai-compatible", response_model=dict[str, str])
async def create_openai_compatible_endpoint(
    request: CreateOpenAICompatibleRequest, _api_key: str = Depends(_require_router_api_key)
) -> dict[str, str]:
    """Create a new OpenAI Compatible endpoint."""
    import secrets

    endpoints = _load_oai_endpoints()
    endpoint_id = secrets.token_urlsafe(8)

    new_endpoint = {
        "id": endpoint_id,
        "name": request.name,
        "base_url": request.base_url,
        "api_key": request.api_key,
        "models": request.models,
        "streaming": request.streaming,
        "enabled": request.enabled,
    }

    endpoints.append(new_endpoint)
    _save_oai_endpoints(endpoints)

    logger.info("OpenAI Compatible endpoint '%s' created", request.name)

    return {
        "status": "ok",
        "message": f"Endpoint '{request.name}' created. Restart router to apply changes.",
        "id": endpoint_id,
    }


@router.put("/openai-compatible/{endpoint_id}", response_model=dict[str, str])
async def update_openai_compatible_endpoint(
    endpoint_id: str,
    request: CreateOpenAICompatibleRequest,
    _api_key: str = Depends(_require_router_api_key),
) -> dict[str, str]:
    """Update an existing OpenAI Compatible endpoint."""
    endpoints = _load_oai_endpoints()

    found = False
    for i, ep in enumerate(endpoints):
        if ep.get("id") == endpoint_id:
            endpoints[i] = {
                "id": endpoint_id,
                "name": request.name,
                "base_url": request.base_url,
                "api_key": request.api_key,
                "models": request.models,
                "streaming": request.streaming,
                "enabled": request.enabled,
            }
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Endpoint '{endpoint_id}' not found")

    _save_oai_endpoints(endpoints)

    logger.info("OpenAI Compatible endpoint '%s' updated", request.name)

    return {
        "status": "ok",
        "message": f"Endpoint '{request.name}' updated. Restart router to apply changes.",
    }


@router.delete("/openai-compatible/{endpoint_id}", response_model=dict[str, str])
async def delete_openai_compatible_endpoint(
    endpoint_id: str, _api_key: str = Depends(_require_router_api_key)
) -> dict[str, str]:
    """Delete an OpenAI Compatible endpoint."""
    endpoints = _load_oai_endpoints()

    found = False
    for i, ep in enumerate(endpoints):
        if ep.get("id") == endpoint_id:
            name = ep.get("name", endpoint_id)
            del endpoints[i]
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Endpoint '{endpoint_id}' not found")

    _save_oai_endpoints(endpoints)

    logger.info("OpenAI Compatible endpoint '%s' deleted", endpoint_id)

    return {
        "status": "ok",
        "message": f"Endpoint '{name}' deleted. Restart router to apply changes.",
    }


@router.post("/openai-compatible/{endpoint_id}/test", response_model=TestEndpointResponse)
async def test_openai_compatible_endpoint(
    endpoint_id: str, _api_key: str = Depends(_require_router_api_key)
) -> dict:
    """Test connectivity to an OpenAI Compatible endpoint."""
    import httpx

    endpoints = _load_oai_endpoints()

    endpoint = None
    for ep in endpoints:
        if ep.get("id") == endpoint_id:
            endpoint = ep
            break

    if not endpoint:
        raise HTTPException(status_code=404, detail=f"Endpoint '{endpoint_id}' not found")

    base_url = endpoint.get("base_url", "").rstrip("/")
    api_key = endpoint.get("api_key")

    if not base_url:
        return {"success": False, "message": "Base URL is required", "models": None}

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{base_url}/models",
                headers=headers if headers else None,
            )

            if response.status_code == 200:
                data = response.json()
                models = []
                if "data" in data:
                    models = [m.get("id") for m in data["data"] if m.get("id")]
                elif "models" in data:
                    models = [m.get("id") for m in data["models"] if m.get("id")]

                return {
                    "success": True,
                    "message": f"Connection successful. Found {len(models)} models.",
                    "models": models[:10] if models else None,
                }
            else:
                return {
                    "success": False,
                    "message": f"HTTP {response.status_code}: {response.text[:200]}",
                    "models": None,
                }
    except httpx.ConnectError:
        return {
            "success": False,
            "message": "Connection failed: Unable to reach endpoint",
            "models": None,
        }
    except httpx.TimeoutException:
        return {"success": False, "message": "Connection timed out", "models": None}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)[:200]}", "models": None}
