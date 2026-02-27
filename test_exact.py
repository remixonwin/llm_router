#!/usr/bin/env python3
"""Test with the exact model from llm_router."""

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()


# Exact copy of the model from admin.py
class CreateOpenAICompatibleRequest(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    models: str = ""
    streaming: bool = True
    enabled: bool = True


def _require_router_api_key():
    return "test"


@app.post("/test")
async def test_endpoint(
    request: CreateOpenAICompatibleRequest, _api_key: str = Depends(_require_router_api_key)
):
    return {
        "status": "ok",
        "name": request.name,
        "base_url": request.base_url,
        "api_key": request.api_key,
        "models": request.models,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7997)
