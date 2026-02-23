#!/bin/bash
# dev.sh — Robust startup script for LLM Router

# Default port
PORT=${ROUTER_PORT:-8001}

echo "--- Starting LLM Router on port $PORT ---"

# Use the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Kill existing process and start the router
uv run llm-router --kill --port "$PORT" "$@"
