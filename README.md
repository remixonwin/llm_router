# Intelligent LLM Router

<p align="center">
  <a href="https://pypi.org/project/llm_router/">
    <img src="https://img.shields.io/pypi/v/llm_router.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/llm_router/">
    <img src="https://img.shields.io/pypi/pyversions/llm_router.svg" alt="Python versions">
  </a>
  <a href="https://github.com/anomalyco/llm_router/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/llm_router.svg" alt="License">
  </a>
</p>

Intelligent LLM Router is a Python library that routes requests across multiple LLM providers with automatic failover, quota management, and response caching.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google Gemini, Groq, Mistral, Cohere, DeepSeek, Together, HuggingFace, OpenRouter, xAI, DashScope, and Ollama
- **Automatic Failover**: Automatically switches to the next provider if one fails
- **Quota Management**: Tracks RPM/RPD limits per provider and routes around rate limits
- **Response Caching**: Exact and semantic caching to reduce costs and latency
- **Multiple Routing Strategies**: Auto, Cost-Optimized, Quality-First, Latency-First, Round-Robin
- **Vision & Embeddings**: Full support for vision models and embeddings

## Installation

```bash
pip install llm_router
```

Or with specific extras:

```bash
pip install llm_router[server]  # Includes FastAPI server
```

## Quick Start

```python
import asyncio
from llm_router import IntelligentRouter, RoutingOptions, RoutingStrategy

async def main():
    # Initialize the router
    router = IntelligentRouter()
    await router.start()
    
    # Define your request
    request_data = {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
    }
    
    # Configure routing options (optional)
    options = RoutingOptions(
        strategy=RoutingStrategy.AUTO,
    )
    
    # Route the request
    response = await router.route(request_data, options)
    
    print(response["choices"][0]["message"]["content"])
    print(f"Provider: {response['routing_metadata']['provider']}")
    
    await router.stop()

asyncio.run(main())
```

## Configuration

### Environment Variables

All configuration can be done via environment variables with the `ROUTER_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_LLM_TIMEOUT` | 60 | Timeout for LLM calls in seconds |
| `ROUTER_MAX_RETRIES` | 3 | Maximum retry attempts |
| `ROUTER_ENABLE_OLLAMA_FALLBACK` | true | Enable Ollama fallback when cloud providers fail |
| `ROUTER_CACHE_DIR` | /tmp/llm_router_cache | Directory for response cache |
| `ROUTER_RESPONSE_CACHE_TTL` | 3600 | Cache TTL in seconds |
| `ROUTER_DEFAULT_STRATEGY` | auto | Default routing strategy |

### Provider API Keys

Set API keys as environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk_...
export GEMINI_API_KEY=AIza...
# etc.
```

## Routing Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Balanced: quota + latency + quality (default) |
| `cost_optimized` | Maximize remaining free quota |
| `quality_first` | Prioritize highest quality models |
| `latency_first` | Prioritize fastest responding models |
| `round_robins` | Uniform spread across providers |

## Request Options

```python
from llm_router import RoutingOptions, CachePolicy

options = RoutingOptions(
    strategy=RoutingStrategy.COST_OPTIMIZED,
    free_tier_only=True,  # Only use free tier providers
    preferred_providers=["groq", "gemini"],  # Prefer these providers
    excluded_providers=["openai"],  # Skip these providers
    cache_policy=CachePolicy.ENABLED,  # Enable response caching
)
```

## API Reference

### IntelligentRouter

```python
router = IntelligentRouter()
await router.start()  # Initialize router
response = await router.route(request_data, options)  # Route request
stats = router.get_stats()  # Get router statistics
await router.stop()  # Cleanup
```

### Models

- `RoutingOptions`: Configuration for routing behavior
- `RoutingStrategy`: Enum for routing strategies
- `TaskType`: Enum for task types (chat, embeddings, vision, etc.)
- `CachePolicy`: Enum for caching behavior
- `settings`: Global settings object

## Running the Server

```bash
# Install with server extras
pip install llm_router[server]

# Run the server
llm-router

# Or with custom settings
ROUTER_PORT=8001 llm-router
```

The server provides a FastAPI-compatible API at `/v1/chat/completions`, `/v1/embeddings`, and `/health`.

## Development

```bash
# Clone the repository
git clone https://github.com/anomalyco/llm_router.git

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.
