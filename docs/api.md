# API Reference

## IntelligentRouter

The main router class for handling LLM requests.

```python
from llm_router import IntelligentRouter
```

### Constructor

```python
router = IntelligentRouter()
```

### Methods

#### `async start() -> None`

Initializes the router. Must be called before routing any requests.

```python
await router.start()
```

#### `async stop() -> None`

Stops the router and cleans up resources.

```python
await router.stop()
```

#### `async route(request_data: dict, routing_options: RoutingOptions | None = None) -> dict`

Routes a request to the optimal LLM provider.

**Parameters:**
- `request_data` (dict): The request data containing messages, input, or image data
- `routing_options` (RoutingOptions, optional): Configuration for routing behavior

**Returns:**
- dict: The response from the LLM provider with routing metadata

**Example:**

```python
response = await router.route({
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
}, RoutingOptions(strategy=RoutingStrategy.AUTO))
```

#### `get_stats() -> dict`

Gets router statistics including provider stats, cache stats, and models per provider.

```python
stats = router.get_stats()
```

---

## RoutingOptions

Configuration for routing behavior.

```python
from llm_router import RoutingOptions
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | RoutingStrategy | AUTO | The routing strategy to use |
| `free_tier_only` | bool | False | Only use free tier providers |
| `preferred_providers` | list[str] | [] | Preferred provider list |
| `excluded_providers` | list[str] | [] | Excluded provider list |
| `cache_policy` | CachePolicy | ENABLED | Caching behavior |
| `require_capability` | str \| None | None | Required capability |

---

## RoutingStrategy

Enum for routing strategies.

```python
from llm_router import RoutingStrategy
```

### Values

| Value | Description |
|-------|-------------|
| `AUTO` | Balanced: quota + latency + quality |
| `COST_OPTIMIZED` | Maximize remaining free quota |
| `QUALITY_FIRST` | Prioritize highest quality models |
| `LATENCY_FIRST` | Prioritize fastest responding models |
| `ROUND_ROBIN` | Uniform spread across providers |

---

## TaskType

Enum for task types.

```python
from llm_router import TaskType
```

### Values

| Value | Description |
|-------|-------------|
| `TEXT_GENERATION` | Text generation |
| `CHAT_COMPLETION` | Chat completion |
| `EMBEDDINGS` | Text embeddings |
| `VISION_CLASSIFY` | Image classification |
| `VISION_DETECT` | Object detection |
| `VISION_OCR` | Optical character recognition |
| `VISION_QA` | Visual question answering |
| `VISION_CAPTION` | Image captioning |
| `VISION_UNDERSTANDING` | Vision understanding |
| `FUNCTION_CALLING` | Function calling |
| `UNKNOWN` | Unknown task |

---

## CachePolicy

Enum for caching behavior.

```python
from llm_router import CachePolicy
```

### Values

| Value | Description |
|-------|-------------|
| `ENABLED` | Use cache if available |
| `DISABLED` | Bypass cache entirely |
| `REFRESH` | Force re-fetch and repopulate |

---

## Settings

Global settings object.

```python
from llm_router import settings
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | "0.0.0.0" | Server host |
| `port` | int | 8001 | Server port |
| `log_level` | str | "INFO" | Log level |
| `debug` | bool | False | Debug mode |
| `llm_timeout` | int | 60 | LLM call timeout (seconds) |
| `discovery_timeout` | int | 10 | Discovery timeout (seconds) |
| `max_retries` | int | 3 | Maximum retry attempts |
| `retry_base_delay` | float | 1.0 | Base delay for retries (seconds) |
| `enable_ollama_fallback` | bool | True | Enable Ollama fallback |
| `cache_dir` | str | "/tmp/llm_router_cache" | Cache directory |
| `response_cache_ttl` | int | 3600 | Response cache TTL (seconds) |
| `default_strategy` | str | "auto" | Default routing strategy |

---

## PROVIDER_CATALOGUE

Dictionary containing all supported providers and their configurations.

```python
from llm_router import PROVIDER_CATALOGUE

# List all providers
print(list(PROVIDER_CATALOGUE.keys()))
```

### Supported Providers

- `openai` - OpenAI GPT models
- `anthropic` - Anthropic Claude models
- `gemini` - Google Gemini models
- `groq` - Groq models
- `mistral` - Mistral AI models
- `cohere` - Cohere models
- `deepseek` - DeepSeek models
- `together` - Together AI models
- `huggingface` - HuggingFace models
- `openrouter` - OpenRouter models
- `xai` - xAI Grok models
- `dashscope` - Alibaba DashScope models
- `ollama` - Local Ollama models

---

## Request Data Format

### Chat Completion

```python
{
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "stream": False,
}
```

### Embeddings

```python
{
    "input": "Text to embed",
    # or
    "input": ["text1", "text2"]
}
```

### Vision

```python
{
    "image_url": "https://example.com/image.jpg",
    "question": "What is in this image?"
}
# or
{
    "image_base64": "data:image/jpeg;base64,...",
    "task_type": "vision_understand"
}
```

---

## Response Format

All responses include `routing_metadata`:

```python
{
    "choices": [...],
    "usage": {...},
    "routing_metadata": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "strategy": "auto",
        "latency_ms": 123.4,
        "cache_hit": False,
        "cost_usd": 0.0,
        "fallback": False,
        "original_provider": "groq"
    }
}
```
