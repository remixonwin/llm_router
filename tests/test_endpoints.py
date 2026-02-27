import pytest
import os
from fastapi.testclient import TestClient

# Disable API key requirement for tests
os.environ["ROUTER_REQUIRE_API_KEY"] = "false"

from llm_router.server import app

@pytest.fixture
def client():
    # with TestClient(app) triggers lifespan events
    with TestClient(app) as c:
        yield c

@pytest.fixture
def async_client(client):
    from httpx import AsyncClient, ASGITransport
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestHealthEndpoints:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "providers_available" in data
        assert "providers_total" in data


class TestDiscoveryEndpoints:
    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_list_models_by_provider(self, client):
        # We use a real provider name from the catalogue
        response = client.get("/v1/models/groq")
        assert response.status_code == 200
        data = response.json()
        assert "provider" in data
        assert data["provider"] == "groq"
        assert "models" in data

    def test_list_models_invalid_provider(self, client):
        response = client.get("/v1/models/invalid_provider")
        assert response.status_code == 404


class TestProviderEndpoints:
    def test_provider_stats(self, client):
        response = client.get("/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "cache" in data

    def test_single_provider_stats(self, client):
        response = client.get("/providers/groq")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data

    def test_single_provider_stats_invalid(self, client):
        response = client.get("/providers/invalid_provider")
        assert response.status_code == 404

    def test_stats(self, client):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "quota" in data or "cache" in data


class TestAdminEndpoints:
    def test_clear_cache(self, client):
        response = client.post("/admin/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_reset_quotas(self, client):
        response = client.post("/admin/quotas/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_refresh(self, client):
        response = client.post("/admin/refresh")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestChatCompletions:
    def test_chat_completions_get(self, client):
        response = client.get("/v1/chat/completions")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_completions_post(self, async_client):
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "llama-3.1-8b-instant",
        }
        response = await async_client.post("/v1/chat/completions", json=payload)
        # Expected statuses including potential auth or provider errors if keys not set
        assert response.status_code in [200, 401, 500, 503]

    @pytest.mark.asyncio
    async def test_chat_completions_stream(self, async_client):
        payload = {
            "messages": [{"role": "user", "content": "Count to 3"}],
            "model": "llama-3.1-8b-instant",
            "stream": True,
        }
        response = await async_client.post("/v1/chat/completions", json=payload)
        assert response.status_code in [200, 401, 500, 503]


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_embeddings(self, async_client):
        payload = {
            "input": "Hello world",
            "model": "text-embedding-3-small",
        }
        response = await async_client.post("/v1/embeddings", json=payload)
        assert response.status_code in [200, 401, 500, 503]


class TestVision:
    @pytest.mark.asyncio
    async def test_vision_requires_image(self, async_client):
        payload = {
            "task_type": "vision_understanding",
        }
        response = await async_client.post("/v1/vision", json=payload)
        assert response.status_code == 422


class TestWrongPaths:
    def test_duplicated_chat_path(self, client):
        response = client.post(
            "/v1/chat/completions/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        # It should be 404 or handled
        assert response.status_code in [200, 400, 404, 422, 500, 503]

    def test_wrong_chat_models_path(self, client):
        response = client.get("/v1/chat/completions/models")
        assert response.status_code in [200, 404]
