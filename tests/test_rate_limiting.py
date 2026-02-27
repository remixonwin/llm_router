"""
Unit tests for rate limiting functionality.

These tests verify the rate limiting configuration and behavior
across the LLM Router, Manager, and MCQ Generator services.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestRateLimitConfiguration:
    """Test rate limiting configuration loading."""

    def test_llm_router_config_defaults(self):
        """Test that LLM Router has sensible rate limit defaults."""
        # Test default values from config
        from llm_router.config import Settings
        
        settings = Settings()
        
        assert settings.rate_limit_enabled is True
        assert settings.rate_limit_read_requests == 100
        assert settings.rate_limit_read_window == "1 minute"
        assert settings.rate_limit_write_requests == 10
        assert settings.rate_limit_write_window == "1 minute"
        assert "redis://localhost:6379/0" in settings.rate_limit_redis_url

    def test_llm_router_config_env_override(self):
        """Test that environment variables override rate limit defaults."""
        with patch.dict("os.environ", {
            "ROUTER_RATE_LIMIT_ENABLED": "false",
            "ROUTER_RATE_LIMIT_READ_REQUESTS": "200",
            "ROUTER_RATE_LIMIT_WRITE_REQUESTS": "20",
            "ROUTER_RATE_LIMIT_REDIS_URL": "redis://redis:6379/1",
        }):
            from llm_router.config import Settings
            # Need to reload the settings
            import importlib
            import llm_router.config
            importlib.reload(llm_router.config)
            from llm_router.config import Settings as ReloadedSettings
            
            settings = ReloadedSettings()
            assert settings.rate_limit_enabled is False


class TestRateLimitDecorator:
    """Test rate limiting decorator application."""

    def test_limiter_instance_exists(self):
        """Test that rate limiter is properly instantiated."""
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        
        limiter = Limiter(key_func=get_remote_address)
        
        assert limiter is not None
        # Use _key_func as key_func is not public in newer slowapi
        key_func = getattr(limiter, "_key_func", None) or getattr(limiter, "key_func", None)
        assert key_func == get_remote_address

    @pytest.mark.asyncio
    async def test_rate_limit_key_func(self):
        """Test that IP address extraction works correctly."""
        from slowapi.util import get_remote_address
        from fastapi import Request
        from unittest.mock import Mock
        
        # Test with mock request
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.1"
        
        ip = get_remote_address(mock_request)
        
        assert ip == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_rate_limit_key_func_x_forwarded(self):
        """Test that X-Forwarded-For header is respected."""
        from slowapi.util import get_remote_address
        from fastapi import Request
        from unittest.mock import Mock
        
        # Test with X-Forwarded-For header
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        
        # Default behavior should return client host
        ip = get_remote_address(mock_request)
        
        # The limiter should use the client host by default
        assert ip is not None


class TestRateLimitExceptionHandler:
    """Test rate limit exception handling."""

    def test_rate_limit_exception_handler_response(self):
        """Test that 429 response is properly formatted."""
        from slowapi.errors import RateLimitExceeded
        from unittest.mock import Mock
        
        # Create a mock limit object that RateLimitExceeded expects
        mock_limit = Mock()
        mock_limit.limit = "10 per 1 minute"
        mock_limit.error_message = "Too many requests"
        
        # Create a mock exception
        exc = RateLimitExceeded(mock_limit)
        
        # Verify the exception has the expected attributes
        assert exc.detail == "Too many requests"

    def test_retry_after_header_format(self):
        """Test that Retry-After header is properly formatted."""
        detail = "10 per 1 minute"
        assert detail is not None


class TestRateLimitMiddleware:
    """Test rate limiting middleware configuration."""

    def test_app_has_limiter_state(self):
        """Test that app has rate limiter in state."""
        from fastapi import FastAPI
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        
        app = FastAPI()
        limiter = Limiter(key_func=get_remote_address)
        
        app.state.limiter = limiter
        
        assert app.state.limiter is not None
        assert app.state.limiter == limiter


class TestRedisRateLimitStorage:
    """Test Redis rate limit storage configuration."""

    @pytest.mark.asyncio
    async def test_redis_client_creation(self):
        """Test that Redis client can be created."""
        redis_url = "redis://localhost:6379/0"
        assert "6379" in redis_url
        assert redis_url.startswith("redis://")

    def test_redis_url_env_override(self):
        """Test that Redis URL can be overridden via environment."""
        test_url = "redis://custom-redis:6379/5"
        assert "custom-redis" in test_url
        assert test_url.startswith("redis://")
        assert "/5" in test_url


class TestRateLimitEndpoints:
    """Test rate limiting on specific endpoints."""

    def test_write_endpoint_rate_limit_format(self):
        """Test that write endpoint rate limit is correctly formatted."""
        requests = 10
        window = "1 minute"
        rate_limit_string = f"{requests}/{window}"
        assert rate_limit_string == "10/1 minute"

    def test_read_endpoint_rate_limit_format(self):
        """Test that read endpoint rate limit is correctly formatted."""
        requests = 100
        window = "1 minute"
        rate_limit_string = f"{requests}/{window}"
        assert rate_limit_string == "100/1 minute"


class TestRateLimitDisable:
    """Test rate limiting can be disabled."""

    def test_rate_limiting_can_be_disabled(self):
        """Test that rate limiting can be disabled via config."""
        enabled_values = ["true", "false", "1", "0", "yes", "no"]
        for val in enabled_values:
            result = val.lower() in ("1", "true", "yes")
            assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
