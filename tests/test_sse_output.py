import json

from fastapi.testclient import TestClient

from llm_router import server


def test_sse_stream_shape():
    """Ensure SSE stream lines start with `data:` and decode to JSON or include an error."""
    client = TestClient(server.app)

    class FakeRouter:
        async def route(self, request_data, routing):
            async def _aiter():
                yield {"choices": [{"message": {"content": "chunk1"}}]}
                yield {"choices": [{"message": {"content": "chunk2"}}]}

            return _aiter()

    # Inject a minimal router instance so endpoints can run without full startup
    server._router = FakeRouter()

    payload = {"messages": [{"role": "user", "content": "Hello"}], "stream": True}

    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        # iterate over event-stream lines
        for raw in resp.iter_lines():
            if not raw:
                continue
            # httpx returns bytes; decode to str
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            assert raw.startswith("data: ")
            data_part = raw[len("data: ") :]
            # Either JSON-decode to a dict/list or contain an 'error' field/expected text
            try:
                obj = json.loads(data_part)
            except Exception:
                assert "chunk" in data_part or "error" in data_part
            else:
                assert isinstance(obj, (dict, list))
