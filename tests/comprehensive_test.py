
import pytest
import requests

# This file is a convenience script for manual end-to-end checks. It is not a
# unit test and expects a live router at localhost. When running the test suite
# under pytest we skip this module to avoid fixture injection errors.
if __name__ != "__main__":
    pytest.skip(
        "comprehensive_test is a standalone script — skipped during pytest runs",
        allow_module_level=True,
    )

BASE_URL = "http://localhost:8001"

# Red dot 5x5 PNG
RED_DOT_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Red_dot.svg/10px-Red_dot.svg.png"


def test_endpoint(name, path, payload, expected_status=200):
    print(f"\n--- Testing {name} ({path}) ---")
    resp = requests.post(f"{BASE_URL}{path}", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Status: {resp.status_code}")
        meta = data.get("routing_metadata", {})
        print(
            f"Routed via: {meta.get('provider')} / {meta.get('model')} (Strategy: {meta.get('strategy')})"
        )
        if (
            "data" in data
            and isinstance(data["data"], list)
            and "embedding" in data["data"][0]
        ):
            print(f"Embedding success (dim: {len(data['data'][0]['embedding'])})")
        elif "choices" in data:
            print(f"Response: {data['choices'][0]['message']['content'][:200]}...")
        else:
            print(f"Response: {str(data)[:200]}...")
    else:
        print(f"Status: {resp.status_code}")
        print(f"Error: {resp.text}")

    assert resp.status_code == expected_status, (
        f"Expected {expected_status}, got {resp.status_code}: {resp.text}"
    )


if __name__ == "__main__":
    # 1. Text-only Chat
    test_endpoint(
        "Text Chat",
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "routing": {"strategy": "cost_optimized"},
        },
    )

    # 2. Vision (Base64 URL) via /v1/vision
    test_endpoint(
        "Vision Base64 URL",
        "/v1/vision",
        {
            "image_url": f"data:image/png;base64,{RED_DOT_BASE64}",
            "question": "What color is this dot?",
            "routing": {"strategy": "auto"},
        },
    )

    # 3. Vision (Base64 Direct) via /v1/vision
    test_endpoint(
        "Vision Base64 Direct",
        "/v1/vision",
        {
            "image_base64": RED_DOT_BASE64,
            "question": "What is in this image?",
            "routing": {"strategy": "auto"},
        },
    )

    # 4. Mixed Content Chat (Text + Image URL)
    test_endpoint(
        "Mixed Chat (URL)",
        "/v1/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{RED_DOT_BASE64}"
                            },
                        },
                    ],
                }
            ],
            "routing": {"strategy": "quality_first"},
        },
    )

    # 5. Mixed Content Chat (Text + Base64)
    test_endpoint(
        "Mixed Chat (Base64)",
        "/v1/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{RED_DOT_BASE64}"
                            },
                        },
                    ],
                }
            ],
            "routing": {"strategy": "quality_first"},
        },
    )

    # 6. Embedding Request
    test_endpoint(
        "Embeddings",
        "/v1/embeddings",
        {
            "input": "This is a test sentence for embeddings.",
            "routing": {"strategy": "cost_optimized"},
        },
    )

    print("\nAll comprehensive tests PASSED!")
