import requests
import json
import time

BASE_URL = "http://localhost:7544"

def test_health():
    print("Testing /health...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(resp.json())
    assert resp.status_code == 200

def test_models():
    print("\nTesting /v1/models...")
    resp = requests.get(f"{BASE_URL}/v1/models")
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Total models: {len(data.get('data', []))}")
    assert resp.status_code == 200

def test_chat():
    print("\nTesting /v1/chat/completions...")
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "routing": {"strategy": "cost_optimized", "free_tier_only": True}
    }
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(resp.json()["choices"][0]["message"]["content"][:100] + "...")
    else:
        print(resp.text)
    assert resp.status_code == 200

def test_embeddings():
    print("\nTesting /v1/embeddings...")
    payload = {
        "input": "The quick brown fox jumps over the lazy dog",
        "routing": {"free_tier_only": True}
    }
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Embedding dimensions: {len(resp.json()['data'][0]['embedding'])}")
    else:
        print(resp.text)
    assert resp.status_code == 200

def test_providers():
    print("\nTesting /providers...")
    resp = requests.get(f"{BASE_URL}/providers")
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200

def test_stats():
    print("\nTesting /stats...")
    resp = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200

if __name__ == "__main__":
    try:
        test_health()
        test_models()
        test_chat()
        test_embeddings()
        test_providers()
        test_stats()
        print("\nAll integration tests PASSED!")
    except Exception as e:
        print(f"\nIntegration test FAILED: {e}")
