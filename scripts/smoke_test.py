"""Smoke test the running FastAPI app: /health and /chat endpoints.

Usage: python scripts/smoke_test.py
"""
from __future__ import annotations
import json
import sys

try:
    import httpx
except Exception:
    print("httpx is required. Install with: pip install httpx")
    sys.exit(2)

BASE = "http://127.0.0.1:8000"

def health():
    try:
        r = httpx.get(f"{BASE}/health", timeout=10.0)
        print("HEALTH", r.status_code, r.text)
    except Exception as e:
        print("HEALTH ERROR", e)


def chat():
    payload = {
        "messages": [
            {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus."}
        ]
    }
    try:
        r = httpx.post(f"{BASE}/chat", json=payload, timeout=20.0)
        print("CHAT", r.status_code)
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)
    except Exception as e:
        print("CHAT ERROR", e)


if __name__ == '__main__':
    health()
    chat()
