from __future__ import annotations

import json
import os
from typing import Any


def build_openrouter_chat_model():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
    timeout = float(os.getenv("OPENROUTER_TIMEOUT", "25"))
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        timeout=timeout,
    )


def maybe_extract_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        return None
