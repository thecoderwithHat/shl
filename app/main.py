from __future__ import annotations

from fastapi import FastAPI

from .recommender import ChatRequest, ChatResponse, build_chat_response


app = FastAPI(title="SHL Conversational Recommender", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return build_chat_response(request.messages)

