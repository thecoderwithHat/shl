# SHL Conversational Recommender

FastAPI service that turns the supplied SHL catalog and sample conversations into a conversational assessment recommender.

## What it does

The service accepts the full conversation history on every request, applies deterministic catalog logic first, and optionally uses an OpenRouter-backed LLM path when the LangChain extras are installed and configured. It returns strict JSON with `reply`, `recommendations`, and `end_of_conversation`.

## Setup

```bash
python -m pip install -e .[dev]
```

Optional LLM extras:

```bash
python -m pip install -e .[llm]
```

## Environment variables

- `OPENROUTER_API_KEY`: enables the OpenRouter LLM path when the optional LangChain packages are installed.
- `OPENROUTER_MODEL`: defaults to `meta-llama/llama-3-8b-instruct`.
- `ENABLE_LLM`: set to `1` to allow the optional LLM extraction path.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

### `GET /health`

Readiness check. The service is stateless, so this is safe to call on every deployment probe.

### `POST /chat`

Request body:

```json
{
  "messages": [
    {"role": "user", "content": "We need a solution for senior leadership."}
  ]
}
```

Response body:

```json
{
  "reply": "...",
  "recommendations": [],
  "end_of_conversation": false
}
```

## Render deployment

Use `uvicorn app.main:app` as the start command. The health endpoint is designed for cold starts, and the service keeps request handling stateless so Render can restart it without losing conversation state.

## Scope boundaries

This recommender stays inside the catalog facts and the conversation context. It can help compare products, build shortlists, and refuse legal/compliance interpretations, but it does not provide legal advice, infer uncatalogued SHL products, or replace human review when the role requirements are still ambiguous.
