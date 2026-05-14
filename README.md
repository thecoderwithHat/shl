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
- `FAISS_INDEX_PATH`: path where the FAISS vector index is persisted (defaults to `./faiss_index`).
- `FAISS_ALLOW_DANGEROUS_DESERIALIZATION`: opt-in flag to allow langchain to unpickle a persisted FAISS index. Set to `1` only if you trust the index files created locally. Default: `1` in `.example.env` for local development.
- `MAX_RECOMMENDATIONS`: maximum number of recommendations returned in API responses (default `10`).
- `OPENROUTER_TIMEOUT`: per-call timeout (seconds) for the optional OpenRouter LLM path (default `25`).

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For local development you can set env vars in a `.env` file or export them inline. Example (PowerShell):

```powershell
$env:FAISS_ALLOW_DANGEROUS_DESERIALIZATION=1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If you prefer not to allow pickle deserialization, rebuild the index locally and start the server:

```bash
python scripts/build_faiss_index.py --force
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
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

Notes on response schema and behavior
- `recommendations` is always an array (never `null`). When non-empty each item is a compact object with exactly these keys: `name`, `url`, `test_type`.
- The recommender uses the full conversation message count to gate clarifications and finalization. A hard cap of 8 total turns (user+assistant) forces finalization when reached.
- Shortlists are deterministically truncated to `MAX_RECOMMENDATIONS` before being returned.

## Render deployment

Use `uvicorn app.main:app` as the start command. The health endpoint is designed for cold starts, and the service keeps request handling stateless so Render can restart it without losing conversation state.

## Scope boundaries

This recommender stays inside the catalog facts and the conversation context. It can help compare products, build shortlists, and refuse legal/compliance interpretations, but it does not provide legal advice, infer uncatalogued SHL products, or replace human review when the role requirements are still ambiguous.
