# Ollama Context-Pruning Middleware

Local FastAPI middleware that sits between Ollama-compatible clients and two local Ollama servers:

- LLM server: `http://localhost:8001`
- Embedding server: `http://localhost:8002`
- Middleware API: `http://localhost:8000`

It stores the complete chat history locally, retrieves relevant history with
ChromaDB and `nomic-embed-text`, expands matches to their neighboring chunks
and user/assistant exchange, and forwards only that focused context to the LLM.

History is never stored as one giant vector. Each turn is split into ordered,
overlapping chunks with conversation, turn, role, and chunk-index metadata.
The latest turns are kept directly while older details are recalled through
RAG. `RETRIEVAL_CHUNK_NEIGHBORS` controls how much text around each vector hit
is restored, and `RETRIEVAL_INCLUDE_TURN_PAIR` keeps the matching question and
answer together.

## Setup

Install Ollama first, then install Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
```

Run the middleware:

```bash
.venv/bin/python -m ollama_middle_layer
```

Open the chat interface:

```text
http://127.0.0.1:8000
```

The interface includes saved conversations, streaming replies, a direct
Ollama/middle-layer comparison toggle, the forwarded context window, and live
NVIDIA GPU telemetry when `nvidia-smi` is available.

On startup it will:

1. Start two `ollama serve` processes on ports `8001` and `8002` if they are not already reachable.
2. Use `.data/ollama_models` as the local Ollama model folder.
3. Pull `phi4-mini:3.8b` and `nomic-embed-text:v1.5` only when missing.
4. Serve the proxy API on `http://127.0.0.1:8000`.

## Endpoints

- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `POST /api/embeddings`
- `GET /api/tags`
- `GET /health`
- `GET /debug/context-preview`
- `POST /admin/reset`

Point an Ollama-compatible client at `http://localhost:8000`.
