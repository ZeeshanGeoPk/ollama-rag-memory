# Ollama Context-Pruning Middleware

Local FastAPI middleware that sits between Ollama-compatible clients and two local Ollama servers:

- LLM server: `http://localhost:8000`
- Embedding server: `http://localhost:8001`
- Middleware API: `http://localhost:8080`

It stores chat history locally, retrieves relevant history with ChromaDB, prunes noisy context, and forwards a smaller request to the LLM server.

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
http://127.0.0.1:8080
```

The interface includes saved conversations, streaming replies, a direct
Ollama/middle-layer comparison toggle, the forwarded context window, and live
NVIDIA GPU telemetry when `nvidia-smi` is available.

On startup it will:

1. Start two `ollama serve` processes on ports `8000` and `8001` if they are not already reachable.
2. Use `.data/ollama_models` as the local Ollama model folder.
3. Pull `phi4-mini:3.8b` and `nomic-embed-text:v1.5` only when missing.
4. Serve the proxy API on `http://127.0.0.1:8080`.

## Endpoints

- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `POST /api/embeddings`
- `GET /api/tags`
- `GET /health`
- `GET /debug/context-preview`
- `POST /admin/reset`

Point an Ollama-compatible client at `http://localhost:8080`.
