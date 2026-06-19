# Ollama Context-Pruning Middleware

A local semantic context-virtualization proxy for Ollama.

This project keeps complete conversation history on local storage, recalls the
parts relevant to the current request, reconstructs coherent conversation
exchanges around those matches, and forwards a much smaller context to a local
LLM.

It is designed for laptops and edge devices where sending a 32K or 128K
transcript on every request can make local inference slow or exhaust available
memory.

## Why this project exists

Long context windows are useful, but their full cost is paid on every prompt:

- more prompt tokens to evaluate;
- higher time-to-first-token;
- greater RAM and VRAM pressure;
- repeated processing of history unrelated to the current request.

This middleware treats the model context window as working memory and local
storage as long-term memory. The complete history remains available, but only
recent and relevant information is placed back into the model's active context.

## Distinctive highlights

The individual ideas behind RAG and external memory are established. What makes
this project distinctive is their packaging for local Ollama:

- **Drop-in Ollama-compatible proxy** — point compatible clients at the
  middleware instead of changing application-level prompting code.
- **Complete local history** — SQLite remains the source of truth; semantic
  retrieval is an index over history, not a replacement for it.
- **Coherent memory reconstruction** — a vector hit expands to neighboring
  chunks and, optionally, its user/assistant turn pair. The model receives an
  understandable exchange rather than disconnected matching sentences.
- **Recent + relevant context** — recent conversation continuity and older
  semantic memories have separate token budgets.
- **Follow-up-aware retrieval** — requests such as “implement it” are combined
  with the previous user request before embedding.
- **Global-history detection** — requests asking for a recap of the whole chat
  bypass top-k retrieval so unrelated portions are not silently omitted.
- **Fully local stack** — Ollama, `nomic-embed-text`, ChromaDB, and SQLite run
  without a hosted model or external memory service.
- **Observable optimization** — the included UI shows forwarded context,
  original versus forwarded token estimates, retrieved chunks, reduction
  percentage, and optional NVIDIA GPU telemetry.
- **Direct comparison mode** — switch between the middle layer and unpruned
  Ollama history for the same conversation.
- **Graceful fallback** — if vector retrieval fails, the proxy forwards a
  bounded window of recent messages instead of failing the chat request.

## How it works

```text
Ollama-compatible client / included web UI
                    |
                    v
        FastAPI middleware :8000
                    |
        +-----------+------------+
        |                        |
        v                        v
 SQLite complete history    nomic-embed-text
        |                    Ollama :8002
        |                        |
        +----------> ChromaDB <--+
                         |
              relevant chunk matches
                         |
            neighbor + turn-pair expansion
                         |
             recent/retrieved token budgets
                         |
                         v
                 Chat Ollama :8001
```

For each middleware request:

1. Complete turns are stored in SQLite.
2. Long turns are split into ordered, overlapping chunks.
3. Chunks are embedded with `search_document:` using
   `nomic-embed-text:v1.5`.
4. The current request is embedded with `search_query:`.
5. ChromaDB retrieves matches only from the active conversation.
6. Matching chunks are expanded to nearby chunks and their conversation pair.
7. Recent turns are sentence-ranked while the latest user-led exchange is
   protected.
8. Both sections are packed into configurable token budgets and forwarded as
   model context.

History is never embedded as one giant vector. Chunk order, turn IDs, roles,
and conversation IDs are retained as metadata so matching text can be restored
in its original conversational setting.

## Technology

- Python 3.12+
- FastAPI and Uvicorn
- Ollama
- `phi4-mini:3.8b` by default
- `nomic-embed-text:v1.5`
- ChromaDB with cosine similarity
- SQLite
- Plain HTML, CSS, and JavaScript frontend

## Quick start

Install [Ollama](https://ollama.com/) first.

```bash
git clone YOUR_REPOSITORY_URL
cd ollama_middle_layer

python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env

.venv/bin/python -m ollama_middle_layer
```

Then open:

```text
http://127.0.0.1:8000
```

With `OLLAMA_BOOTSTRAP=true`, startup will:

1. reuse Ollama servers already listening at the configured addresses;
2. otherwise start chat and embedding Ollama processes;
3. share the configured local model directory between both processes;
4. pull the configured models only when they are missing;
5. rebuild an empty Chroma index from SQLite history when necessary.

You can also use the installed command:

```bash
.venv/bin/ollama-middle-layer
```

## Suggested demo

The included UI makes the retrieval behavior visible without additional tools:

1. Start a middleware conversation and establish a specific fact:

   ```text
   We chose PostgreSQL because the project needs transactional migrations.
   ```

2. Continue the conversation with several unrelated requests so that the
   decision moves outside the recent-turn window.

3. Ask:

   ```text
   Which database did we choose, and why?
   ```

4. Open the **Context window** panel and compare:

   - the retrieved historical chunk;
   - the reconstructed user/assistant exchange;
   - original versus forwarded token estimates;
   - the reduction percentage.

5. Switch to **Direct Ollama** to show the same request with the complete
   transcript forwarded instead of retrieved memory.

For an open-source showcase, a short screen recording of this flow communicates
the project more clearly than a synthetic performance claim.

## Using it as an Ollama-compatible API

Point an Ollama-compatible client to:

```text
http://127.0.0.1:8000
```

Provide a stable `conversation_id` so retrieval remains isolated to the correct
chat:

```bash
curl http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi4-mini:3.8b",
    "conversation_id": "demo-project",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "What database did we choose earlier and why?"
      }
    ]
  }'
```

For clients that only preserve custom values inside Ollama options, this is
also supported:

```json
{
  "options": {
    "conversation_id": "demo-project"
  }
}
```

API clients should resend their normal transcript on subsequent chat requests.
The middleware synchronizes the unseen suffix and avoids re-embedding the
matching stored prefix. The included web UI persists both user and completed
assistant turns directly.

## Inspecting forwarded context

Preview what the middleware would send without running a generation:

```bash
curl --get http://127.0.0.1:8000/debug/context-preview \
  --data-urlencode "conversation_id=demo-project" \
  --data-urlencode "q=What database did we choose?"
```

The response includes:

- raw retrieved chunks;
- compressed recent turns;
- reconstructed historical memory;
- approximate original and forwarded token counts;
- estimated reduction percentage.

## Configuration

Every setting is documented inline in [.env.example](.env.example). Important
retrieval controls include:

| Variable | Purpose |
|---|---|
| `RECENT_TURNS_TO_KEEP` | Turns handled as recent continuity instead of older vector memory. |
| `RETRIEVAL_TOP_K` | Maximum vector matches requested from ChromaDB. |
| `RETRIEVAL_CHUNK_NEIGHBORS` | Chunks restored before and after each vector match. |
| `RETRIEVAL_INCLUDE_TURN_PAIR` | Keeps matching questions and answers together. |
| `SENTENCE_SCORE_THRESHOLD` | Relevance cutoff used while compressing recent turns. |
| `MAX_CONTEXT_TOKENS` | Overall approximate budget for generated context. |
| `RECENT_CONTEXT_TOKENS` | Budget reserved for recent conversation continuity. |
| `RETRIEVED_CONTEXT_TOKENS` | Budget reserved for older retrieved memory. |
| `OLLAMA_BOOTSTRAP` | Whether this app starts Ollama and pulls missing models. |

The token estimator intentionally uses a fast character approximation rather
than loading a model-specific tokenizer.

## API endpoints

### Ollama-compatible

- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `POST /api/embeddings`
- `GET /api/tags`

### Middleware and UI

- `GET /health`
- `GET /debug/context-preview`
- `POST /admin/reset`
- `GET /ui/api/conversations`
- `POST /ui/api/conversations`
- `GET /ui/api/conversations/{conversation_id}`
- `PATCH /ui/api/conversations/{conversation_id}`
- `DELETE /ui/api/conversations/{conversation_id}`
- `GET /ui/api/conversations/{conversation_id}/context`
- `POST /ui/api/chat`
- `GET /ui/api/gpu`

Interactive OpenAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Project layout

```text
src/ollama_middle_layer/
├── app.py               FastAPI proxy, UI API, and streaming
├── bootstrap.py         Local Ollama process and model management
├── config.py            Environment-based settings
├── context_pipeline.py  Retrieval, reconstruction, pruning, and budgets
├── ollama_clients.py    Chat/embedding clients and retrieval prefixes
├── pruning.py           Chunking, scoring, deduplication, and token limits
├── storage.py           SQLite history and ChromaDB vector index
├── gpu.py               Optional NVIDIA telemetry
└── web/                 Dependency-free browser interface
```

## Testing

```bash
.venv/bin/pytest -q
```

The tests cover storage ordering, retrieval expansion, neighboring chunks,
question/answer pairing, deduplication, context pruning helpers, API schemas,
bootstrap behavior, GPU fallback, and streamed assistant persistence.

## Current limitations

- Token counts are approximations, not model-tokenizer-exact values.
- Semantic recall depends on the embedding model, chunk boundaries, and chosen
  retrieval budgets.
- Global-history detection currently uses lightweight phrase patterns.
- The proxy does not currently rerank Chroma results with a cross-encoder.
- API assistant responses are learned from the transcript supplied on a later
  request; the included UI persists completed assistant streams immediately.
- The service has no authentication or TLS. Keep it bound to localhost unless
  it is placed behind a secure reverse proxy.
- GPU telemetry currently targets `nvidia-smi`; inference itself does not
  require an NVIDIA GPU.

## Roadmap

- Exact tokenizer-aware budgeting.
- Optional cross-encoder reranking.
- Configurable embedding chunk size and overlap.
- Hybrid BM25 and vector retrieval.
- Memory scoring based on recency, role, and retrieval frequency.
- Import/export and conversation-level index repair.
- Retrieval benchmarks for recall, latency, and prompt-token reduction.
- Docker and system service deployment examples.

## Positioning

This is not intended to replace general-purpose agent-memory platforms such as
Letta, Mem0, or Zep. It is a smaller and more transparent component:

> A local semantic context-virtualization proxy that gives Ollama applications
> access to long conversation history without forwarding the entire transcript
> on every request.

That narrow scope makes it useful for experiments, demonstrations, local chat
applications, and resource-constrained machines where prompt processing is the
bottleneck.

## Contributing

Issues and pull requests are welcome. Useful contributions include retrieval
quality tests, additional embedding models, alternative vector stores,
tokenizer integrations, deployment examples, and reproducible performance
benchmarks.

Before publishing the repository, add the open-source license you want to use
as a top-level `LICENSE` file.
