# gieok

Ask questions about your own documents, entirely offline.

*gieok* — Korean 기억, "memory". Which is what this is: somewhere to put what you have
read, so you can recall it later with the page still attached.

A retrieval-augmented generation (RAG) CLI that runs entirely on your machine: index your
documents into a local vector database, then question them with a local LLM. Nothing leaves
the host, and every answer cites the passages it came from.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), running locally

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh

ollama pull granite-embedding:278m   # embeddings, 768-dim, multilingual
ollama pull granite4.1:3b            # generation
```

## Install

```bash
uv sync
```

## Usage

```bash
uv run gieok status                      # daemon reachability, models, collection size
uv run gieok ingest ./docs               # index *.md and *.txt recursively
uv run gieok ingest ./docs --reset       # rebuild the collection from scratch
uv run gieok ingest ./src -p '*.py'      # custom glob, repeatable
uv run gieok ask "How is retrieval configured?"
uv run gieok ask "..." --top-k 8 --no-sources
```

Re-running `ingest` over unchanged files is idempotent: chunk ids are derived from content,
so identical chunks overwrite themselves rather than accumulating.

## Configuration

Every setting is an environment variable prefixed `LAE_`, or a line in a local `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LAE_OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon address |
| `LAE_EMBEDDING_MODEL` | `granite-embedding:278m` | Model used for embeddings |
| `LAE_CHAT_MODEL` | `granite4.1:3b` | Model used for generation |
| `LAE_CHROMA_PATH` | `.chroma` | Where ChromaDB persists |
| `LAE_COLLECTION_NAME` | `documents` | Collection name |
| `LAE_CHUNK_SIZE` | `800` | Max chunk length, in characters |
| `LAE_CHUNK_OVERLAP` | `150` | Context carried between chunks |
| `LAE_EMBED_BATCH_SIZE` | `32` | Chunks embedded per round trip |
| `LAE_TOP_K` | `6` | Chunks retrieved per question |

Changing the embedding model invalidates the index — re-run `ingest --reset`.

### Choosing models

Every figure below is reproducible with `uv run python benchmarks/retrieval.py`
and `benchmarks/generation.py`; see [benchmarks/README.md](benchmarks/README.md) for the
methodology and its known limitations.

Both defaults were picked by measurement on a CPU-only laptop (i5-1135G7, no discrete GPU),
because that is the setup where the choice actually bites.

**Embeddings.** Benchmarked on this repository's own source and docs — 106 chunks, 44
questions with a known set of acceptable target chunks, deliberately paraphrased so lexical
overlap does not do the work. Half the questions are asked in the other language from the
document that answers them, because that is the case that breaks retrieval.

| Model | Size | Recall@6 | Recall@4 | MRR | vs. best (paired McNemar) |
| --- | --- | --- | --- | --- | --- |
| nomic-embed-text-v2-moe | 957 MB | 36/44 | 33/44 | 0.582 | — |
| paraphrase-multilingual | 562 MB | 38/44 | 32/44 | 0.566 | p=1.000 |
| **granite-embedding:278m** | 562 MB | 37/44 | 31/44 | 0.572 | p=0.754 |
| embeddinggemma | 621 MB | 29/44 | 28/44 | 0.380 | p=0.227 |
| bge-m3 | 1.2 GB | 31/44 | 28/44 | 0.574 | p=0.180 |
| nomic-embed-text | 274 MB | 19/44 | 16/44 | 0.338 | **p<0.0001** |

The five multilingual models are **statistically indistinguishable from one another** — a
smaller pilot on 13 chunks appeared to rank them, and that ranking did not survive a larger
sample. What does survive is the gap to `nomic-embed-text`, the most-downloaded embedding
model on Ollama, which loses 17-0 head-to-head against the leader and collapses to 1/16 on
questions asked in the other language.

`granite-embedding:278m` is the default because among equals it is the smallest and the
fastest to index (16.9s for the corpus, against 53.1s for `nomic-embed-text-v2-moe` and
83.7s for `bge-m3`) — not because it retrieves better.

**Retrieval depth matters more than the model.** On the same corpus, raising `top_k` from 4
to 6 lifted recall by more than any model swap, and unlike the model swap it is significant:

| | Recall@4 | Recall@6 | Recall@8 |
| --- | --- | --- | --- |
| granite-embedding:278m | 31/44 | 37/44 (p=0.031) | 40/44 (p=0.250 vs. 6) |
| paraphrase-multilingual | 32/44 | 38/44 (p=0.031) | 40/44 (p=0.500 vs. 6) |

Each extra chunk costs roughly 200 prompt tokens, and CPU prompt evaluation runs at about
37 tok/s — so ~5s per chunk per question. Going to 8 buys no significant recall for another
ten seconds, which is why the default is 6.

**Generation.** On CPU, reading the prompt costs more than writing the answer. Same prompt,
cold start, cache-busting context:

| Model | Size | Prompt eval | Generation | Total | Facts | Refuses correctly |
| --- | --- | --- | --- | --- | --- | --- |
| **granite4.1:3b** | 2.1 GB | 41.0 tok/s | 7.45 tok/s | **30.3s** | 2/2 | yes |
| qwen3.5:4b | 3.4 GB | 29.8 tok/s | 6.00 tok/s | 60.2s | 2/2 | yes |
| llama3.1:8b | 4.9 GB | 16.4 tok/s | 3.73 tok/s | 84.5s | 2/2 | hedges |
| qwen3.5:2b | 1.9 GB | 241.5 tok/s | 14.95 tok/s | 4.7s | **0/2** | over-refuses |

`granite4.1:3b` answers 2.8x faster than `llama3.1:8b` with the same factual accuracy and a
cleaner refusal. `qwen3.5:2b` is the fastest by a mile and useless: it refuses questions its
context plainly answers.

**A note on reasoning models.** Qwen 3.5 emits its scratchpad in the response's `thinking`
field and leaves `content` empty until it finishes. Left alone it renders as a blank panel
and can spend its whole token budget deliberating before answering. The Ollama adapter
therefore passes `think=False` (`llm/ollama_client.py`). Extractive RAG does not need
deliberation, and on CPU those tokens are the scarcest resource available.

Models differ in dimensionality, so switching the embedder always requires `--reset` and a
full re-index.

## Architecture

Ports and adapters, in three layers:

```
cli/          presentation   Typer commands, Rich rendering, composition root
core/         domain         chunking, ingestion, RAG orchestration, port definitions
db/ llm/ filesystem/   infrastructure   ChromaDB, Ollama, filesystem adapters
```

`core/ports.py` declares what the domain needs as `typing.Protocol` classes. Adapters
satisfy them structurally — nothing to inherit, nothing to register — so `core/` never
imports `chromadb` or `ollama`. The object graph is assembled once in
`cli/app.py:build_services`, and every service takes its collaborators through its
constructor.

The practical payoff: the entire test suite runs with no Ollama daemon and no vector
database, against in-memory fakes in `tests/conftest.py`.

## Development

```bash
uv run pytest              # no external services required
uv run ruff check .
uv run ruff format .
uv run mypy src
```

Benchmarks live in [`benchmarks/`](benchmarks/) and need a running Ollama:

```bash
uv run python benchmarks/retrieval.py      # embedding models
uv run python benchmarks/generation.py     # chat models
uv run python benchmarks/crosslingual.py   # language isolated as a variable
```
