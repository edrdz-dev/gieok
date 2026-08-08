# Contributing

Thanks for taking an interest. This is a small project with a deliberate architecture, so
the notes below are mostly about keeping that architecture intact.

## Getting set up

```bash
uv sync
uv run pytest        # should pass immediately, with no Ollama running
```

If the tests need a model or a database to pass, something has gone wrong — see
*Architecture* below.

To actually run the CLI you need [Ollama](https://ollama.com/) and two models:

```bash
ollama pull granite-embedding:278m
ollama pull granite4.1:3b
```

## Before opening a pull request

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
```

CI runs exactly these four, so a green local run means a green pipeline.

## Architecture

Three layers, and the dependencies only ever point inward:

- `cli/` — Typer commands and Rich rendering. No business logic.
- `core/` — chunking, ingestion, RAG orchestration. Depends only on the `Protocol`
  definitions in `core/ports.py`, never on ChromaDB or Ollama.
- `db/`, `llm/`, `filesystem/` — adapters for the outside world.

The rule that matters: **`core/` must not import an infrastructure package.** If a change
requires it, the port is missing an abstraction. This is what keeps the whole test suite
runnable offline against the fakes in `tests/conftest.py`.

Other conventions, all enforced by tooling:

- Type hints on every function and method; `mypy --strict` must pass.
- A Google-style docstring on every public function and class.
- Catch specific exceptions. Infrastructure adapters translate third-party errors into the
  domain exceptions in `exceptions.py`; nothing outside those adapters should catch a
  vendor exception, and `except Exception` is never the answer.

## Changing a default

Defaults for models, chunk size and `top_k` are not opinions — each one is backed by a
measurement in [`benchmarks/`](benchmarks/). If you want to change one, please include the
benchmark output that justifies it, and read `benchmarks/README.md` first: it documents
several measurement traps (prompt-cache contamination, reasoning models reporting empty
output) that produce convincing but wrong numbers.

## Reporting a bug

Include the output of `uv run gieok status`, which reports the Ollama connection, the
resolved models and the collection size. Most issues are visible from there.
