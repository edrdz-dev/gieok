# Benchmarks

The measurements behind the defaults in `config.py`. Every number in the project README
comes from here and can be reproduced with one command.

```bash
uv run python benchmarks/retrieval.py     # embedding models: recall, MRR, significance
uv run python benchmarks/generation.py    # chat models: latency, groundedness, refusal
```

Both accept model names as arguments to override the default set:

```bash
uv run python benchmarks/retrieval.py bge-m3 granite-embedding:278m
```

Ollama must be running, and the models under test must already be pulled. Only the two
application defaults are needed to *use* the project; the rest exist solely to be compared
against, so pull them on demand:

```bash
# Embedding models compared in retrieval.py / crosslingual.py
ollama pull granite-embedding:278m nomic-embed-text-v2-moe paraphrase-multilingual
ollama pull embeddinggemma bge-m3 nomic-embed-text

# Generation models compared in generation.py
ollama pull granite4.1:3b qwen3.5:4b llama3.1:8b
```

That is roughly 14 GB. Delete them again with `ollama rm <model>` once you are done — the
defaults `granite-embedding:278m` and `granite4.1:3b` are the only two the CLI itself needs.

## What is measured

`retrieval.py` chunks this repository's own source and documentation exactly as the
application would, then asks 44 questions with known answers and records the rank of the
first acceptable chunk. The corpus is deliberately mixed — English code and docstrings
beside a Spanish `CLAUDE.md` — because a language-mixed corpus is what breaks the popular
English-first embedders.

`generation.py` holds the retrieved context fixed and varies only the model, so differences
are the generator's own rather than the retriever's.

## Methodology notes

These are the traps that produced wrong answers during development, all of them now
handled in code:

**Markers, not chunk indices.** Ground truth in `questions.py` identifies target passages
by distinctive substring. Indices rot the moment a source file is edited and chunk
boundaries shift. Matching happens on whitespace-flattened text, so a marker survives a
line wrap — a subtle one, since source files wrap at 100 columns and routinely split a
marker phrase across two lines.

**A set of acceptable chunks, not one.** Retrieval owes the model *some* passage that
answers the question. Demanding one specific chunk penalises a model for choosing an
equally valid neighbour.

**Paired statistics.** All models answer the same questions, so comparisons use an exact
McNemar test (`stats.py`). Questions that every model gets right, or every model gets
wrong, carry no information about which is better; only disagreements count. Treating the
result sets as independent samples throws away most of the power.

**Prompt-cache defeat.** Ollama caches the shared prefix of successive prompts. Rotating
the context per model is not enough, because a previous run of the script can leave the
same prefix warm — during development this produced a reading of 6945 tok/s for a model
that actually runs at 38. `generation.py` therefore prefixes every prompt with a
per-run nonce.

**Reasoning models.** Some models put their output in the response's `thinking` field and
leave `content` empty, scoring zero in a naive harness. Requests set `think=False`,
matching the application.

**Determinism.** Embeddings are bit-identical across calls, and generation with
`temperature=0` reproduces exactly. A rerun on the same machine reproduces to the digit;
a rerun on different hardware changes the timings completely but not the retrieval
results, since the same weights produce the same vectors.

## Current results

CPU-only, Intel i5-1135G7, no discrete GPU. Corpus 113 chunks, 44 questions, k=6.

### Embedding models

| Model | Size | Recall@6 | ES→EN | Index time | vs. best |
| --- | --- | --- | --- | --- | --- |
| granite-embedding:278m | 562 MB | 39/44 | 15/16 | 28.6s | — |
| paraphrase-multilingual | 562 MB | 39/44 | 14/16 | **19.6s** | p=1.000 |
| nomic-embed-text-v2-moe | 957 MB | 38/44 | 13/16 | 58.6s | p=1.000 |
| bge-m3 | 1.2 GB | 38/44 | 14/16 | 92.4s | p=1.000 |
| embeddinggemma | 621 MB | 32/44 | 10/16 | 35.9s | p=0.092 |
| nomic-embed-text | 274 MB | 21/44 | **2/16** | 33.7s | **p=0.0001** |

The five multilingual models are statistically indistinguishable from one another. The
default picks among equals on size and indexing speed, not on accuracy — a smaller pilot
appeared to rank them and that ranking did not survive a larger sample.

What does survive is the gap to `nomic-embed-text`, the most-downloaded embedding model on
Ollama, which loses 20-2 head to head and answers 2 of 16 cross-lingual questions.

### Generation models

| Model | Size | Prompt eval | Generation | Total | Grounded | Refuses |
| --- | --- | --- | --- | --- | --- | --- |
| granite4.1:3b | 2.1 GB | 38.1 tok/s | 7.55 tok/s | **27.7s** | 1/1 | yes |
| qwen3.5:4b | 3.4 GB | 30.0 tok/s | 6.02 tok/s | 48.0s | 1/1 | yes |
| llama3.1:8b | 4.9 GB | 16.2 tok/s | 3.58 tok/s | 95.0s | 1/1 | yes |

On CPU, reading the prompt costs more than writing the answer.

## Known limitations

**The generation benchmark is thin.** One grounded question and one refusal per model is a
smoke test, not an evaluation — it catches a model that ignores its context or hallucinates
past a refusal, and nothing subtler. Running the full 44 questions through generation was
prohibitive on CPU at 30-95s per query; on a GPU it takes minutes and is the obvious next
step.

**44 questions bounds what is detectable.** Differences smaller than roughly 15% stay
invisible. That is adequate here, because the models within that band are interchangeable
in practice, but it does mean "indistinguishable" should be read as "not shown to differ"
rather than "shown to be equal".

**One corpus, one author.** The questions were written by the same person who wrote the
code being queried. They are paraphrased to avoid rewarding lexical overlap, but a second
corpus from a different domain would test generality.
