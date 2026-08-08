"""Isolate language as a variable, using parallel corpora.

The repository's own corpus mixes languages *and* content, so it cannot separate two very
different claims: "this model is worse at Spanish" and "this model is worse when the
question and the document disagree on language". The parallel corpora in ``corpora.py``
hold content constant and vary only language, which separates them cleanly.

Four conditions per corpus:

* ``EN/EN`` and ``ES/ES`` measure monolingual quality in each language.
* ``EN docs / ES questions`` and its mirror measure the cross-lingual penalty.

Usage::

    uv run python benchmarks/crosslingual.py
    uv run python benchmarks/crosslingual.py bge-m3 nomic-embed-text
"""

import statistics
import sys
from pathlib import Path

# Running a script puts its own directory on the path, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.corpora import ALL, ParallelCorpus
from benchmarks.retrieval import rank_of_first_hit, unit
from gieok.llm.ollama_client import OllamaClient

DEFAULT_MODELS = ("granite-embedding:278m", "bge-m3", "nomic-embed-text")
HOST = "http://localhost:11434"


def score(
    client: OllamaClient, notes: tuple[str, ...], queries: tuple[tuple[str, int], ...]
) -> tuple[int, float]:
    """Return exact matches and the mean margin over the best distractor.

    The margin matters as much as the hit count: on an easy set every decent model scores
    perfectly, and only the margin reveals how much headroom is left before a harder
    corpus starts producing errors.
    """
    documents = [unit(v) for v in client.embed(list(notes))]
    embedded = [unit(v) for v in client.embed([q for q, _ in queries])]

    hits, margins = 0, []
    for (_, target), vector in zip(queries, embedded, strict=True):
        scored = sorted(
            (
                (sum(a * b for a, b in zip(vector, d, strict=True)), i)
                for i, d in enumerate(documents)
            ),
            reverse=True,
        )
        hits += rank_of_first_hit(vector, documents, {target}) == 1
        margins.append(
            next(s for s, i in scored if i == target) - max(s for s, i in scored if i != target)
        )
    return hits, statistics.mean(margins)


def report(model: str, corpus: ParallelCorpus) -> None:
    """Print all four conditions for one model and corpus."""
    client = OllamaClient(host=HOST, embedding_model=model, chat_model="unused")
    conditions = (
        ("EN docs / EN questions", corpus.notes_en, corpus.queries_en),
        ("ES docs / ES questions", corpus.notes_es, corpus.queries_es),
        ("EN docs / ES questions", corpus.notes_en, corpus.queries_es),
        ("ES docs / EN questions", corpus.notes_es, corpus.queries_en),
    )
    total = len(corpus.queries_en)
    cells = []
    for _, notes, queries in conditions:
        hits, margin = score(client, notes, queries)
        cells.append(f"{hits:2}/{total} ({margin:+.3f})")
    print(f"  {model:26}" + "".join(f"{c:>18}" for c in cells))


def main() -> None:
    """Run the benchmark and print the report."""
    models = sys.argv[1:] or list(DEFAULT_MODELS)
    for corpus in ALL:
        print(f"\n=== {corpus.name} corpus ({len(corpus.notes_en)} notes) ===")
        print(
            f"  {'model':26}"
            + "".join(f"{h:>18}" for h in ("EN/EN", "ES/ES", "EN docs, ES q", "ES docs, EN q"))
        )
        for model in models:
            report(model, corpus)
    print("\nCell format: exact matches (mean margin over best distractor).")


if __name__ == "__main__":
    main()
