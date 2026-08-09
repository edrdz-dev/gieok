"""Compare embedding models on retrieval quality.

Run against this repository's own source and documentation, which is a genuinely mixed
corpus: English code with English docstrings beside a Spanish CLAUDE.md. That mix is the
whole point -- it is where the popular English-first embedders fall apart.

Usage::

    uv run python benchmarks/retrieval.py                    # default model set
    uv run python benchmarks/retrieval.py bge-m3 granite-embedding:278m

Results are deterministic: identical input produces bit-identical embeddings, so a rerun
on the same machine reproduces exactly, and a rerun on different hardware reproduces to
well within the margins reported here.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

# Running a script puts its own directory on the path, not the repository root, so add
# the root explicitly to support both `python benchmarks/retrieval.py` and `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.questions import QUESTIONS, Direction, Question
from benchmarks.stats import mcnemar_exact, wilson_interval
from gieok.core.chunking import chunk_text
from gieok.filesystem.loader import iter_documents
from gieok.llm.ollama_client import OllamaClient
from gieok.models import Chunk

DEFAULT_MODELS = (
    "granite-embedding:278m",
    "nomic-embed-text-v2-moe",
    "paraphrase-multilingual",
    "embeddinggemma",
    "bge-m3",
    "nomic-embed-text",
)
DEPTHS = (1, 2, 4, 6, 8, 10)
PRIMARY_DEPTH = 6
HOST = "http://localhost:11434"


def build_corpus() -> list[Chunk]:
    """Chunk the benchmark corpus the same way the application would.

    The corpus is named explicitly rather than globbed over the whole repository, because
    the ground truth in ``questions.py`` pins exact phrases and would silently rot every
    time an unrelated document was reworded. It deliberately mixes languages: English
    source with English docstrings, one English document, and one Spanish fixture. That
    mix is the condition under which the default embedding model was chosen, so the
    benchmark has to reproduce it.

    ``CLAUDE.md`` is excluded on purpose. It is a near-translation of the Spanish fixture,
    and having both would let a model answer a Spanish-targeted question from the English
    twin, scoring a legitimate answer as a miss.

    ``fixtures/wrapped-text.pdf`` adds a paginated source: two pages of hard-wrapped,
    justified-looking lines built specifically to exercise ``filesystem.pdf._normalise``'s
    paragraph-reconstruction step, the one heuristic in this feature (the ``0.75 * width``
    fill-ratio threshold) invented rather than measured.
    """
    root = Path(__file__).resolve().parent.parent
    documents = list(iter_documents(root / "src", ["*.py"]))
    documents += list(iter_documents(root / "README.md"))
    documents += list(iter_documents(root / "benchmarks" / "fixtures", ["*.md"]))
    documents += list(iter_documents(root / "benchmarks" / "fixtures", ["*.pdf"]))
    return [
        chunk
        for document in documents
        for chunk in chunk_text(
            document.content, source=str(document.source), size=800, overlap=150
        )
    ]


def flatten(text: str) -> str:
    """Collapse all runs of whitespace, so a marker survives a line wrap.

    Source files wrap at 100 columns, which routinely splits a marker phrase across two
    lines. Matching on flattened text makes the ground truth immune to reformatting.
    """
    return " ".join(text.split())


def resolve_targets(question: Question, corpus: list[Chunk]) -> set[int]:
    """Return the indices of every chunk that legitimately answers ``question``.

    Raises:
        LookupError: If no chunk contains any marker, which means the ground truth has
            drifted away from the corpus and the run would be meaningless.
    """
    markers = [flatten(m) for m in question.markers]
    targets = {
        i for i, chunk in enumerate(corpus) if any(m in flatten(chunk.text) for m in markers)
    }
    if not targets:
        raise LookupError(f"no chunk matches {question.markers!r} for: {question.text}")
    return targets


def rank_of_first_hit(query: list[float], vectors: list[list[float]], targets: set[int]) -> int:
    """Return the 1-based rank of the best-placed acceptable chunk."""
    scored = sorted(
        ((sum(a * b for a, b in zip(query, v, strict=True)), i) for i, v in enumerate(vectors)),
        reverse=True,
    )
    return next(rank for rank, (_, i) in enumerate(scored, start=1) if i in targets)


def unit(vector: list[float]) -> list[float]:
    """Normalise to unit length so a dot product is the cosine similarity."""
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else vector


def evaluate(model: str, corpus: list[Chunk], targets: list[set[int]]) -> tuple[list[int], float]:
    """Return the rank achieved for every question, and seconds spent indexing."""
    client = OllamaClient(host=HOST, embedding_model=model, chat_model="unused")
    started = time.perf_counter()
    document_vectors = [unit(v) for v in client.embed([c.text for c in corpus])]
    elapsed = time.perf_counter() - started
    query_vectors = [unit(v) for v in client.embed([q.text for q in QUESTIONS])]
    return [
        rank_of_first_hit(qv, document_vectors, t)
        for qv, t in zip(query_vectors, targets, strict=True)
    ], elapsed


def main() -> None:
    """Run the benchmark and print the report."""
    models = sys.argv[1:] or list(DEFAULT_MODELS)
    corpus = build_corpus()
    targets = [resolve_targets(q, corpus) for q in QUESTIONS]
    total = len(QUESTIONS)
    print(f"corpus {len(corpus)} chunks | {total} questions | primary depth k={PRIMARY_DEPTH}\n")

    ranks: dict[str, list[int]] = {}
    header = f"{'model':26}" + "".join(f"{'k=' + str(k):>7}" for k in DEPTHS) + f"{'index':>9}"
    print(header)
    print("-" * len(header))
    for model in models:
        ranks[model], elapsed = evaluate(model, corpus, targets)
        cells = "".join(f"{sum(1 for r in ranks[model] if r <= k):7}" for k in DEPTHS)
        print(f"{model:26}{cells}{elapsed:8.1f}s")

    print(f"\nBy direction at k={PRIMARY_DEPTH}:\n")
    directions = list(Direction)
    print(f"{'model':26}" + "".join(f"{d:>9}" for d in directions))
    for model in models:
        buckets: dict[Direction, list[int]] = defaultdict(list)
        for question, rank in zip(QUESTIONS, ranks[model], strict=True):
            buckets[question.direction].append(rank)
        cells = "".join(
            f"{sum(1 for r in buckets[d] if r <= PRIMARY_DEPTH):5}/{len(buckets[d]):<4}"
            for d in directions
        )
        print(f"{model:26}{cells}")

    best = max(models, key=lambda m: sum(1 for r in ranks[m] if r <= PRIMARY_DEPTH))
    hits = {m: [r <= PRIMARY_DEPTH for r in ranks[m]] for m in models}
    low, high = wilson_interval(sum(hits[best]), total)
    print(
        f"\nPaired comparison against {best} ({sum(hits[best])}/{total}, "
        f"95% CI [{low:.0%}, {high:.0%}]):\n"
    )
    for model in models:
        if model == best:
            continue
        result = mcnemar_exact(hits[best], hits[model])
        verdict = "SIGNIFICANT" if result.significant else "indistinguishable"
        print(
            f"  vs {model:26} {sum(hits[model]):2}/{total}  "
            f"+{result.wins:2}/-{result.losses:<2}  p={result.p_value:.4f}  {verdict}"
        )

    # A question every model misses is evidence about the corpus, not about the models:
    # usually a passage split across a chunk boundary. Only meaningful with several models.
    others = [m for m in models if m != "nomic-embed-text"]
    if len(others) >= 2:
        universal = [
            q.text
            for n, q in enumerate(QUESTIONS)
            if all(ranks[m][n] > PRIMARY_DEPTH for m in others)
        ]
        if universal:
            print(
                f"\nMissed by all {len(others)} multilingual models "
                f"(likely a chunking problem, not a model one):"
            )
            for text in universal:
                print(f"  - {text}")


if __name__ == "__main__":
    main()
