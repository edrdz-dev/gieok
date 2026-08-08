"""Compare generation models on RAG prompts: latency, faithfulness and refusal.

On a CPU the dominant cost is reading the prompt, not writing the answer: a RAG prompt is
roughly a thousand tokens of retrieved context. Prompt-evaluation throughput therefore
matters more than generation throughput, and both are reported separately.

Two measurement traps this script avoids:

* **Prefix caching.** Ollama caches the shared prefix of successive prompts, which makes a
  second run on the same context look impossibly fast. Every model here is given a
  different slice of the corpus, so no prefix is ever reused.
* **Reasoning models.** Some models put their scratchpad in ``thinking`` and leave
  ``content`` empty, so a naive harness scores them as producing nothing. Requests set
  ``think=False``, matching what the application itself does.

Usage::

    uv run python benchmarks/generation.py
    uv run python benchmarks/generation.py granite4.1:3b qwen3.5:4b
"""

import json
import secrets
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Running a script puts its own directory on the path, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.retrieval import build_corpus, flatten
from gieok.core.rag import build_prompt
from gieok.models import Chunk, RetrievedChunk

DEFAULT_MODELS = ("granite4.1:3b", "qwen3.5:4b", "llama3.1:8b")
HOST = "http://localhost:11434"
REFUSAL = "do not contain the answer"
TOP_K = 6
TIMEOUT_SECONDS = 900


@dataclass(frozen=True, slots=True)
class Timing:
    """Token counts and durations for one request."""

    prompt_tokens: int
    prompt_seconds: float
    output_tokens: int
    output_seconds: float
    text: str

    @property
    def prompt_rate(self) -> float:
        """Prompt-evaluation throughput in tokens per second."""
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds else 0.0

    @property
    def output_rate(self) -> float:
        """Generation throughput in tokens per second."""
        return self.output_tokens / self.output_seconds if self.output_seconds else 0.0

    @property
    def total_seconds(self) -> float:
        """Wall-clock cost of the whole request."""
        return self.prompt_seconds + self.output_seconds


def complete(model: str, prompt: str, max_tokens: int, nonce: str) -> Timing:
    """Send one non-streaming chat request and return its timings.

    Args:
        model: Ollama model name.
        prompt: The fully built RAG prompt.
        max_tokens: Cap on generated tokens, so a rambling model cannot skew the run.
        nonce: Unique prefix that guarantees a prompt-cache miss. Rotating the context
            alone is not enough, because an earlier run of this script may have left
            the very same prefix warm and would make one model look impossibly fast.

    Returns:
        Token counts, durations and the answer text.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": f"[run {nonce}]\n{prompt}"}],
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": 0, "seed": 1},
        }
    ).encode()
    request = urllib.request.Request(
        f"{HOST}/api/chat", payload, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        body = json.load(response)
    return Timing(
        prompt_tokens=body["prompt_eval_count"],
        prompt_seconds=body["prompt_eval_duration"] / 1e9,
        output_tokens=body["eval_count"],
        output_seconds=body["eval_duration"] / 1e9,
        text=body["message"].get("content", ""),
    )


def build_context(corpus: list[Chunk], marker: str, filler_start: int) -> list[RetrievedChunk]:
    """Assemble a context that always contains the answer, padded differently each time.

    The evidence chunk is held constant so every model answers from identical material and
    differences are the model's own. The padding rotates per model purely to defeat
    Ollama's prefix cache, which would otherwise make later models look faster.

    Args:
        corpus: All chunks.
        marker: Distinctive text identifying the chunk that answers the question.
        filler_start: Offset into the corpus for the padding chunks.

    Returns:
        Exactly ``TOP_K`` chunks, the first of which answers the question.

    Raises:
        LookupError: If no chunk contains ``marker``.
    """
    flat = flatten(marker)
    evidence = next((c for c in corpus if flat in flatten(c.text)), None)
    if evidence is None:
        raise LookupError(f"no chunk contains {marker!r}")

    filler = [c for c in corpus if c.id != evidence.id]
    rotated = filler[filler_start:] + filler[:filler_start]
    chosen = [evidence, *rotated[: TOP_K - 1]]
    return [RetrievedChunk(chunk=c, score=0.5) for c in chosen]


def main() -> None:
    """Run the benchmark and print the report."""
    models = sys.argv[1:] or list(DEFAULT_MODELS)
    corpus = build_corpus()
    # A distinct offset per model keeps Ollama's prefix cache from ever hitting.
    offsets = [(i * 17) % max(1, len(corpus) - TOP_K) for i in range(len(models))]

    # A crisp, unambiguous probe: the answer is a single identifier that either appears
    # verbatim or does not. Vaguer questions make "grounded" a matter of interpretation.
    nonce = secrets.token_hex(8)
    question = "Which environment variable controls how many chunks are retrieved?"
    marker = "`GIEOK_TOP_K`"
    expected = ("GIEOK_TOP_K",)
    off_topic = "What is the capital city of Australia?"

    for model, offset in zip(models, offsets, strict=True):
        context = build_context(corpus, marker, offset)
        answer = complete(model, build_prompt(question, context), 250, nonce)
        lowered = flatten(answer.text).lower()
        grounded = sum(term.lower() in lowered for term in expected)
        refusal = complete(model, build_prompt(off_topic, context), 80, nonce)
        refuses = REFUSAL in refusal.text.lower()

        print(
            f"{model:22}"
            f"{answer.prompt_tokens:6}t {answer.prompt_rate:6.1f}/s"
            f"{answer.output_tokens:7}t {answer.output_rate:6.2f}/s"
            f"{answer.total_seconds:8.1f}s"
            f"{grounded:7}/{len(expected)}"
            f"{'yes' if refuses else 'NO':>9}"
        )
        if not refuses:
            print(f"    hallucinated: {flatten(refusal.text)[:88]}")


if __name__ == "__main__":
    main()
