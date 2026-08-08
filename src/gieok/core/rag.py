"""Retrieval-augmented generation: retrieve evidence, then answer strictly from it.

The prompt is deliberately restrictive. A local 8B model left to its own devices will
happily answer from parametric memory and present it as if it came from your documents,
which is the single worst failure mode for a tool like this. Instructing it to decline is
cheaper and far more reliable than trying to detect the hallucination afterwards.

The refusal is framed as all-or-nothing on purpose. An earlier wording ("if the context does
not contain the answer, reply ...") made the model hedge: it emitted the refusal sentence and
then answered anyway, which is worse than either outcome alone. Partial evidence is now
explicitly routed to "answer and flag the gap" rather than to the refusal branch.
"""

from collections.abc import Iterator, Sequence

from gieok.core.ports import ChatModel, Embedder, VectorStore
from gieok.exceptions import EmptyCorpusError
from gieok.models import Answer, RetrievedChunk

PROMPT_TEMPLATE = """\
You are a precise research assistant. Answer the question using ONLY the context below.

Rules:
- Never use outside knowledge. Everything you state must be supported by the context.
- If nothing in the context is relevant, your entire reply must be exactly this sentence, with \
nothing before or after it: "The provided documents do not contain the answer to that question."
- Otherwise, answer from the context and do NOT include that sentence. Partial evidence still \
counts as an answer: give what the context supports, then note what is missing.
- Cite the sources you relied on by their [n] marker.
- Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""

_CONTEXT_ENTRY = "[{n}] (source: {source})\n{text}"


def build_prompt(question: str, sources: Sequence[RetrievedChunk]) -> str:
    """Render the grounded prompt sent to the model.

    Kept as a module-level pure function so tests can assert on the exact prompt without
    standing up a service or a model.

    Args:
        question: The user's question, verbatim.
        sources: Retrieved chunks, most relevant first.

    Returns:
        The fully rendered prompt.
    """
    context = "\n\n".join(
        _CONTEXT_ENTRY.format(n=n, source=retrieved.chunk.source, text=retrieved.chunk.text)
        for n, retrieved in enumerate(sources, start=1)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


class RagService:
    """Answers questions grounded in an indexed corpus."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        chat: ChatModel,
        top_k: int,
    ) -> None:
        """Wire the retrieval and generation collaborators.

        Args:
            embedder: Embeds the question into the corpus vector space.
            store: Source of candidate chunks.
            chat: Generates the final answer.
            top_k: Default number of chunks to retrieve.
        """
        self._embedder = embedder
        self._store = store
        self._chat = chat
        self._top_k = top_k

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """Find the chunks most relevant to ``question``.

        Args:
            question: The user's question.
            top_k: Overrides the configured default when given.

        Returns:
            Retrieved chunks, most relevant first.

        Raises:
            EmptyCorpusError: If nothing has been indexed yet.
        """
        if self._store.count() == 0:
            raise EmptyCorpusError

        (embedding,) = self._embedder.embed([question])
        return self._store.query(embedding, top_k or self._top_k)

    def ask_stream(
        self, question: str, *, top_k: int | None = None
    ) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Retrieve evidence eagerly, then return a lazy stream of answer fragments.

        Splitting the two lets the CLI render sources immediately and stream tokens as they
        arrive. Retrieval happens before returning, so failures surface here rather than
        halfway through rendering a live view.

        Args:
            question: The user's question.
            top_k: Overrides the configured default when given.

        Returns:
            The retrieved sources, and an iterator over answer fragments.

        Raises:
            EmptyCorpusError: If nothing has been indexed yet.
        """
        sources = self.retrieve(question, top_k=top_k)
        return sources, self._chat.stream(build_prompt(question, sources))

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        """Answer a question, collecting the stream into a single value.

        Args:
            question: The user's question.
            top_k: Overrides the configured default when given.

        Returns:
            The complete answer plus the evidence it was grounded on.

        Raises:
            EmptyCorpusError: If nothing has been indexed yet.
        """
        sources, stream = self.ask_stream(question, top_k=top_k)
        return Answer(question=question, text="".join(stream), sources=tuple(sources))
