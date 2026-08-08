"""Retrieval, prompt construction and answer assembly."""

import pytest

from gieok.core.rag import PROMPT_TEMPLATE, RagService, build_prompt
from gieok.exceptions import EmptyCorpusError
from gieok.models import Chunk, RetrievedChunk

QUESTION = "What database does the project use?"


def seed(store, embedder, texts):
    chunks = [Chunk.create("notes.md", n, text) for n, text in enumerate(texts)]
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    return chunks


def make_service(embedder, store, chat, *, top_k=2):
    return RagService(embedder=embedder, store=store, chat=chat, top_k=top_k)


def test_build_prompt_embeds_context_and_question():
    chunk = Chunk.create("notes.md", 0, "The project stores vectors in ChromaDB.")
    prompt = build_prompt(QUESTION, [RetrievedChunk(chunk=chunk, score=0.9)])

    assert "The project stores vectors in ChromaDB." in prompt
    assert QUESTION in prompt
    assert "notes.md" in prompt
    assert "[1]" in prompt


def test_build_prompt_numbers_sources_sequentially():
    sources = [
        RetrievedChunk(chunk=Chunk.create("a.md", n, f"Fact {n}."), score=0.5) for n in range(3)
    ]
    prompt = build_prompt(QUESTION, sources)
    assert "[1]" in prompt and "[2]" in prompt and "[3]" in prompt


def test_prompt_instructs_the_model_to_refuse_when_unsupported():
    assert "do not contain the answer" in PROMPT_TEMPLATE
    assert "ONLY" in PROMPT_TEMPLATE


def test_ask_returns_answer_and_sources(embedder, store, chat):
    seed(store, embedder, ["The project stores vectors in ChromaDB.", "Unrelated filler text."])
    answer = make_service(embedder, store, chat).ask(QUESTION)

    assert answer.question == QUESTION
    assert answer.text == "".join(chat.fragments)
    assert len(answer.sources) == 2


def test_retrieved_context_reaches_the_model(embedder, store, chat):
    seed(store, embedder, ["ChromaDB is the vector database.", "Typer builds the CLI."])
    make_service(embedder, store, chat).ask(QUESTION)
    assert "ChromaDB is the vector database." in chat.last_prompt


def test_top_k_limits_retrieval(embedder, store, chat):
    seed(store, embedder, [f"Fact number {n} about the system." for n in range(10)])
    answer = make_service(embedder, store, chat, top_k=5).ask(QUESTION)
    assert len(answer.sources) == 5


def test_top_k_argument_overrides_the_default(embedder, store, chat):
    seed(store, embedder, [f"Fact number {n} about the system." for n in range(10)])
    answer = make_service(embedder, store, chat, top_k=5).ask(QUESTION, top_k=2)
    assert len(answer.sources) == 2


def test_results_are_ordered_by_descending_score(embedder, store, chat):
    seed(store, embedder, [f"Fact number {n} about the system." for n in range(8)])
    sources = make_service(embedder, store, chat, top_k=8).retrieve(QUESTION)
    assert [s.score for s in sources] == sorted((s.score for s in sources), reverse=True)


def test_empty_corpus_raises_before_calling_the_model(embedder, store, chat):
    service = make_service(embedder, store, chat)
    with pytest.raises(EmptyCorpusError):
        service.ask(QUESTION)
    assert chat.prompts == [], "the model must not be called without evidence"


def test_ask_stream_retrieves_eagerly(embedder, store, chat):
    seed(store, embedder, ["ChromaDB is the vector database."])
    sources, stream = make_service(embedder, store, chat).ask_stream(QUESTION)

    assert sources, "sources must be available before the stream is consumed"
    assert chat.prompts == [], "generation must stay lazy until the stream is iterated"
    assert "".join(stream) == "".join(chat.fragments)
