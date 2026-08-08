"""Ground truth for retrieval over this repository's own source and documentation.

Forty-four questions, each paired with markers identifying which chunks legitimately
answer it. Design notes, because they are what make the numbers mean anything:

* **Markers, not chunk indices.** A chunk is acceptable if its text contains any of the
  question's markers. Indices would silently rot the moment a source file is edited and
  the chunk boundaries shift; a distinctive substring survives re-chunking.
* **A set of acceptable chunks, not one.** What retrieval owes the model is *some* passage
  that answers the question. Insisting on one particular chunk would penalise a model for
  picking an equally valid neighbour.
* **Deliberately paraphrased.** Questions avoid the vocabulary of the passage they target,
  so the benchmark measures semantic retrieval rather than keyword overlap.
* **Half cross-lingual.** The source is English and CLAUDE.md is Spanish, so asking in the
  other language is the realistic failure case and gets equal weight.
"""

from dataclasses import dataclass
from enum import StrEnum


class Direction(StrEnum):
    """Language of the question relative to the document that answers it."""

    EN_TO_EN = "EN->EN"
    ES_TO_EN = "ES->EN"
    ES_TO_ES = "ES->ES"
    EN_TO_ES = "EN->ES"


@dataclass(frozen=True, slots=True)
class Question:
    """A question and the markers identifying passages that answer it."""

    text: str
    markers: tuple[str, ...]
    direction: Direction


def _pair(en: str, es: str, *markers: str, spanish_source: bool = False) -> list[Question]:
    """Build the English and Spanish forms of one question sharing the same targets."""
    if spanish_source:
        forward, cross = Direction.ES_TO_ES, Direction.EN_TO_ES
    else:
        forward, cross = Direction.EN_TO_EN, Direction.ES_TO_EN
    return [
        Question(en, markers, cross if spanish_source else forward),
        Question(es, markers, forward if spanish_source else cross),
    ]


QUESTIONS: tuple[Question, ...] = tuple(
    q
    for group in (
        # --- answered by the English source code ---
        _pair(
            "Which module is the only one allowed to import Rich?",
            "¿Que modulo es el unico que puede importar Rich?",
            "rich`` import in the project lives",
        ),
        _pair(
            "How is a fragment identifier kept stable between runs?",
            "¿Como se garantiza que el identificador de un fragmento no cambie?",
            "deterministic, content-derived id",
        ),
        _pair(
            "Why is the vector store told not to embed anything itself?",
            "¿Por que se le dice a la base vectorial que no genere embeddings?",
            "ships its own ONNX",
        ),
        _pair(
            "Which construct converts library failures into domain failures?",
            "¿Que construccion convierte fallos de libreria en fallos de dominio?",
            "builds one from a generator",
            "One class implements two ports",
        ),
        _pair(
            "How is peak memory kept independent of how big the corpus is?",
            "¿Como se evita que la memoria dependa del tamano del corpus?",
            "peak memory depends on",
        ),
        _pair(
            "What replaced the module-level TypeVar declaration?",
            "¿Que sintaxis sustituyo a la declaracion de TypeVar del modulo?",
            "replaces the older module-level",
        ),
        _pair(
            "Where are the concrete adapters actually instantiated?",
            "¿Donde se instancian realmente los adaptadores concretos?",
            "composition root",
        ),
        _pair(
            "What happens when a domain exception escapes a command?",
            "¿Que ocurre cuando una excepcion de dominio escapa de un comando?",
            "raise typer.Exit(code=1) from exc",
        ),
        _pair(
            "Why must the model's refusal be all or nothing?",
            "¿Por que el rechazo del modelo debe ser todo o nada?",
            "worse than either outcome alone",
        ),
        _pair(
            "How is an implicit latest tag handled when matching model names?",
            "¿Como se trata la etiqueta latest implicita al comparar modelos?",
            "tolerating an implicit :latest tag",
        ),
        _pair(
            "What is the distance returned by the database turned into?",
            "¿En que se convierte la distancia que devuelve la base de datos?",
            "similarity is its complement",
        ),
        _pair(
            "Which error is raised when nothing has been indexed yet?",
            "¿Que error se lanza cuando todavia no hay nada indexado?",
            "class EmptyCorpusError",
        ),
        _pair(
            "What keeps every slice within the configured length limit?",
            "¿Que asegura que cada porcion respete el limite de longitud?",
            "every emitted chunk satisfies",
        ),
        _pair(
            "How is interface conformance checked without any inheritance?",
            "¿Como se comprueba la conformidad de interfaces sin herencia?",
            "nothing to inherit and",
        ),
        _pair(
            "Which environment variable sets how many fragments are retrieved?",
            "¿Que variable de entorno fija cuantos fragmentos se recuperan?",
            "GIEOK_TOP_K",
        ),
        _pair(
            "Why does indexing the same files twice not duplicate anything?",
            "¿Por que indexar dos veces los mismos ficheros no duplica nada?",
            "chunk ids are derived from content",
            "identical chunks overwrite",
        ),
        # --- answered by the Spanish CLAUDE.md ---
        _pair(
            "What docstring format does the project require?",
            "¿Que formato de docstring exige el proyecto?",
            "Google o Sphinx",
            spanish_source=True,
        ),
        _pair(
            "Is composition preferred over deep class hierarchies?",
            "¿Se prefiere la composicion o las jerarquias de clases profundas?",
            "Composición sobre Herencia",
            spanish_source=True,
        ),
        _pair(
            "What proportion of functions need type annotations?",
            "¿Que proporcion de funciones necesita anotaciones de tipo?",
            "100% de las funciones",
            spanish_source=True,
        ),
        _pair(
            "Should answers begin with long greetings?",
            "¿Deben las respuestas empezar con saludos largos?",
            "Cero *Boilerplate*",
            spanish_source=True,
        ),
        _pair(
            "What should a complex concept be compared to when explaining it?",
            "¿Con que conviene comparar un concepto complejo al explicarlo?",
            "arquitectura empresarial",
            spanish_source=True,
        ),
        _pair(
            "Which dependency manager should be avoided?",
            "¿Que gestor de dependencias hay que evitar?",
            "evitar `pip` crudo",
            spanish_source=True,
        ),
    )
    for q in group
)
