import pytest
from pydantic import BaseModel
from takeoff_sdk import Takeoff

from takeoff_haystack import TakeoffGenerator


class HaystackResponse(BaseModel):
    replies: list[str]


@pytest.fixture(scope="module")
def takeoff_port():
    takeoff = Takeoff(model_name="sshleifer/tiny-gpt2", device="cpu")
    takeoff.start()
    yield takeoff.takeoff_port
    takeoff.cleanup()


@pytest.mark.parametrize(
    "generation_kwargs",
    [
        {"sampling_temperature": 0.5},
        {"sampling_temperature": 0.5, "max_new_tokens": 100, "min_new_tokens": 10},
        {
            "sampling_temperature": 0.5,
            "max_new_tokens": 100,
            "min_new_tokens": 10,
            "sampling_topp": 0.9,
            "sampling_topk": 50,
        },
    ],
)
def test_takeoff_generator_generation_kwargs(generation_kwargs, takeoff_port):
    generator = TakeoffGenerator(
        base_url="http://localhost", port=int(takeoff_port), generation_kwargs=generation_kwargs
    )
    result = generator.run("This is a test prompt.")
    HaystackResponse(**result)


def test_docs_example(takeoff_port):
    from haystack import Document, Pipeline
    from haystack.components.builders.prompt_builder import PromptBuilder
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    from takeoff_haystack import TakeoffGenerator

    document_store = InMemoryDocumentStore()
    document_store.write_documents(
        [
            Document(content="Super Mario was an important politician"),
            Document(content="Mario owns several castles and uses them to conduct important political business"),
            Document(
                content="Super Mario was a successful military leader who fought off several invasion attempts by "
                "his arch rival - Bowser"
            ),
        ]
    )

    template = """
    Given only the following information, answer the question.
    Ignore your own knowledge.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ query }}?
    """

    pipe = Pipeline()

    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("llm", TakeoffGenerator(base_url="http://localhost", port=int(takeoff_port)))
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")

    query = "Who is Super Mario?"

    response = pipe.run({"prompt_builder": {"query": query}, "retriever": {"query": query}})

    print(response["llm"]["replies"])
