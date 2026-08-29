from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import VectorStore


QUESTIONS = [
    "What are the foundational paradigms for augmenting generation with retrieved results?",
    "What benefits can retrieval provide to large language models according to the survey?",
    "Why are Approximate Nearest Neighbor indexes used in dense retrieval?",
]


embedder = Embedder()
store = VectorStore()


for question in QUESTIONS:

    print("\n")
    print("=" * 80)
    print("QUESTION:")
    print(question)
    print("=" * 80)

    query_embedding = embedder.embed(
        [question]
    )[0]

    results = store.search(
        query_embedding,
        top_k=10
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for i, (text, metadata) in enumerate(
        zip(documents, metadatas)
    ):

        print("\n" + "-" * 80)

        print(f"RANK: {i + 1}")

        print(
            f"PAGE: {metadata.get('page')}"
        )

        print(
            f"SOURCE: {metadata.get('source')}"
        )

        print("\nTEXT:")

        print(text[:800])