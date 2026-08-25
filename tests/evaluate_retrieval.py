from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import VectorStore

from tests.evaluation_questions import EVALUATION_QUESTIONS


store = VectorStore()
embedder = Embedder()


for item in EVALUATION_QUESTIONS:
    question = item["question"]

    query_embedding = embedder.embed([question])[0]

    results = store.search(
        query_embedding,
        top_k=1
    )

    distance = results["distances"][0][0]

    print(
        f"Expected: "
        f"{'RELEVANT' if item['expected_relevant'] else 'IRRELEVANT'}"
    )

    print(f"Distance: {distance:.4f}")
    print(f"Question: {question}")
    print("-" * 60)