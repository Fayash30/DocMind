from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import VectorStore

from tests.evaluation_dataset import EVALUATION_DATASET


def evaluate_retrieval(expected_pages, retrieved_pages):
    # For questions that should NOT be answerable,
    # retrieving nothing is the correct behavior.
    if not expected_pages:
        return len(retrieved_pages) == 0

    # For answerable questions,
    # at least one expected page should appear in the results.
    return any(
        page in expected_pages
        for page in retrieved_pages
    )


embedder = Embedder()
store = VectorStore()


total = 0
correct = 0


for item in EVALUATION_DATASET:

    question = item["question"]
    expected_pages = item["relevant_pages"]

    query_embedding = embedder.embed(
        [question]
    )[0]

    results = store.search(
        query_embedding,
        top_k=5
    )

    retrieved_pages = [
        metadata["page"]
        for metadata in results["metadatas"][0]
    ]

    hit = evaluate_retrieval(
        expected_pages,
        retrieved_pages
    )

    total += 1

    if hit:
        correct += 1

    print("\nQUESTION:")
    print(question)

    print("Expected pages:", expected_pages)
    print("Retrieved pages:", retrieved_pages)

    print(
        "RESULT:",
        "PASS" if hit else "FAIL"
    )


accuracy = correct / total


print("\n==============================")
print("RETRIEVAL EVALUATION")
print("==============================")

print(f"Passed: {correct}/{total}")
print(f"Accuracy: {accuracy:.2%}")