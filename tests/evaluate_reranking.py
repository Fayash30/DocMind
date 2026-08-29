from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import VectorStore
from app.retrieval.reranker import Reranker

from tests.evaluation_dataset import EVALUATION_DATASET


def reciprocal_rank(expected_pages, retrieved_pages):
    """
    Return the reciprocal rank of the first relevant result.

    Example:
        relevant at position 1 -> 1.0
        relevant at position 2 -> 0.5
        relevant at position 3 -> 0.333
    """

    for index, page in enumerate(retrieved_pages):
        if page in expected_pages:
            return 1 / (index + 1)

    return 0.0


embedder = Embedder()
store = VectorStore()
reranker = Reranker()


baseline_scores = []
reranked_scores = []


for item in EVALUATION_DATASET:

    question = item["question"]
    expected_pages = item["relevant_pages"]

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    # --------------------------------------------------
    # 1. Embed the question
    # --------------------------------------------------

    query_embedding = embedder.embed(
        [question]
    )[0]


    # --------------------------------------------------
    # 2. Vector search
    # --------------------------------------------------

    results = store.search(
        query_embedding,
        top_k=5
    )


    retrieved_chunks = []

    for i, text in enumerate(results["documents"][0]):
        retrieved_chunks.append({
            "text": text,
            "metadata": results["metadatas"][0][i]
        })


    baseline_pages = [
        chunk["metadata"]["page"]
        for chunk in retrieved_chunks
    ]


    # --------------------------------------------------
    # 3. Baseline MRR
    # --------------------------------------------------

    baseline_mrr = reciprocal_rank(
        expected_pages,
        baseline_pages
    )


    # --------------------------------------------------
    # 4. Rerank
    # --------------------------------------------------

    if retrieved_chunks:

        reranked_results = reranker.rerank(
            question,
            retrieved_chunks,
            top_k=5
        )

        reranked_chunks = [
            result["chunk"]
            for result in reranked_results
        ]

    else:
        reranked_chunks = []


    reranked_pages = [
        chunk["metadata"]["page"]
        for chunk in reranked_chunks
    ]


    # --------------------------------------------------
    # 5. Reranked MRR
    # --------------------------------------------------

    reranked_mrr = reciprocal_rank(
        expected_pages,
        reranked_pages
    )


    baseline_scores.append(baseline_mrr)
    reranked_scores.append(reranked_mrr)


    # --------------------------------------------------
    # 6. Print results
    # --------------------------------------------------

    print("\nExpected pages:")
    print(expected_pages)

    print("\nBaseline retrieval:")
    print(baseline_pages)

    print(
        f"Baseline MRR: {baseline_mrr:.3f}"
    )

    print("\nAfter reranking:")
    print(reranked_pages)

    print(
        f"Reranked MRR: {reranked_mrr:.3f}"
    )


# ------------------------------------------------------
# Final metrics
# ------------------------------------------------------

baseline_average = (
    sum(baseline_scores) / len(baseline_scores)
)

reranked_average = (
    sum(reranked_scores) / len(reranked_scores)
)


print("\n")
print("=" * 70)
print("RERANKING EVALUATION")
print("=" * 70)

print(
    f"Baseline MRR:  {baseline_average:.3f}"
)

print(
    f"Reranked MRR:  {reranked_average:.3f}"
)

improvement = (
    reranked_average - baseline_average
)

print(
    f"Improvement:   {improvement:+.3f}"
)