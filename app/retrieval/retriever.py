import numpy as np


class Retriever:
    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query_embedding, top_k=3):
        scores = []

        for chunk, embedding in zip(self.chunks, self.embeddings):
            score = self._cosine_similarity(query_embedding, embedding)

            scores.append({
                "chunk": chunk,
                "score": float(score)
            })

        scores.sort(key=lambda x: x["score"], reverse=True)

        return scores[:top_k]

    @staticmethod
    def _cosine_similarity(vector_a, vector_b):
        vector_a = np.array(vector_a)
        vector_b = np.array(vector_b)

        return np.dot(vector_a, vector_b) / (
            np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        )