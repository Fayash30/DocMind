from embeddings.embedder import Embedder


embedder = Embedder()

texts = [
    "Transformers use attention mechanisms.",
    "Attention helps models focus on relevant information.",
    "The company reported higher revenue this year."
]

vectors = embedder.embed(texts)

print("Number of vectors:", len(vectors))
print("Vector dimensions:", len(vectors[0]))
print("First vector:", vectors[0])