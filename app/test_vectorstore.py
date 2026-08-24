from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages
from embeddings.embedder import Embedder
from vectorstore.chroma_store import VectorStore


pdf_path = "data/docmind_rag_survey.pdf"

pages = load_pdf(pdf_path)
chunks = chunk_pages(pages)

embedder = Embedder()

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

chunk_embeddings = embedder.embed(chunk_texts)

store = VectorStore()

store.add_chunks(
    chunks,
    chunk_embeddings
)

query = "What is this document mainly about?"

query_embedding = embedder.embed([query])[0]

results = store.search(
    query_embedding,
    top_k=3
)

for i, document in enumerate(results["documents"][0]):
    print(f"\n--- RESULT {i + 1} ---")
    print(document[:300])
    print("Metadata:", results["metadatas"][0][i])