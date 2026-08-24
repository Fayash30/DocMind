from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages
from embeddings.embedder import Embedder
from retrieval.retriever import Retriever


pdf_path = "data/docmind_rag_survey.pdf"

pages = load_pdf(pdf_path)
chunks = chunk_pages(pages)

embedder = Embedder()

chunk_texts = [chunk["text"] for chunk in chunks]
chunk_embeddings = embedder.embed(chunk_texts)

retriever = Retriever(chunks, chunk_embeddings)

query = "What is this document mainly about?"

query_embedding = embedder.embed([query])[0]

results = retriever.search(query_embedding, top_k=3)

for result in results:
    print("\n--- RESULT ---")
    print("Score:", result["score"])
    print("Source:", result["chunk"]["metadata"]["source"])
    print("Page:", result["chunk"]["metadata"]["page"])
    print("Text:", result["chunk"]["text"][:300])