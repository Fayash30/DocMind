from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages
from embeddings.embedder import Embedder
from vectorstore.chroma_store import VectorStore
from generation.llm import LLM
from generation.prompt import build_prompt


pdf_path = "data/docmind_rag_survey.pdf"


# 1. Load PDF
pages = load_pdf(pdf_path)

print(f"Loaded {len(pages)} pages")


# 2. Chunk
chunks = chunk_pages(pages)

print(f"Created {len(chunks)} chunks")


# 3. Create embeddings
embedder = Embedder()

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

chunk_embeddings = embedder.embed(chunk_texts)

print("Created embeddings")


# 4. Store in Chroma
store = VectorStore()

store.add_chunks(
    chunks,
    chunk_embeddings
)

print("Stored chunks in vector database")


# 5. Ask a question
question = "What are the main components of a Retrieval-Augmented Generation system?"


# 6. Embed question
query_embedding = embedder.embed([question])[0]


# 7. Retrieve relevant chunks
results = store.search(
    query_embedding,
    top_k=5
)


# 8. Convert Chroma results into our chunk format
retrieved_chunks = []

for i, text in enumerate(results["documents"][0]):
    retrieved_chunks.append({
        "text": text,
        "metadata": results["metadatas"][0][i]
    })


# 9. Build grounded prompt
prompt = build_prompt(
    question,
    retrieved_chunks
)


# 10. Ask Gemini
llm = LLM()

answer = llm.generate(prompt)


# 11. Display answer
print("\n==============================")
print("           DOCMIND")
print("==============================")

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)

print("\nSOURCES:")

for chunk in retrieved_chunks:
    print(
        f"- {chunk['metadata']['source']}, "
        f"page {chunk['metadata']['page']}"
    )