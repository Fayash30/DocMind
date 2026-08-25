from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages
from ingestion.document import generate_document_id
from embeddings.embedder import Embedder
from vectorstore.chroma_store import VectorStore
from generation.llm import LLM
from generation.prompt import build_prompt
from retrieval.reranker import Reranker


pdf_path = "data/docmind_rag_survey.pdf"


# 1. Generate document ID
document_id = generate_document_id(pdf_path)

print("Document ID:", document_id)


# 2. Connect to vector store
store = VectorStore()


# 3. Check whether document already exists
if store.document_exists(document_id):
    print("Document already indexed. Skipping ingestion.")

else:
    # 4. Load PDF
    pages = load_pdf(pdf_path)

    print(f"Loaded {len(pages)} pages")

    # 5. Chunk
    chunks = chunk_pages(
        pages,
        document_id
    )

    print(f"Created {len(chunks)} chunks")

    # 6. Generate embeddings
    embedder = Embedder()

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    chunk_embeddings = embedder.embed(chunk_texts)

    print("Created embeddings")

    # 7. Store
    store.add_chunks(
        chunks,
        chunk_embeddings
    )

    print("Stored chunks in vector database")


# 8. Create embedder for the user query
embedder = Embedder()


# 9. Ask a question
# question = (
#     "What are the main components of a "
#     "Retrieval-Augmented Generation system?"
# )
# question = "How do I make chicken biryani?"

question = "What are some limitations of Retrieval-Augmented Generation?"

# 10. Embed the question
query_embedding = embedder.embed([question])[0]


# 11. Retrieve relevant chunks
results = store.search(
    query_embedding,
    top_k=5
)

if not results["documents"][0]:
    print("\nDOCMIND:")
    print("I couldn't find sufficient evidence in the uploaded documents.")
    exit()

print("\nRETRIEVAL RESULTS:")

for i, distance in enumerate(results["distances"][0]):
    print(
        f"{i + 1}. distance={distance:.4f} "
        f"| page={results['metadatas'][0][i]['page']}"
    )

# 12. Convert Chroma results into our chunk format
retrieved_chunks = []

for i, text in enumerate(results["documents"][0]):
    retrieved_chunks.append({
        "text": text,
        "metadata": results["metadatas"][0][i]
    })


reranker = Reranker()

reranked_results = reranker.rerank(
    question,
    retrieved_chunks,
    top_k=3
)

retrieved_chunks = [
    result["chunk"]
    for result in reranked_results
]


print("\nRERANKED RESULTS:")

for result in reranked_results:
    print(
        f"score={result['score']:.4f} "
        f"| page={result['chunk']['metadata']['page']}"
    )

    
# 13. Build grounded prompt
prompt = build_prompt(
    question,
    retrieved_chunks
)


# 14. Ask Gemini
llm = LLM()

answer = llm.generate(prompt)


# 15. Display answer
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