from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages


pdf_path = "D:\project\DocMind\Sample.pdf"

pages = load_pdf(pdf_path)

chunks = chunk_pages(pages)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")

for chunk in chunks[:3]:
    print("\n--- CHUNK ---")
    print("Page:", chunk["metadata"]["page"])
    print(chunk["text"][:200])