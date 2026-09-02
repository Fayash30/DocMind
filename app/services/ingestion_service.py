from pathlib import Path

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.document import generate_document_id
from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import VectorStore


class IngestionService:

    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()

    def ingest(self, file_path: str):

        # 1. Generate deterministic document ID
        document_id = generate_document_id(file_path)

        # 2. Skip if already indexed
        if self.store.document_exists(document_id):
            return {
                "document_id": document_id,
                "status": "already_indexed"
            }

        # 3. Load PDF
        pages = load_pdf(file_path)

        # 4. Chunk pages
        chunks = chunk_pages(
            pages,
            document_id
        )

        # 5. Generate embeddings
        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedder.embed(
            chunk_texts
        )

        # 6. Store
        self.store.add_chunks(
            chunks,
            embeddings
        )

        return {
            "document_id": document_id,
            "status": "indexed",
            "pages": len(pages),
            "chunks": len(chunks)
        }