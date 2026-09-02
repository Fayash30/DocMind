from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import shutil

from app.services.rag_service import RAGService
from app.services.ingestion_service import IngestionService


app = FastAPI(
    title="DocMind API",
    description="Document-grounded question answering API",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    document_id: str
    question: str


rag_service = RAGService()
ingestion_service = IngestionService()


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


@app.get("/")
def root():
    return {
        "message": "DocMind API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/query")
def query(request: QueryRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not request.document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Document ID cannot be empty."
        )

    return rag_service.query(
        question,
        request.document_id
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        result = ingestion_service.ingest(
            str(file_path)
        )

        return {
            "filename": file.filename,
            **result
        }

    except Exception as e:

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(e)}"
        )

@app.get("/documents")
def get_documents():
    return ingestion_service.store.get_documents()

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):

    if not ingestion_service.store.document_exists(
        document_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    ingestion_service.store.delete_document(
        document_id
    )

    return {
        "document_id": document_id,
        "status": "deleted"
    }