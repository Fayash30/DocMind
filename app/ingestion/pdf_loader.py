from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path):
    reader = PdfReader(file_path)

    source = Path(file_path).name
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text,
            "source": source
        })

    return pages