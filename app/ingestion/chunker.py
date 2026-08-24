def chunk_pages(pages, document_id, chunk_size=1000, overlap=200):
    chunks = []

    chunk_index = 0

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "id": f"{document_id}_chunk_{chunk_index}",
                "text": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "page": page_number,
                    "source": page["source"]
                }
            })

            chunk_index += 1
            start += chunk_size - overlap

    return chunks