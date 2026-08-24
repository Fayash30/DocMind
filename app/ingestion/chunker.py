def chunk_pages(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "page": page_number,
                    "source": page["source"]
                }
            })

            start += chunk_size - overlap

    return chunks