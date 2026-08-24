def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        [
            (
                f"Source: {chunk['metadata']['source']}\n"
                f"Page: {chunk['metadata']['page']}\n"
                f"Content:\n{chunk['text']}"
            )
            for chunk in retrieved_chunks
        ]
    )

    return f"""
You are DocMind, an evidence-grounded document assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer,
say that you could not find sufficient evidence in the documents.

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""