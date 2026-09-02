from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import VectorStore
from app.retrieval.reranker import Reranker
from app.generation.llm import LLM
from app.generation.prompt import build_prompt


class RAGService:

    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.reranker = Reranker()
        self.llm = LLM()

    def query(self, question: str, document_id: str):

        # 1. Embed question
        query_embedding = self.embedder.embed(
            [question]
        )[0]

        # 2. Retrieve ONLY from selected document
        results = self.store.search(
            query_embedding,
            top_k=5,
            document_id=document_id
        )

        retrieved_chunks = []

        if results["documents"] and results["documents"][0]:

            for i, text in enumerate(
                results["documents"][0]
            ):
                retrieved_chunks.append({
                    "text": text,
                    "metadata": results["metadatas"][0][i]
                })

        # 3. Rerank
        if retrieved_chunks:

            reranked_results = self.reranker.rerank(
                question,
                retrieved_chunks,
                top_k=5
            )

            retrieved_chunks = [
                result["chunk"]
                for result in reranked_results
            ]

        # 4. Build grounded prompt
        prompt = build_prompt(
            question,
            retrieved_chunks
        )

        # 5. Generate answer
        answer = self.llm.generate(prompt)

        # 6. Build unique sources
        sources = []
        seen = set()

        for chunk in retrieved_chunks:

            metadata = chunk["metadata"]

            source = (
                metadata.get("source"),
                metadata.get("page")
            )

            if source in seen:
                continue

            seen.add(source)

            sources.append({
                "source": metadata.get("source"),
                "page": metadata.get("page")
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }