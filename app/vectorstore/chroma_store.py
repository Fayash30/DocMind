import chromadb


class VectorStore:
    def __init__(self, persist_directory="data/chroma"):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="docmind"
        )

    def add_chunks(self, chunks, embeddings):
        ids = [
            chunk["id"]
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            chunk["metadata"]
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k=3, max_distance=1.0):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        filtered_results = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            if distance <= max_distance:
                filtered_results["documents"][0].append(document)
                filtered_results["metadatas"][0].append(metadata)
                filtered_results["distances"][0].append(distance)

        return filtered_results


    def document_exists(self, document_id):
        results = self.collection.get(
            where={
                "document_id": document_id
            },
            limit=1
        )

        return len(results["ids"]) > 0