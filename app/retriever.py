from collections import Counter
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from app.config import settings


class HybridRetriever:
    """Dense (ChromaDB) + Sparse (BM25) -> RRF fusion -> CrossEncoder rerank -> score filter."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vs = Chroma(
            collection_name=settings.COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DIR,
        )
        self.reranker = CrossEncoder(settings.RERANK_MODEL)
        self.bm25 = None
        self._rebuild_bm25()

    # ---------- index management ----------
    def _rebuild_bm25(self):
        data = self.vs.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        self.bm25 = BM25Retriever.from_documents(docs, k=settings.BM25_K) if docs else None

    def add_documents(self, chunks: List[Document]) -> int:
        if not chunks:
            return 0
        self.delete_source(chunks[0].metadata["source"], rebuild=False)  # re-upload = replace
        ids = [c.metadata["chunk_id"] for c in chunks]
        self.vs.add_documents(chunks, ids=ids)
        self._rebuild_bm25()
        return len(chunks)

    def delete_source(self, source: str, rebuild: bool = True):
        self.vs._collection.delete(where={"source": source})
        if rebuild:
            self._rebuild_bm25()

    def list_sources(self) -> dict:
        data = self.vs.get(include=["metadatas"])
        return dict(Counter(m["source"] for m in data["metadatas"]))

    # ---------- retrieval ----------
    @staticmethod
    def _rrf(result_lists: List[List[Document]], k: int) -> List[Document]:
        scores, docs = {}, {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                cid = doc.metadata["chunk_id"]
                docs[cid] = doc
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(scores, key=scores.get, reverse=True)
        return [docs[c] for c in ordered]

    def search(self, query: str, top_n: int | None = None) -> List[Tuple[Document, float]]:
        top_n = top_n or settings.FINAL_TOP_N

        dense = self.vs.similarity_search(query, k=settings.DENSE_K)
        sparse = self.bm25.invoke(query) if self.bm25 else []

        fused = self._rrf([dense, sparse], settings.RRF_K)[: settings.RERANK_CANDIDATES]
        if not fused:
            return []

        scores = self.reranker.predict([(query, d.page_content) for d in fused])
        ranked = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)[:top_n]
        # Drop irrelevant chunks: if nothing is relevant, return an empty list -> "not found"
        return [(d, float(s)) for d, s in ranked if s >= settings.MIN_RERANK_SCORE]