import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ---- LLM / models ----
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ---- Storage ----
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "data/chroma")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    COLLECTION: str = "enterprise_docs"

    # ---- Chunking ----
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 700))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

    # ---- Retrieval ----
    DENSE_K: int = int(os.getenv("DENSE_K", 10))          # chunks from ChromaDB
    BM25_K: int = int(os.getenv("BM25_K", 10))            # chunks from BM25
    RRF_K: int = 60                                        # Reciprocal Rank Fusion constant
    RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", 15))
    FINAL_TOP_N: int = int(os.getenv("FINAL_TOP_N", 4))   # final chunks sent to the LLM
    # Chunks scoring below this are dropped (so off-topic questions get a clean "not found")
    MIN_RERANK_SCORE: float = float(os.getenv("MIN_RERANK_SCORE", -2.0))

    # ---- Chat memory ----
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", 3))

    # ---- Security / limits ----
    API_KEY: str = os.getenv("API_KEY", "")               # empty = auth disabled
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", 25))


settings = Settings()
os.makedirs(settings.CHROMA_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)