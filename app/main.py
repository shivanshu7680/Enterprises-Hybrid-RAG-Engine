import json
import logging
import secrets
import shutil
import traceback
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.config import settings
from app.generator import generate_answer, rewrite_question, stream_answer
from app.ingestion import SUPPORTED, load_and_split
from app.retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")


# ---------------------------------------------------------------- setup
@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_retriever()  # load models at startup
    yield


app = FastAPI(title="Enterprise Hybrid RAG Engine", version="2.0.0", lifespan=lifespan)

# ---------------------------------------------------------------- auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str | None = Security(api_key_header)):
    """If settings.API_KEY is set, every protected endpoint requires the X-API-Key header."""
    if not settings.API_KEY:
        return
    if not key or not secrets.compare_digest(key.encode(), settings.API_KEY.encode()):
        raise HTTPException(401, "Invalid or missing API key.")


auth = [Depends(verify_api_key)]


# ---------------------------------------------------------------- schemas
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str
    top_n: int = Field(default=settings.FINAL_TOP_N, ge=1, le=20)
    history: List[Message] = Field(default_factory=list)


# ---------------------------------------------------------------- helpers
def _prepare(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty.")
    history = [m.model_dump() for m in req.history]
    try:
        standalone = rewrite_question(req.question, history)
        results = get_retriever().search(standalone, req.top_n)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Retrieval failed: {type(e).__name__}: {e}")
    return standalone, results, history


def _sources(results):
    return [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "score": round(s, 4),
            "text": d.page_content,
        }
        for d, s in results
    ]


def _line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_model": settings.LLM_MODEL,
        "groq_key_set": bool(settings.GROQ_API_KEY),
        "auth_enabled": bool(settings.API_KEY),
    }


@app.post("/ingest", dependencies=auth)
def ingest(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name or not name.lower().endswith(SUPPORTED):
        raise HTTPException(400, f"Supported file types: {SUPPORTED}")
    if file.size is not None and file.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.MAX_UPLOAD_MB} MB.")

    save_path = Path(settings.UPLOAD_DIR) / name
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = load_and_split(str(save_path), name)
        count = get_retriever().add_documents(chunks)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Ingestion failed: {type(e).__name__}: {e}")
    return {"file": name, "chunks_indexed": count}


@app.post("/query", dependencies=auth)
def query(req: QueryRequest):
    standalone, results, history = _prepare(req)
    try:
        answer = generate_answer(req.question, results, history)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"LLM error: {type(e).__name__}: {e}")
    return {"answer": answer, "standalone_question": standalone, "sources": _sources(results)}


@app.post("/query/stream", dependencies=auth)
def query_stream(req: QueryRequest):
    """NDJSON stream: first {"type":"sources"}, then {"type":"token"}..., finally {"type":"done"}."""
    standalone, results, history = _prepare(req)

    def event_stream():
        yield _line({"type": "sources", "standalone_question": standalone, "sources": _sources(results)})
        try:
            for token in stream_answer(req.question, results, history):
                yield _line({"type": "token", "text": token})
            yield _line({"type": "done"})
        except Exception as e:
            traceback.print_exc()
            yield _line({"type": "error", "detail": f"LLM error: {type(e).__name__}: {e}"})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/documents", dependencies=auth)
def documents():
    return get_retriever().list_sources()


@app.delete("/documents/{source}", dependencies=auth)
def delete_document(source: str):
    get_retriever().delete_source(source)
    (Path(settings.UPLOAD_DIR) / Path(source).name).unlink(missing_ok=True)
    return {"deleted": source}