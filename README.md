# Enterprise Hybrid RAG Engine

Pipeline: **Upload → Chunk → ChromaDB (dense) + BM25 (keyword) → RRF fusion → CrossEncoder rerank → Groq LLM → Answer + citations**

## Run (Docker)
```bash
cp .env.example .env      # GROQ_API_KEY daalein (https://console.groq.com)
docker compose up --build
```
- UI:  http://localhost:8501
- API docs: http://localhost:8000/docs

## Run (local)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload            # terminal 1
streamlit run ui/streamlit_app.py        # terminal 2
```

## API
| Method | Endpoint | Kaam |
|---|---|---|
| POST | /ingest | file upload + index |
| POST | /query | `{"question": "...", "top_n": 4}` |
| GET | /documents | indexed documents |
| DELETE | /documents/{name} | document hatao |
| GET | /health | status |
