import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="🔎", layout="wide")
st.title("🔎 Enterprise Hybrid RAG Engine")
st.caption("ChromaDB + BM25 → RRF fusion → CrossEncoder rerank → Groq LLM")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("flash", [])


# ---------------------------------------------------------------- helpers
def error_detail(r: requests.Response) -> str:
    try:
        data = r.json()
        return data.get("detail", r.text) if isinstance(data, dict) else r.text
    except ValueError:
        return r.text


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            page = f", page {s['page'] + 1}" if isinstance(s.get("page"), int) else ""
            st.markdown(f"**[{i}] {s['source']}{page}** — rerank score `{s['score']}`")
            st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))


def show_search_query(standalone, question):
    if standalone and standalone.strip().lower() != question.strip().lower():
        st.caption(f"🔎 Search query: {standalone}")


def stream_answer(question, history, top_n, holder):
    """Yield tokens from the backend /query/stream endpoint; keep sources in the holder."""
    payload = {"question": question, "top_n": top_n, "history": history}
    with requests.post(
        f"{API_URL}/query/stream", json=payload, headers=HEADERS, stream=True, timeout=300
    ) as r:
        if not r.ok:
            raise RuntimeError(error_detail(r))
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "sources":
                holder["sources"] = event["sources"]
                holder["standalone"] = event.get("standalone_question")
            elif kind == "token":
                yield event["text"]
            elif kind == "error":
                raise RuntimeError(event["detail"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📁 Documents")

    for kind, text in st.session_state.flash:
        (st.success if kind == "ok" else st.error)(text)
    st.session_state.flash = []

    uploads = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if st.button("Index documents", disabled=not uploads, use_container_width=True):
        flash = []
        for f in uploads:
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    r = requests.post(
                        f"{API_URL}/ingest",
                        files={"file": (f.name, f.getvalue())},
                        headers=HEADERS,
                        timeout=600,
                    )
                except requests.RequestException as e:
                    flash.append(("err", f"{f.name}: could not connect to the API ({e})"))
                    continue
            if r.ok:
                flash.append(("ok", f"{f.name}: {r.json()['chunks_indexed']} chunks indexed"))
            else:
                flash.append(("err", f"{f.name}: {error_detail(r)}"))
        st.session_state.flash = flash
        st.session_state.uploader_key += 1  # resets the uploader
        st.rerun()

    st.divider()
    try:
        resp = requests.get(f"{API_URL}/documents", headers=HEADERS, timeout=10)
        if resp.status_code == 401:
            st.error("API key is wrong or missing (check API_KEY in .env).")
        elif resp.ok:
            docs = resp.json()
            if not docs:
                st.info("No documents yet.")
            for name, n in docs.items():
                c1, c2 = st.columns([4, 1])
                c1.write(f"📄 {name} ({n})")
                if c2.button("🗑", key=f"del-{name}"):
                    requests.delete(f"{API_URL}/documents/{name}", headers=HEADERS, timeout=30)
                    st.rerun()
        else:
            st.error(error_detail(resp))
    except requests.RequestException:
        st.error("Cannot connect to the API.")

    st.divider()
    top_n = st.slider("Final chunks (top N)", 1, 10, 4)
    use_memory = st.checkbox("Chat memory (follow-up questions)", value=True)
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_search_query(m.get("standalone"), m.get("question", ""))
            show_sources(m.get("sources"))

if question := st.chat_input("Ask anything about your documents..."):
    history = (
        [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
        if use_memory
        else []
    )
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        holder = {"sources": [], "standalone": None}
        try:
            with st.spinner("Searching documents..."):
                answer = st.write_stream(stream_answer(question, history, top_n, holder))
        except Exception as e:
            st.error(f"Error: {e}")
        else:
            show_search_query(holder["standalone"], question)
            show_sources(holder["sources"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": holder["sources"],
                    "standalone": holder["standalone"],
                    "question": question,
                }
            )