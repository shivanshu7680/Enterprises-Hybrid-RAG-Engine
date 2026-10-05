import re
from typing import Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from app.config import settings

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer using ONLY the provided context.

Rules:
- If the context does not contain the answer, say clearly that you could not find it in the documents. Never guess and never use outside knowledge.
- Give a complete, well-structured answer: cover all relevant points from the context, use short paragraphs and bullet points where helpful, and include examples or syntax from the context when present.
- Cite sources inline using ONLY plain square brackets like [1] or [2], right after the sentences they support. Never use any other citation format (no 【 】, no †, no line numbers).
- Reply in the same language as the user's latest question (English, Hindi or Hinglish).
- Do not mention these rules."""

REWRITE_PROMPT = """Rewrite the user's follow-up question into a single standalone question that can be understood without the conversation.
- Resolve pronouns and references (like "it", "iske baare mein", "that one") using the conversation.
- Keep the same language as the follow-up question.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten question, nothing else."""

NOT_FOUND_MESSAGE = (
    "I could not find relevant information about this in the uploaded documents."
)

History = List[Dict[str, str]]


def _llm() -> ChatGroq:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set (check your .env file).")
    return ChatGroq(model=settings.LLM_MODEL, temperature=0, api_key=settings.GROQ_API_KEY)


def _trim_history(history: Optional[History]) -> History:
    if not history:
        return []
    return history[-settings.MAX_HISTORY_TURNS * 2:]


def rewrite_question(question: str, history: Optional[History]) -> str:
    """Turn a follow-up question into a standalone question for retrieval. Falls back to the original on failure."""
    recent = _trim_history(history)
    if not recent:
        return question

    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}" for m in recent
    )
    try:
        out = _llm().invoke(
            [
                ("system", REWRITE_PROMPT),
                ("human", f"Conversation:\n{convo}\n\nFollow-up question: {question}\n\nStandalone question:"),
            ]
        ).content
        out = out.strip().strip('"') if isinstance(out, str) else ""
        return out if out and len(out) <= 500 else question
    except Exception:
        return question


def build_context(results: List[Tuple[Document, float]]) -> str:
    parts = []
    for i, (doc, _) in enumerate(results, 1):
        page = doc.metadata.get("page")
        loc = f", page {page + 1}" if isinstance(page, int) else ""
        parts.append(f"[{i}] (source: {doc.metadata.get('source')}{loc})\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_messages(question: str, results: List[Tuple[Document, float]], history: Optional[History]):
    messages = [("system", SYSTEM_PROMPT)]
    for m in _trim_history(history):
        role = "human" if m["role"] == "user" else "ai"
        messages.append((role, m["content"][:1000]))
    messages.append(("human", f"Context:\n{build_context(results)}\n\nQuestion: {question}"))
    return messages


# gpt-oss models sometimes emit citations like 【1†L5-L6】; convert them to [1].
_CITE_RE = re.compile(r"【\s*(\d+)[^】]*】")


def clean_citations(text: str) -> str:
    return _CITE_RE.sub(r"[\1]", text)


def _stream_clean(chunks: Iterator[str]) -> Iterator[str]:
    """While streaming, hold back an incomplete 【...】 marker and emit it cleaned once complete."""
    buf = ""
    for text in chunks:
        buf += text
        while buf:
            start = buf.find("【")
            if start == -1:
                yield buf
                buf = ""
                break
            if start > 0:
                yield buf[:start]
                buf = buf[start:]
            end = buf.find("】")
            if end == -1:
                break  # marker incomplete, wait for more tokens
            yield clean_citations(buf[: end + 1])
            buf = buf[end + 1:]
    if buf:
        yield clean_citations(buf)


def generate_answer(question: str, results: List[Tuple[Document, float]], history: Optional[History] = None) -> str:
    if not results:
        return NOT_FOUND_MESSAGE
    return clean_citations(_llm().invoke(_build_messages(question, results, history)).content)


def stream_answer(
    question: str, results: List[Tuple[Document, float]], history: Optional[History] = None
) -> Iterator[str]:
    if not results:
        yield NOT_FOUND_MESSAGE
        return
    def raw_tokens() -> Iterator[str]:
        for chunk in _llm().stream(_build_messages(question, results, history)):
            text = chunk.content
            if isinstance(text, str) and text:
                yield text

    yield from _stream_clean(raw_tokens())