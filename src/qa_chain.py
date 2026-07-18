"""
Generation stage: answer strictly from retrieved chunks.

The core RAG failure mode this guards against is the model quietly
falling back on parametric knowledge when retrieval comes up thin.
The prompt forces two disciplines:
  1. Cite the page(s) the answer came from.
  2. Explicitly say the document doesn't contain the answer, rather than
     guessing -- this is the single highest-value line in the whole app.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

GENERATION_MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = """You are a document-grounded question answering assistant.

Rules you must follow:
1. Answer ONLY using the CONTEXT provided below. Do not use outside knowledge,
   even if you are confident it is correct.
2. If the context does not contain enough information to answer, respond
   exactly with: "The document does not contain information to answer this
   question." Do not guess or partially answer.
3. When you do answer, cite the page number(s) your answer relies on, in the
   form (p. X), immediately after the relevant sentence.
4. Be concise. Do not repeat the question back to the user.

CONTEXT:
{context}
"""


@dataclass
class QAResult:
    answer: str
    sources: list[Document]
    grounded: bool  # False if the model reported no answer in the document


def _format_context(chunks: list[Document]) -> str:
    parts = []
    for c in chunks:
        page = c.metadata.get("page", "?")
        parts.append(f"[page {page}]\n{c.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    question: str, chunks: list[Document], api_key: str
) -> QAResult:
    llm = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL, google_api_key=api_key, temperature=0
    )
    context = _format_context(chunks)
    prompt = SYSTEM_PROMPT.format(context=context)

    response = llm.invoke(
        [
            ("system", prompt),
            ("human", question),
        ]
    )
    text = _extract_text(response.content).strip()
    grounded = "does not contain information" not in text.lower()
    return QAResult(answer=text, sources=chunks, grounded=grounded)
def _extract_text(content) -> str:
    """
    Normalize response.content to a plain string.

    Newer Gemini models (3.x) can return content as a list of blocks
    (e.g. [{"type": "text", "text": "..."}]) instead of a plain string
    the way 1.x/2.x models did. Handle both shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)