"""
Embedding + retrieval stage: turn chunks into a searchable FAISS index.

Why FAISS over a hosted vector DB here: this app is single-document,
single-session by design (see PROBLEM_STATEMENT in the README) -- an
in-memory / on-disk FAISS index is the right amount of infrastructure.
Swapping in Chroma or a hosted store later only touches this file.
"""

from __future__ import annotations

import hashlib
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "models/gemini-embedding-001"
INDEX_CACHE_DIR = ".faiss_cache"


def _doc_fingerprint(pdf_path: str) -> str:
    """Hash file contents so the same PDF re-uses a cached index."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()[:16]


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


def build_vectorstore(
    chunks: list[Document],
    api_key: str,
    pdf_path: str | None = None,
    use_cache: bool = True,
) -> VectorStore:
    """
    Embed chunks and build (or load a cached) FAISS index.

    Caching matters in practice: re-embedding a 100-page PDF on every
    Streamlit rerun is slow and burns API quota for no reason.
    """
    embeddings = get_embeddings(api_key)

    if use_cache and pdf_path:
        os.makedirs(INDEX_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(INDEX_CACHE_DIR, _doc_fingerprint(pdf_path))
        if os.path.exists(cache_path):
            return FAISS.load_local(
                cache_path, embeddings, allow_dangerous_deserialization=True
            )
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(cache_path)
        return vs

    return FAISS.from_documents(chunks, embeddings)


def retrieve(vs: VectorStore, query: str, k: int = 4) -> list[Document]:
    return vs.similarity_search(query, k=k)
