"""
Ingestion stage: load a PDF and split it into overlapping, retrievable chunks.

Design notes
------------
- We keep page metadata on every chunk so answers can cite a page number,
  not just "somewhere in the document".
- chunk_size / chunk_overlap are exposed as parameters (not hardcoded)
  because they materially affect retrieval quality -- too small and you
  lose context across a chunk boundary, too large and irrelevant text
  dilutes the embedding and pushes the real answer out of the top-k.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@dataclass
class IngestConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200


def load_pdf(pdf_path: str) -> list[Document]:
    """Load a PDF into one LangChain Document per page."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def chunk_documents(
    docs: list[Document], config: IngestConfig | None = None
) -> list[Document]:
    """Split page-level documents into overlapping retrieval chunks."""
    config = config or IngestConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Tag each chunk with a stable id so we can reference it in citations.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def ingest_pdf(pdf_path: str, config: IngestConfig | None = None) -> list[Document]:
    """Full ingestion: load + chunk in one call."""
    pages = load_pdf(pdf_path)
    return chunk_documents(pages, config)
