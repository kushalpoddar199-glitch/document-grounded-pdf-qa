"""
Streamlit UI for the document-grounded PDF Q&A system.

Pipeline: upload PDF -> ingest/chunk -> embed into FAISS -> retrieve top-k
on each question -> generate a grounded, cited answer with Gemini.
"""

import os
import tempfile

import streamlit as st

from src.ingest import IngestConfig, ingest_pdf
from src.vectorstore import build_vectorstore, retrieve
from src.qa_chain import answer_question

st.set_page_config(page_title="Document-Grounded PDF Q&A", page_icon="📄")
st.title("📄 Document-Grounded PDF Q&A")
st.caption(
    "Ask questions about a single PDF. Answers are generated strictly from "
    "the document's own content, with page citations, and the assistant "
    "will say so explicitly if the answer isn't in the document."
)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Get a key from https://aistudio.google.com/apikey. "
        "Not stored anywhere -- only held for this session.",
    )
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)
    top_k = st.slider("Chunks retrieved per question", 1, 10, 4)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "vs" not in st.session_state:
    st.session_state.vs = None
    st.session_state.pdf_name = None

if uploaded_file is not None and api_key:
    if st.session_state.pdf_name != uploaded_file.name:
        with st.spinner("Ingesting and embedding document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            config = IngestConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = ingest_pdf(tmp_path, config)
            vs = build_vectorstore(chunks, api_key, pdf_path=tmp_path)

            st.session_state.vs = vs
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.num_chunks = len(chunks)

        st.success(
            f"Indexed '{uploaded_file.name}' into "
            f"{st.session_state.num_chunks} chunks."
        )

if st.session_state.vs is not None:
    question = st.text_input("Ask a question about the document")
    if question:
        with st.spinner("Retrieving and generating answer..."):
            retrieved = retrieve(st.session_state.vs, question, k=top_k)
            result = answer_question(question, retrieved, api_key)

        if result.grounded:
            st.markdown("### Answer")
        else:
            st.markdown("### Answer (not found in document)")
        st.write(result.answer)

        with st.expander("Retrieved source chunks"):
            for c in result.sources:
                page = c.metadata.get("page", "?")
                st.markdown(f"**Page {page}**")
                st.text(c.page_content[:500])
                st.divider()
elif not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started.")
else:
    st.info("Upload a PDF to get started.")
