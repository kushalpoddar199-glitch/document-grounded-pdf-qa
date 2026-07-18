# Document-Grounded PDF Q&A

A single-document Retrieval-Augmented Generation (RAG) system. Upload a PDF,
ask questions in natural language, and get answers generated **strictly**
from the document's own content — with page citations, and an explicit
refusal when the answer isn't in the document.

Built to explore the core RAG pipeline end to end: ingestion → chunking →
embedding → retrieval → grounded generation.

## Why this exists

Generic LLMs hallucinate when asked about private documents they've never
seen. This project constrains generation to only what's retrievable from the
uploaded PDF, so the assistant can say "I don't know" instead of guessing.

## Architecture

```
PDF upload
   │
   ▼
[ingest.py]        load pages, split into overlapping chunks (with page metadata)
   │
   ▼
[vectorstore.py]    embed chunks (Gemini embeddings) → FAISS index (cached by file hash)
   │
   ▼
 user question
   │
   ▼
[vectorstore.py]    similarity search → top-k relevant chunks
   │
   ▼
[qa_chain.py]       Gemini generates an answer constrained to the retrieved
                     context, with page citations, or an explicit "not found"
   │
   ▼
[app.py]            Streamlit UI ties it together
```

## Stack

Python · Gemini API · LangChain · FAISS · Streamlit

## Setup

```bash
git clone <your-repo-url>
cd pdf-rag-qa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
Either paste it into the app's sidebar at runtime, or copy `.env.example` to
`.env` and set `GOOGLE_API_KEY` there.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, upload a PDF, and start asking
questions.

## Design decisions worth calling out

- **Chunk size / overlap are tunable in the UI**, not hardcoded — this is
  the single biggest lever on answer quality, and it's worth seeing the
  effect directly rather than trusting a default.
- **Page-level metadata survives chunking**, so every answer can cite a
  page number instead of a vague "somewhere in this document."
- **Refusal is a first-class output**, not an afterthought: the generation
  prompt explicitly instructs the model to say the document doesn't contain
  the answer rather than fall back on its own parametric knowledge. This is
  the difference between a RAG demo and a RAG system you can trust.
- **FAISS index is cached by file hash** on disk, so re-asking questions
  against the same PDF doesn't re-embed (and re-bill) every rerun.

## Known limitations / next steps

- Single document, single session — no multi-doc corpus or persistent
  library of PDFs (would swap FAISS for Chroma + a doc-id namespace).
- No re-ranking step after retrieval — a cross-encoder re-ranker would
  likely improve precision on longer, denser PDFs.
- No evaluation harness yet — a small labeled Q&A set per test document
  would make chunk-size/overlap tuning empirical instead of by feel.

## License

MIT
