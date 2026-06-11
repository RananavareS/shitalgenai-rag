# ShitalGenAI — RAG Edition

A **Retrieval-Augmented Generation (RAG)** chatbot built on top of the original ShitalGenAI codebot. Upload your documents and ask questions — the AI answers using content from your files.

## What's new vs the original CodeBot

| Feature | CodeBot | RAG Edition |
|---------|---------|-------------|
| Purpose | Code generation / debugging | Document Q&A |
| Document upload | ❌ | ✅ PDF, DOCX, TXT, MD, code files |
| Chunking & indexing | ❌ | ✅ TF-IDF vector store |
| Context retrieval | ❌ | ✅ Top-K semantic search |
| Source citations | ❌ | ✅ Shown per answer |
| Chat modes | Generate / Debug / Refactor… | Ask / Summarize / Extract / Compare / Explain |

## How it works

1. **Upload** — Files are sent to `/api/upload`, parsed, split into ~500-word overlapping chunks, and TF-IDF-embedded in memory.
2. **Query** — When you send a message, the server retrieves the top-5 most relevant chunks using cosine similarity.
3. **Augment** — Relevant chunks are injected into the system prompt as context.
4. **Generate** — Groq LLM (Llama 3.3 70B by default) answers using that context.

## Supported file types

- **PDF** (`.pdf`) — text-layer extraction via PyPDF2
- **Word** (`.docx`) — via python-docx
- **Plain text** (`.txt`, `.md`, `.csv`, `.json`)
- **Code files** (`.py`, `.js`, `.ts`, `.html`, `.css`, etc.)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key in .env
#    GROQ_API_KEY=gsk_...

# 3. Run
python server.py

# 4. Open http://localhost:8080
```

## Project structure

```
codebot_rag/
├── server.py        ← Python HTTP server with RAG endpoints
├── index.html       ← Main UI
├── rag.js           ← Frontend logic (upload, chat, history)
├── rag.css          ← RAG-specific styles
├── style.css        ← Original base styles
├── logo.png         ← Logo
├── requirements.txt ← Python deps
└── .env             ← API keys (not committed)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload a document (JSON: `{filename, data: base64}`) |
| `GET`  | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/:id` | Remove a document and its chunks |
| `POST` | `/api/chat` | Chat with optional RAG context (`{model, messages, rag: true}`) |

## Notes

- Document chunks are stored **in memory** — they reset when you restart the server.  
  For persistence, replace the in-memory `CHUNKS` list with SQLite or a vector DB (ChromaDB, Qdrant, FAISS).
- The TF-IDF embedder is lightweight and dependency-free. For better retrieval quality, swap in `sentence-transformers` (already in `requirements.txt`).
