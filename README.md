# langchainrag

A **Retrieval-Augmented Generation (RAG)** pipeline that answers questions grounded in your own documents, using a fully free stack:

- 🔗 **LangChain** — orchestration framework
- 🤖 **Groq** — LLM inference (llama-3.3-70b-versatile, free)
- 🤗 **HuggingFace** — local embeddings (all-MiniLM-L6-v2, free)
- 🌲 **Pinecone** — managed vector database (free tier)

Based on the [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               INGESTION  (run once)                  │
│                                                     │
│  Web pages / docs                                   │
│       │                                             │
│       ▼                                             │
│  WebBaseLoader                                      │
│       │                                             │
│       ▼                                             │
│  RecursiveCharacterTextSplitter                     │
│  (chunk_size=1000, overlap=200)                     │
│       │                                             │
│       ▼                                             │
│  HuggingFaceEmbeddings          ┌──────────────┐   │
│  (all-MiniLM-L6-v2, 384-dim)   │   Pinecone   │   │
│       │                         │  Vector DB   │   │
│       └──────── upsert ────────►│  (cosine)    │   │
│                                 └──────────────┘   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│               QUERY  (at runtime)                    │
│                                                     │
│  User question                                      │
│       │                                             │
│       ├─────────────────────────────────┐           │
│       ▼                                 │           │
│  HuggingFaceEmbeddings                  │           │
│  (embed question)                       │           │
│       │                                 │           │
│       ▼                                 │           │
│  Pinecone retriever                     │           │
│  (top-4 similar chunks)                 │           │
│       │                                 │           │
│       ▼                                 ▼           │
│  format_docs()          question (passthrough)      │
│       │                      │                      │
│       └──────────┬───────────┘                      │
│                  ▼                                   │
│           ChatPromptTemplate                         │
│           (context + question)                       │
│                  │                                   │
│                  ▼                                   │
│           ChatGroq (llama-3.3-70b)                   │
│                  │                                   │
│                  ▼                                   │
│          StrOutputParser                             │
│                  │                                   │
│                  ▼                                   │
│        Answer + Sources                              │
└─────────────────────────────────────────────────────┘
```

### Key components

| Component | File | Purpose |
|---|---|---|
| `WebBaseLoader` | `ingest.py` | Fetches raw HTML from URLs |
| `RecursiveCharacterTextSplitter` | `ingest.py` | Splits docs into overlapping chunks |
| `HuggingFaceEmbeddings` | both | Converts text to 384-dim vectors locally |
| `PineconeVectorStore` | both | Stores and queries vectors |
| `ChatGroq` | `query.py` | Generates the final answer (free) |
| `RunnablePassthrough` | `query.py` | Passes the question unchanged through the chain |

---

## Prerequisites

- Python 3.10+
- [Groq API key](https://console.groq.com) — free, no credit card
- [Pinecone API key](https://app.pinecone.io) — free starter plan

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/langchainrag.git
cd langchainrag

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and fill in your API keys
```

---

## Usage

### Step 1 — Ingest documents (run once)

```bash
python ingest.py
```

```
📥 Loading documents...
   Loaded 2 document(s)
✂️  Splitting into chunks...
   2 doc(s) → 147 chunks
🔢 Loading embedding model...
🌲 Connecting to Pinecone...
   Creating index 'langchainrag'...
⬆️  Embedding and uploading chunks to Pinecone...
✅ Done! 147 chunks stored in index 'langchainrag'.
   You can now run:  python query.py
```

### Step 2 — Ask questions

```bash
# Interactive mode
python query.py

# Single question from CLI
python query.py "What is an LLM-powered autonomous agent?"
```

```
🤖 RAG System — Groq + Pinecone
   Type a question and press Enter. Type 'exit' to quit.

📝 What is chain-of-thought prompting?
💬 Answer:
Chain-of-thought prompting is a technique where the model is encouraged to
produce intermediate reasoning steps before arriving at a final answer.
This improves performance on complex tasks like math and logic problems...

📚 Sources (4 chunks retrieved):
   • https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
```

---

## Customising the data sources

Edit the `SOURCES` list in `ingest.py` to index your own content:

```python
# Web pages
SOURCES = ["https://your-docs-site.com/page1", ...]

# Local PDF files
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("my_document.pdf")
```

---

## Why RAG?

| Without RAG | With RAG |
|---|---|
| Model relies on training data (may be outdated) | Answer grounded in your up-to-date documents |
| Hallucinations are common | Hallucinations reduced — model stays within context |
| No source attribution | Sources cited for every answer |
| Entire context window used | Only the most relevant chunks are injected |

---

## Project structure

```
langchainrag/
├── ingest.py         # Document loading, chunking, embedding → Pinecone
├── query.py          # RAG chain: retrieve → prompt → generate
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── .gitignore        # Files excluded from git
└── README.md         # This file
```
