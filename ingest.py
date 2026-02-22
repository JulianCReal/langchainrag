"""
ingest.py — Load documents, split into chunks, embed and store in Pinecone.
Run this ONCE before querying.

Usage:
    python ingest.py
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "langchainrag")

# Documents to ingest (you can add or replace these URLs)
SOURCES = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

# 1. Load documents
print("📥 Loading documents...")
loader = WebBaseLoader(SOURCES)
docs = loader.load()
print(f"   Loaded {len(docs)} document(s)")

# 2. Split into chunks
print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
chunks = splitter.split_documents(docs)
print(f"   {len(docs)} doc(s) → {len(chunks)} chunks")

# 3. Load embedding model (free, runs locally)
print("🔢 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast & free
)

# 4. Connect to Pinecone and create index if needed
print("🌲 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

existing = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"   Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,        # must match all-MiniLM-L6-v2 output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("   Index created.")
else:
    print(f"   Index '{INDEX_NAME}' already exists.")

# 5. Embed and upsert
print("⬆️  Embedding and uploading chunks to Pinecone...")
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    pinecone_api_key=PINECONE_API_KEY,
)

print(f"\n✅ Done! {len(chunks)} chunks stored in index '{INDEX_NAME}'.")
print("   You can now run:  python query.py")
