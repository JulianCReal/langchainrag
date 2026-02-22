"""
query.py — RAG chain: retrieve relevant chunks from Pinecone, generate answer with Groq.

Usage:
    python query.py                          # interactive mode
    python query.py "your question here"     # single question from CLI
"""

import os
import sys
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "langchainrag")
TOP_K            = 4   # number of chunks to retrieve

# 1. Load embedding model (same as ingest.py)
print("🔢 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to Pinecone index
print("🌲 Connecting to Pinecone...")
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)

# 3. RAG prompt
RAG_TEMPLATE = """You are an assistant for question-answering tasks.
Use ONLY the following retrieved context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Keep your answer concise and clear (3-5 sentences).

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# 4. Initialise Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)

# 5. Format retrieved docs
def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# 6. Build the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Ask function 
def ask(question: str) -> None:
    """Run the RAG chain and print the answer with sources."""
    print(f"\n❓ Question: {question}")
    print("-" * 60)

    # Retrieve docs to show sources
    retrieved_docs = retriever.invoke(question)

    # Stream the answer
    print("💬 Answer:")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")

    # Show sources
    sources = sorted({doc.metadata.get("source", "unknown") for doc in retrieved_docs})
    print(f"📚 Sources ({len(retrieved_docs)} chunks retrieved):")
    for src in sources:
        print(f"   • {src}")
    print()

# 8. Interactive mode
def interactive_mode() -> None:
    print("🤖 RAG System — Groq + Pinecone")
    print("   Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("📝 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not question:
            print("   ⚠️  Please enter a question.\n")
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("👋 Goodbye!")
            break

        ask(question)

# 9. Entry point
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single question from CLI
        ask(" ".join(sys.argv[1:]))
    else:
        # Interactive mode
        interactive_mode()
