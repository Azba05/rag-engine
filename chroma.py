import json
import os
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --------------------------------------------------
# Load chunks
# --------------------------------------------------

print("Loading embedded chunks...")

with open("data/chunks_embedded.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

print(f"Loaded {len(chunks_data)} chunks")

# --------------------------------------------------
# Create LangChain Documents
# --------------------------------------------------

documents = []

for chunk in chunks_data:

    documents.append(
        Document(
            page_content=chunk["text"],
            metadata={
                "company": chunk["company"],
                "pdf_file": chunk["pdf_file"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "tokens": chunk["tokens"]
            }
        )
    )

print(f"Created {len(documents)} documents")

# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Create Chroma Database
# --------------------------------------------------

print("Creating Chroma database...")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\nDatabase created successfully!")
print(f"Stored {len(documents)} documents")
print("Saved to ./chroma_db")

# --------------------------------------------------
# Test Retrieval
# --------------------------------------------------

print("\n==============================")
print("Testing Retrieval")
print("==============================")

query = """
Adani Power

revenue

revenue from operations

sales

statement of profit and loss
"""

results = vectorstore.similarity_search(
    query,
    k=20
)

for i, doc in enumerate(results, 1):

    print(f"\nResult {i}")

    print("Company :", doc.metadata["company"])
    print("PDF     :", doc.metadata["pdf_file"])
    print("Page    :", doc.metadata["page"])
    print("Chunk   :", doc.metadata["chunk_id"])

    print("-" * 80)

    print(doc.page_content[:500])

print("\nDone.")