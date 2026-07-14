import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

print("Loading embedded chunks...")
with open('data/chunks_embedded.json', 'r', encoding='utf-8') as f:
    chunks_data = json.load(f)

print(f"Loaded {len(chunks_data)} embedded chunks")

documents = []
for chunk in chunks_data:
    doc = Document(
        page_content=chunk['text'],
        metadata={
            'company': chunk['company'],
            'pdf_file': chunk['pdf_file'],
            'chunk_id': chunk['chunk_id']
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents for Chroma")

print("Initializing Chroma vector database...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f" Chroma vector database created!")
print(f" Stored {len(documents)} documents in Chroma")
print(f" Database saved to: ./chroma_db")

print("\nTesting retrieval...")
query = "What is the company's revenue?"
results = vectorstore.similarity_search(query, k=20)

print(f" Retrieved {len(results)} relevant chunks for test query")
for i, result in enumerate(results):
    print(f"  Result {i+1}: {result.metadata['company']} (chunk {result.metadata['chunk_id']})")