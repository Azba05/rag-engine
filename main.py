from fastapi import FastAPI
from retrieval import detect_company, vectorstore

app = FastAPI(title="RAG Financial Engine", version="1.0")

@app.get("/")
def home():
    return {"message": "RAG Financial Engine API"}

@app.post("/ask")
def ask_question(question: str):
    company = detect_company(question)
    
    if company is None:
        return {
            "company_found": False,
            "answer": "Requested company not found in the knowledge base.",
            "sources": []
        }
    
    docs = vectorstore.similarity_search(question, k=5)
    
    if len(docs) == 0:
        return {
            "company_found": True,
            "answer": "No relevant documents found.",
            "sources": []
        }
    
    context = "\n".join([d.page_content for d in docs])
    
    return {
        "company_found": True,
        "answer": context[:300],
        "sources": [d.metadata['company'] for d in docs]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)