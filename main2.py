from fastapi import FastAPI
from pydantic import BaseModel

# Import new retrieval file
from retrieval_v2 import retrieve_answers

app = FastAPI(
    title="RAG Financial Engine",
    version="2.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG Financial Engine API"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = retrieve_answers(request.question)

    return {
        "company_found": result["company_found"],
        "answer": result["answer"],
        "sources": result["sources"]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )