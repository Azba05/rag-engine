from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retrieval import ask_grok

app = FastAPI(
    title="Financial RAG",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message":"Financial RAG API"}


@app.post("/ask")
def ask(question:str):

    result=ask_grok(question)

    return{

        "question":question,

        "answer":result["answer"],

        "sources":list(
            {
                d.metadata["company"]
                for d in result["sources"]
            }
        )

    }


if __name__=="__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )