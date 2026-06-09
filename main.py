from fastapi import FastAPI
from pydantic import BaseModel
from rag import rag_pipeline

app = FastAPI()

class Query(BaseModel):
    question: str
    context: str

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model=Answer)
def ask(query: Query):

    answer = rag_pipeline(
        query.context,
        query.question
    )

    return {
        "answer": answer
    }