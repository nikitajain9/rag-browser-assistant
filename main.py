from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str
    context: str

class Answer(BaseModel):
    answer: str

@app.post('/ask',response_model=Answer)
def ask (query: Query):
    answer = rag_pipeline(
        request.content,
        request.question
    )
    return {'answer':answer}
