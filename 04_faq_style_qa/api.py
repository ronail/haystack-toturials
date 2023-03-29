from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel


class Query(BaseModel):
    query: str
    params: Union[dict, None] = None

app = FastAPI()

from ask import pipe
@app.post("/query")
def query(query: Query):
    prediction = pipe.run(query=query.query, params={"Retriever": {"top_k": 1}})
    return {"query": query.query, "answers": prediction["answers"]}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run("api:app", host="127.0.0.1", port=8080, reload=True)
