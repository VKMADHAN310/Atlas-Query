from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from nl2sql_backend import answer_query, get_county_by_geoid


class QueryBody(BaseModel):
    query: str
    model: Optional[str] = None
    provider: Optional[str] = None  # 'ollama' | 'hf'


app = FastAPI(title="NL2SQL Backend API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/answer")
def post_answer(body: QueryBody):
    ans = answer_query(body.query, model=body.model, provider=body.provider)
    return ans.dict()


@app.get("/county/{geoid}")
def get_county(geoid: str):
    ans = get_county_by_geoid(geoid)
    return ans.dict()


