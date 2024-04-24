import os
from pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="gcp-starter",
)

embeddings = OpenAIEmbeddings()

vector_store = PineconeVectorStore.from_existing_index(
    "recipes",
    embeddings,
)

app = FastAPI(
    title="ChefGPT. 세계에서 가장 최고의 인도 요리를 제공함.",
    description="ChefGPT에게 몇가지 재료만 알려주세요. 그러면 레시피를 알려드릴게요.",
    servers=[
        {
            "url": "https://annual-cartridges-caught-recorder.trycloudflare.com",
        }
    ],
)

user_token_db = {}


class Document(BaseModel):
    page_content: str = Field(description="레시피에 대한 자세한 설명")


@app.get(
    "/recipes",
    summary="레시피 목록을 반환합니다.",
    description="재료들을 받아서, 해당 재료가 들어가는 요리의 레시피를 반환합니다.",
    response_description="레시피와 준비 설명서를 포함하는 문서 객체",
    response_model=list[Document],
)
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(ingredient)
    return docs


@app.get(
    "/add_favorite",
    summary="좋아하는 음식 목록에 음식을 추가합니다.",
    description="음식 이름을 받고 그 음식을 좋아하는 음식 리스트에 저장합니다.",
)
def add_favorite_food(request: Request, name: str):
    token = request.headers["authorization"].split()[1]
    if token not in user_token_db:
        user_token_db[token] = {"favorite_food_list": []}
    user_token_db[token]["favorite_food_list"].append(name)
    return {"ok": True}


@app.get(
    "/get_favorite",
    summary="좋아하는 음식 목록을 받습니다.",
    description="사용자의 좋아하는 음식 목록을 받습니다.",
    response_description="사용자의 좋아하는 음식 목록",
    response_model=list[str],
)
def get_favorite_food(request: Request):
    token = request.headers["authorization"].split()[1]
    if token not in user_token_db:
        return []
    else:
        return user_token_db[token]["favorite_food_list"]


@app.get("/authorize", response_class=HTMLResponse, include_in_schema=False)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    user_token_db[client_id] = {"favorite_food_list": []}
    return f"""
        <html>
            <head>
                <title>Goracleus Larunen</title>
            </head>
            <body>
                <h1>Log Into Goracleus Larunen</h1>
                <a href="{redirect_uri}?code={client_id}&state={state}">Authorize Goracleus Larunen GPT</a>
            </body>
        </html>
    """


@app.post("/token", include_in_schema=False)
def handle_token(code=Form(...)):
    return {"access_token": code}
