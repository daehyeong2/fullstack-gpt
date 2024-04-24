from typing import Any
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Goracleus Larunen Quote Giver",
    description="Get a real quote said by Goracleus Larunen himself.",
    servers=[
        {
            "url": "https://nt-onion-hull-denial.trycloudflare.com",
        }
    ],
)


class Quote(BaseModel):
    quote: str = Field(
        description="Goracleus Larunen이 말한 명언",
    )
    year: int = Field(
        description="Goracleus Larunen이 명언을 말한 연도",
    )


user_token_db = {"ABCDEF": "gorani"}


@app.get(
    "/quote",
    summary="Goracleus Larunen의 명언을 무작위로 반환합니다.",
    description="GET 요청을 받으면 Goracleus Larunen이 실제로 말한 명언을 반환합니다.",
    response_description="명언 Object를 반환하며 Object는 명언과 그 명언을 말한 날짜를 포함합니다.",
    response_model=Quote,
)
def get_quote(request: Request):
    return {
        "quote": "Life is short so eat it all.",
        "year": 1324,
    }


@app.get("/authorize", response_class=HTMLResponse, include_in_schema=False)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
        <html>
            <head>
                <title>Goracleus Larunen</title>
            </head>
            <body>
                <h1>Log Into Goracleus Larunen</h1>
                <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Goracleus Larunen GPT</a>
            </body>
        </html>
    """


@app.post("/token", include_in_schema=False)
def handle_token(code=Form(...)):
    return {"access_token": user_token_db[code]}
