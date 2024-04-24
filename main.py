from fastapi import FastAPI
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


@app.get(
    "/quote",
    summary="Goracleus Larunen의 명언을 무작위로 반환합니다.",
    description="GET 요청을 받으면 Goracleus Larunen이 실제로 말한 명언을 반환합니다.",
    response_description="명언 Object를 반환하며 Object는 명언과 그 명언을 말한 날짜를 포함합니다.",
    response_model=Quote,
)
def get_quote():
    return {
        "quote": "Life is short so eat it all.",
        "year": 1324,
    }
