import streamlit as st
import os
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel, Field
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import SystemMessage
import requests

st.set_page_config(page_title="InvestorGPT", page_icon="🧑‍💻")

st.title("InvestorGPT")

st.markdown(
    """
    InvestorGPT에 오신 것을 환영합니다.
    궁금한 회사의 주식 정보에 대해 무엇이든 물어보세요!
"""
)

llm = ChatOpenAI(temperature=0.1)

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class CompanyArgsSchema(BaseModel):
    symbol: str = Field(description="회사의 주식 심볼 예시: APPL, TSLA")


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverviewTool"
    description = """
        회사의 재정 개요에 대해 알아보려면 이 툴을 사용하세요.
        주식 심볼을 입력해야 합니다.
    """
    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatementTool"
    description = """
        회사의 손익 계산서에 대해 알아보려면 이 툴을 사용하세요.
        주식 심볼을 입력해야 합니다.
    """
    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()["annualReports"]


class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformanceTool"
    description = """
        이걸 회사의 주간 성과를 알아보는데에 사용해.
        주식 심볼을 입력해야 합니다.
    """
    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        response = r.json()
        return list(response["Weekly Time Series"].items()[:200])


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
        이 툴은 회사의 주식 심볼을 찾는 툴입니다.
        쿼리를 argument로 활용합니다.
        예시 쿼리: Apple 회사의 주식 심볼
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = (
        StockMarketSymbolSearchToolArgsSchema
    )

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


agent = initialize_agent(
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
당신은 회사 펀드 매니저입니다.
당신은 사용자가 물어본 회사의 주식을 사야하는지 말아야 하는지 판단해줍니다.
판단 할 때 회사 개요, 손익 계산서, 주가 실적을 고려주세요.

당신은 사용자에게 주식을 살지 말지 단호하게 말해줘야 합니다.
"""
        )
    },
)

prompt = "Cloudflare의 주식에 대한 정보를 주고 그게 좋은 투자인지에 대해 분석해줘. 손익계산서와 주가 실적도 고려해줘."

company = st.text_input(
    "관심있는 회사의 이름을 적으세요.", placeholder="회사 이름을 입력하세요."
)

if company:
    result = agent.invoke(company)
    st.write(result["output"].replace("$", "\$"))
