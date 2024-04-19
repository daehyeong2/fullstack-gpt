from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
import streamlit as st

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
당신은 문서 분석 전문가입니다.
주어진 문서들을 보고 질문과 관련 있는 부분을 찾아 답하고 답변에 대한 점수를 부여해야 합니다.
점수는 0점부터 5점까지 있으며 0점은 쓸모 없는 답변, 5점은 매우 유용한 답변입니다.

예시:

질문: 달은 얼마나 멀리 있나요?
답: 달은 384,400 km 만큼 떨어져 있습니다.
점수: 5

질문: 달은 얼마나 멀리 있나요?
답: 달은 384,400 명입니다.
점수: 0

질문: 태양은 얼마나 멀리 있나요?
답: 잘 모르겠습니다.
스코어: 0

당신의 차례입니다!

문서: {context}

질문: {question}
""",
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    주어진 답변들만 이용해서 사용자의 질문에 답하세요.
     
    점수가 높은 답변들을 더 이용하세요. (더욱 유용한 정보입니다)
    그리고 최근 정보들을 우선적으로 이용하세요.
    점수가 낮은 답변은 결과에 포함하지 마세요.
    결과를 생성하는데 사용된 답변의 출처들을 변경하지 말고, 그대로 반환하세요.
    출처는 무조건 표시해야 하며 날짜는 표시하지 않아도 됩니다.

    답변들: {answers}

    예시:

    질문: Next.js 강의는 얼마이며 수강생은 몇명인가요?

    --------------결과--------------
    Next.js는 160,000원입니다. 하지만 현재는 할인 중이라서 75,000원입니다. 수강생은 3866명입니다.

    출처: https://discord.com/about 
    --------------------------------
""",
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"context": doc.page_content, "question": question}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\n출처: {answer['source']}\n날짜: {answer['date']}"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_data(show_spinner="임베딩 중..")
def embed_file(url, _docs):
    cache_dir = LocalFileStore(f"./.cache/site_embeddings/{url.replace('/', '')}")
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(_docs, cached_embeddings)
    retriever = vector_store.as_retriever()
    return retriever


@st.cache_data(show_spinner="웹 사이트 정보 불러오는 중..")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    retriever = embed_file(url, docs)
    return retriever


st.set_page_config(page_title="SiteGPT", page_icon="🖥️")

st.title("SiteGPT")


st.markdown(
    """
            웹 사이트의 내용에 대해 궁금한 것을 질문해 보세요.

            사이드 바에서 웹 사이트의 URL을 입력해서 시작하세요.
"""
)

with st.sidebar:
    url = st.text_input(
        "URL을 입력해 주세요.",
        placeholder="https://example.com",
    )


if url and ".xml" not in url:
    with st.sidebar:
        st.error("사이트맵 URL을 입력해 주세요.")
elif url:
    retriever = load_website(url)
    query = st.text_input("웹 사이트에 대해 무엇이든 물어보세요!")
    if query:
        chain = (
            {"docs": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content)
