import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models.openai import ChatOpenAI

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


@st.cache_data(show_spinner="로딩 중..")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def foramt_document(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "어떤 정보를 사용 하실지 선택해 주세요.",
        (
            "파일",
            "위키피디아",
        ),
    )
    if choice == "파일":
        file = st.file_uploader("문서를 업로드해 주세요.", type=["pdf", "docx", "txt"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input(
            "위키피디아에서 검색", placeholder="검색할 내용을 입력해 주세요."
        )
        if topic:
            retriever = WikipediaRetriever(lang="ko")
            with st.status(f"위키피디아에 검색 중.."):
                docs = retriever.get_relevant_documents(topic)
            topic = ""


if not docs:
    st.markdown(
        """
QuizGPT에 오신 것을 환영합니다.

저는 위키피디아의 자료나 당신이 업로드한 파일을 이용해서 당신의 공부를 도울 것입니다.

사이드바에서 위키피디아에 검색하거나 당신의 파일을 업로드해서 시작해보세요.
"""
    )
else:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 문제 출제 전문가입니다.
당신은 학생들의 지식 수준을 평가하기 위해 주어진 정보를 이용해서 10개의 문제를 출제해야 합니다.
확실한 정보나 애매하지 않은 정보만 이용하세요.
각각의 문제는 무조건 4개의 선택지로 이루어져 있고 그 중 1개만 정답이게 하세요. (모든 선택지는 문장이 아닌 단어여야 합니다.)

문제 예시:

Question: 하늘은 무슨 색깔인가요?
Answer: 파란색(o)|검은색|보라색|노란색

Question: 컴퓨터의 특징을 고르세요.
Answer: 더러움|미끄러움|편리함(o)|굴러감

Question: 웹 개발에 필요한 것이 아닌 것은?
Answer: CSS|HTML|JS|연필깎이(o)

Question: 영화 "타이타닉"의 개봉 연도는?
Answer: 1996|1993|1998|1997(o)

Question: 다음 중 먹을 수 있는 것은?
Answer: 강철|콘크리트(o)|김밥|연필

이제 문제 출제에 이용할 정보를 알려드리겠습니다.
--------Context--------
{context}
-----------------------

Context에서 최대한 다양한 정보들을 이용해서 문제로 만드세요.
이제 Context를 이용해서 적당한 난이도의 문제를 10개 출제 하세요.
                """,
            ),
        ]
    )

    chain = {"context": foramt_document} | prompt | llm

    start = st.button("문제 생성")

    if start:
        chain.invoke(docs)
