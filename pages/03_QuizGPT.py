import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


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


with st.sidebar:
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
            st.write(docs)
    else:
        topic = st.text_input(
            "위키피디아에서 검색", placeholder="검색할 내용을 입력해 주세요."
        )
        if topic:
            retriever = WikipediaRetriever(lang="ko")
            with st.status(f'"위키피디아에 검색 중..'):
                docs = retriever.get_relevant_documents(topic)
            st.write(docs)
