import time
import streamlit as st
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import UnstructuredFileLoader

st.set_page_config(page_title="DocumentGPT", page_icon="📜")

st.title("DocumentGPT")

st.markdown(
    """
환영합니다!

이 챗봇을 사용해서 당신의 문서에 대해 물어보세요!
"""
)


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()

    return retriever


file = st.file_uploader(
    ".txt .pdf .docx 파일을 업로드하세요.", type=["pdf", "txt", "docx"]
)

if file:
    retriever = embed_file(file)
    doc = retriever.invoke("재귀함수는 무엇인가요?")
    st.write(doc)
