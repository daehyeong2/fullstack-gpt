import time
import streamlit as st
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import UnstructuredFileLoader

st.set_page_config(page_title="DocumentGPT", page_icon="📜")


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


@st.cache_data(show_spinner="Embedding..")
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


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


st.title("DocumentGPT")

chat, file_upload = st.tabs(["Chat", "Document"])

with file_upload:
    file = st.file_uploader("🚀 문서를 업로드 해주세요.", type=["pdf", "txt", "docx"])

if file:
    message = st.chat_input("AI에게 문서에 대해 궁금한 것을 물어보세요!")
else:
    st.session_state["messages"] = []

with chat:
    if file:
        retriever = embed_file(file)
        st.success(
            "문서 학습을 완료했습니다. 이제 AI에게 문서에 대해 무엇이든 물어보세요!"
        )
        paint_history()
        if message:
            send_message(message, "human")
    else:
        st.info("먼저 문서를 업로드 해주세요!")
