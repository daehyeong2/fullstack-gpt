import time
import streamlit as st
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models.openai import ChatOpenAI

st.set_page_config(page_title="DocumentGPT", page_icon="📜")

with st.sidebar:
    temperature = st.slider("Temperature", 0.1, 1.0)

llm = ChatOpenAI(temperature=temperature)


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


def foramt_document(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 문서 관련 전문가입니다. 당신은 사용자의 질문에 대답해야 합니다.
대답을 할 때에는 주어진 context만으로 대답하세요. 당신이 원래 알고 있는 지식을 이용하지 마세요.
만약 당신이 모른다면 모른다고 하세요. 말을 지어내지 마세요.
--------Context--------
{context}
-----------------------
""",
        ),
        ("human", "{question}"),
    ]
)

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
            chain = (
                {
                    "context": retriever | RunnableLambda(foramt_document),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            response = chain.invoke(message)
            send_message(response.content, "ai")
    else:
        st.info("먼저 문서를 업로드 해주세요!")
