from operator import itemgetter
import streamlit as st
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models.ollama import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(page_title="PrivateGPT", page_icon="⚙️")

with st.sidebar:
    temperature = st.slider("Temperature", 0.1, 1.0)
    st.session_state["model"] = st.selectbox(
        "모델을 선택 해주세요.", ("mistral", "gemma", "llama2")
    )


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model=st.session_state["model"],
    temperature=temperature,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


@st.cache_data(show_spinner="Embedding..")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(model="mistral:latest")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()

    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def save_memory(input, output):
    st.session_state.memory.save_context({"input": input}, {"output": output})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def foramt_document(docs):
    return "\n\n".join(document.page_content for document in docs)


def invoke_chain(message):
    response = chain.invoke(message)
    save_memory(message, response.content)


prompt = ChatPromptTemplate.from_template(
    """
당신은 문서 관련 전문가입니다. 당신은 사용자의 질문에 대답해야 합니다.
대답을 할 때에는 주어진 context만으로 대답하세요. 트레이닝 데이터를 제외하세요.
context에 있는 내용을 기반으로 결과를 생성해줘.
만약 당신이 모른다면 모른다고 하세요. 말을 지어내지 마세요.

사용자는 과거 대화 기록에서 질문 할 수도 있습니다. 따라서 다음 자료를 확인하세요.
--------History-------
{history}
-----------------------

--------Context--------
{context}
-----------------------

--------Question-------
{question}
-----------------------
"""
)

st.title("PrivateGPT")

chat, file_upload = st.tabs(["Chat", "Document"])

with file_upload:
    file = st.file_uploader("🚀 문서를 업로드 해주세요.", type=["pdf", "txt", "docx"])

if file:
    message = st.chat_input("AI에게 문서에 대해 궁금한 것을 물어보세요!")
else:
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=3000, return_messages=True
    )

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
                | RunnablePassthrough.assign(
                    history=RunnableLambda(
                        st.session_state.memory.load_memory_variables
                    )
                    | itemgetter("history")
                )
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                invoke_chain(message)
    else:
        st.info("먼저 문서를 업로드 해주세요!")
