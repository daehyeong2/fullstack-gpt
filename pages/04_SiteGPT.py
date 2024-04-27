from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from bs4 import BeautifulSoup
import json
import streamlit as st


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


function = {
    "name": "answer_question",
    "description": "이 function은 효율적으로 산출된 답변을 받는 function입니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "isNew": {
                "type": "boolean",
                "description": "결과를 새로 만들지 안 만들지 정하는 property임. (과거 기록에 이미 답변된 정보가 있다면 새로 만들지 않으니 False입니다.)",
            },
            "answer": {
                "type": "string",
                "description": "과거 대화 기록에 답변된 정보가 있을 때 그 정보를 포함하는 property입니다. (만약 답변된 기록이 없다면 이 property는 존재하지 않습니다.)",
            },
        },
    },
    "required": ["isNew"],
}

llm = ChatOpenAI(
    temperature=0.1,
)

streaming_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

function_llm = ChatOpenAI(
    temperature=0.1,
).bind(
    functions=[
        function,
    ],
    function_call={"name": "answer_question"},
)

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=1000
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

사용자가 이전 대화 기록에 관해서 질문을 할 수도 있습니다.
그런 경우에는 이전 대화 기록을 참고해서 답변을 만들고 점수를 부여해 주세요.

이전 대화 기록:
{memory}

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

    예시:(
    질문: Next.js 강의는 얼마이며 수강생은 몇명인가요?
    결과: Next.js는 160,000원입니다. 하지만 현재는 할인 중이라서 75,000원입니다. 수강생은 3866명입니다. (출처: https://discord.com/about)
    )
""",
        ),
        ("human", "{question}"),
    ]
)

cache_prompt = ChatPromptTemplate.from_template(
    """
당신은 과거 기록을 보고 새로운 질문과 비슷한 의미를 지닌 기록을 찾아서 이미 ai가 답한 데이터를 반환해서 효율적으로 일하게 해주는 비서입니다.
따라서 당신은 질문에 대한 답변을 당신이 생성해서는 안되며 기록에서 찾아서 반환해야 합니다. 만약에 존재하지 않다면 반환하지 않습니다.
    
주어진 과거 대화 기록에서 새로운 질문과 비슷한 의미의 질문을 했던 기록을 찾고 그 질문이 ai에게 답변이 완료됐는지 확인하세요.
만약 ai에게 답변이 완료되었고 비슷한 의미의 질문이 존재한다면 isNew를 False로 설정하고(새롭게 생성하지 않고 대화 기록에서 가져오기 때문에 New가 아님) answer를 그 완료된 답변으로 설정하세요.
만약 ai에게 답변이 완료되지 않았거나 비슷한 질문을 했던 기록이 없다면, isNew를 True로 설정하고 answer를 작성하지 마세요.
주의 해야 할 점은 대화 기록의 마지막에 새로운 질문이 저장되어 있습니다. 그 질문은 ai에게 답변되지 않았기 때문에 사용하지 마세요.

자 명심하세요, 과거 대화 기록에 비슷한 의미의 질문이 이미 답변 된 정보가 있다면 isNew는 False가 되며 그 ai에게 이미 답변된 답변을 수정하지 말고 그대로 answer에 넣으세요.
만약 비슷한 의미의 질문이 답변되지 않았다면 isNew는 True가 되며 answer는 존재하지 않습니다.

문장 구조가 비슷해도 가리키는 대상이 다르다면 다른 질문입니다,
반대로 문장 구조가 다르더라도 가리키는 대상이 같고 의미가 같은 경우가 있으니 모든 것을 고려해주세요.

짧은 예시: 4B 연필은 진한가요?와 2B 연필은 진한가요? 은 다른 질문이니 isNew는 True이다, 고라니의 뿔은 무슨 색인가요?와 고라니가 가지고 있는 뿔의 색깔을 알려주세요. 는 같은 질문이니 isNew가 False이다.

--------예시--------
1번 예시:

예시 과거 대화 기록:
human: 하늘은 무슨 색인가요?
ai: 하늘색입니다. (출처: https://example.com/source)
human: 물티슈는 무엇으로 만들어지나요?

예시 질문:
물티슈는 무엇으로 만들어지나요?

예시 결과:
isNew = True

||||||||||||||

2번 예시:

예시 과거 대화 기록:
human: 연필은 무엇을 하는데 사용되나요?

예시 질문:
연필은 무엇을 하는데 사용되나요?

예시 결과:
isNew = True
answer =

||||||||||||||

3번 예시:

예시 과거 대화 기록:
human: 빛의 삼원색은 무엇인가요?
ai: 빨간색, 초록색, 파란색입니다. (출처: https://example.com/source)
human: 안녕하세요 저는 Gorani인데요, 빛의 삼원색을 알려주세요.

예시 질문:
안녕하세요 저는 Gorani인데요, 빛의 삼원색을 알려주세요.

예시 결과:
isNew = False
answer = 빨간색, 초록색, 파란색입니다. (출처: https://example.com/source)

||||||||||||||

3번 예시:

예시 과거 대화 기록:
human: 바다는 무엇으로 이루어져 있는지 궁금해요.
ai: 바다는 대부분 물로 이루어져 있습니다. (출처: https://example.com/source)
human: 안녕하세요 반갑습니다! 바다는 무엇으로 이루어져 있나요?

예시 질문:
안녕하세요 반갑습니다! 바다는 무엇으로 이루어져 있나요?

예시 결과:
isNew = False
answer = 바다는 대부분 물로 이루어져 있습니다. (출처: https://example.com/source)

||||||||||||||

4번 예시:

예시 과거 대화 기록:
human: 일기를 쓸 때 어떤 걸 쓰면 좋을까?
ai: 오늘의 날씨와 있었던 일, 느낀 점을 쓰면 좋습니다. (출처: https://example.com/source)
human: 일기장을 살 때 어떤 걸 사면 좋을까?

예시 질문:
일기장을 살 때 어떤 걸 사면 좋을까?

예시 결과:
isNew = True
answer =

||||||||||||||

5번 예시:

예시 과거 대화 기록:
human: 고라니는 뿔이 몇개 달려 있어?
ai: 암컷은 안 달려 있고 수컷은 2개 달려 있습니다.. (출처: https://example.com/source)
human: 유니콘은 뿔이 몇개 달려 있어?

예시 질문:
유니콘은 뿔이 몇개 달려 있어?

예시 결과:
isNew = True
answer =
--------------------

이제 당신 차례입니다!

실제 과거 대화 기록:
{history}

실제 질문:
{question}
"""
)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def save_memory(input, output):
    st.session_state["memory"].save_context({"input": input}, {"output": output})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def get_answers(inputs):
    with st.spinner("답변 생성 중.."):
        docs = inputs["docs"]
        question = inputs["question"]
        memory = inputs["memory"]
        answers_chain = answers_prompt | llm
        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {
                            "context": doc.page_content,
                            "question": question,
                            "memory": memory,
                        }
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
    choose_chain = choose_prompt | streaming_llm
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


def invoke_chain(query):
    with st.spinner("캐시 확인 중.."):
        condensed = "\n".join(
            f"{message['role']}: {message['message']}"
            for message in st.session_state["messages"]
        )
        cache_chain = cache_prompt | function_llm
        cache_result = json.loads(
            cache_chain.invoke(
                {
                    "history": condensed,
                    "question": query,
                }
            ).additional_kwargs["function_call"]["arguments"]
        )
    if cache_result["isNew"]:
        result = chain.invoke(query)
        save_memory(query, result.content)
    else:
        result = cache_result["answer"]
        st.markdown(result)
        save_message(result, "ai")
        save_memory(query, result)


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
    print(retriever.invoke("플러터"))
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

with st.sidebar:
    url = st.text_input(
        "URL을 입력해 주세요.",
        placeholder="https://example.com/sitemap.xml",
    )

if not url:
    st.markdown(
        """
                웹 사이트의 내용에 대해 궁금한 것을 질문해 보세요.

                사이드 바에서 웹 사이트의 URL을 입력해서 시작하세요.
    """
    )
    st.session_state["messages"] = []
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=1000
    )
else:
    if ".xml" not in url:
        with st.sidebar:
            st.error("사이트맵 URL을 입력해 주세요.")
    else:
        header = st.empty()
        paint_history()
        retriever = load_website(url)
        header.info("웹 사이트 정보를 학습했습니다. 무엇이든 질문해 보세요.")
        query = st.chat_input(placeholder="웹 사이트에 대해 무엇이든 물어보세요!")
        if query:
            send_message(query, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                    "memory": st.session_state["memory"].load_memory_variables,
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            with st.chat_message("ai"):
                invoke_chain(query)
