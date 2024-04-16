from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st


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

html2text_transformer = Html2TextTransformer()

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)
