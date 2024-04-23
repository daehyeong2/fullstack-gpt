import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import openai
import glob
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🤝",
)

st.title("MeetingGPT")
st.markdown(
    "MeetingGPT에 오신 것을 환영합니다. 사이드바에서 동영상을 업로드하면 대화의 요약과 대화에 대한 질문을 할 수 있는 챗봇을 제공해 드립니다."
)

llm = ChatOpenAI(
    temperature=0.1,
)


@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = (
        video_path.replace(".mp4", ".mp3")
        .replace(".avi", ".mp3")
        .replace(".mkv", ".mp3")
        .replace(".mov", ".mp3")
    )
    if os.path.exists(audio_path):
        return
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if os.path.exists("./.cache/chunks/0_chunk.mp3"):
        return
    chunk_len = chunk_size * 60 * 1000
    track = AudioSegment.from_mp3(audio_path)
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"{chunks_folder}/{str(i).zfill(2)}_chunk.mp3",
            format="mp3",
        )


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if os.path.exists(destination):
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("영상 처리 중..", expanded=True):
        st.write("영상 불러오는 중..")
        video_content = video.read()
        video_path = f"./.cache/meeting_files/{video.name}"
        audio_path = (
            video_path.replace(".mp4", ".mp3")
            .replace(".avi", ".mp3")
            .replace(".mkv", ".mp3")
            .replace(".mov", ".mp3")
        )
        transcript_path = (
            video_path.replace(".mp4", ".txt")
            .replace(".avi", ".txt")
            .replace(".mkv", ".txt")
            .replace(".mov", ".txt")
        )
        with open(video_path, "wb") as f:
            f.write(video_content)
        st.write("소리 추출 중..")
        extract_audio_from_video(video_path)
        st.write("소리 분할 중..")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        st.write("대본 추출 중..")
        transcribe_chunks(
            chunks_folder,
            transcript_path,
        )

    transcript_tab, summary_tab, chat_tab = st.tabs(["대본", "요약", "대화"])

    with transcript_tab:
        with open(transcript_path, "r") as f:
            st.write(f.read())
    with summary_tab:
        start = st.button("요약 생성하기")
        if start:
            loader = TextLoader(transcript_path)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=2400,
                chunk_overlap=300,
            )
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                당신은 문서 요약 전문가입니다.
                다음 문서를 정확하게 요약하세요.
                
                {text}
            """
            )

            progress_text = "요약하는 중.."

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            my_bar = st.progress(0, text=f"{progress_text} (0/{len(docs)})")

            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            my_bar.progress(1 / len(docs), text=f"{progress_text} (1/{len(docs)})")

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up
                to a certain point: {existing_summary}
                We have the opportunity to refine the
                existing summary (only if needed) with
                some more context below.
                --------
                {context}
                --------
                Given the new context, refine the
                original summary.
                If the context isn't useful, RETURN the
                original summary.
            """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            for i, doc in enumerate(docs[1:]):
                summary = refine_chain.invoke(
                    {
                        "existing_summary": summary,
                        "context": doc.page_content,
                    }
                )
                my_bar.progress(
                    (i + 2) / len(docs), f"{progress_text} ({i+2}/{len(docs)})"
                )

            st.write(summary)
