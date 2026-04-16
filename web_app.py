import re
from urllib.parse import parse_qs, urlparse

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi


load_dotenv()

VIDEO_PROCESSING_HELP_TEXT = (
    "Could not process this video. Possible reasons:\n"
    "1. Captions are unavailable for the selected language (English/Hindi).\n"
    "2. The URL is invalid or the video is unavailable.\n"
    "3. Transcript access is temporarily blocked/rate-limited.\n"
    "4. You have done multiple requests. Please try changing your IP address.\n"
    "Please try a different video or try again later."
)

ANSWERING_HELP_TEXT = (
    "Could not generate an answer right now. Please try again in a few seconds."
)


def extract_video_id_from_url(url: str) -> str | None:
    value = (url or "").strip()
    if not value:
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    host = parsed.netloc.lower()

    if "youtu.be" in host:
        candidate = parsed.path.strip("/")
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            return candidate

    if "youtube.com" in host:
        params = parse_qs(parsed.query)
        candidate = (params.get("v") or [None])[0]
        if candidate and re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            return candidate

        # Handles /shorts/<id> and /embed/<id>
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed"}:
            candidate = parts[1]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                return candidate

    return None


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_chain(video_id: str):
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=["en", "hi"])
    transcript = " ".join(chunk.text for chunk in transcript_list)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant.\n"
            "Answer ONLY from the provided transcript context.\n"
            "If the context is insufficient, just say you don't know.\n\n"
            "{context}\n"
            "Question: {question}"
        ),
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(temperature=0.2)
    parser = StrOutputParser()

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    return parallel_chain | prompt | llm | parser


def clear_active_video_context():
    st.session_state.rag_chain = None
    st.session_state.video_id = None
    st.session_state.chat_history = []


st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥", layout="centered")
st.title("🎥 YouTube Chatbot (RAG)")
st.write("Paste a YouTube URL, process transcript, then ask questions.")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

process_video_clicked = st.button("Process Video", type="primary")


if process_video_clicked:
    video_id = extract_video_id_from_url(video_url)
    if not video_id:
        clear_active_video_context()
        st.error("Invalid YouTube URL. Please paste a valid full YouTube link.")
    else:
        with st.spinner("Processing video..."):
            try:
                st.session_state.rag_chain = build_chain(video_id)
                st.session_state.video_id = video_id
                st.session_state.chat_history = []
                st.success("Video processed. Ask your questions below.")
            except TranscriptsDisabled:
                clear_active_video_context()
                st.error("No captions available for this video.")
            except Exception:
                clear_active_video_context()
                st.error(VIDEO_PROCESSING_HELP_TEXT)

question = st.text_input("Ask a question about the video")
ask_clicked = st.button("Ask")

if ask_clicked:
    if st.session_state.rag_chain is None:
        st.warning("Process a video first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.rag_chain.invoke(question.strip())
                st.session_state.chat_history.append(
                    {"question": question.strip(), "answer": answer}
                )
            except Exception:
                st.error(ANSWERING_HELP_TEXT)

if st.session_state.video_id:
    st.caption(f"Current video ID: {st.session_state.video_id}")

if st.session_state.chat_history:
    st.subheader("Conversation")
    for item in st.session_state.chat_history:
        st.markdown(f"**You:** {item['question']}")
        st.markdown(f"**Bot:** {item['answer']}")
