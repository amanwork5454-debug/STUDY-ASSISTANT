import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings
from htmlTemplates import css, bot_template, user_template
import numpy as np
from typing import List

load_dotenv()

# ── Custom Hash-based Embeddings (no torch, no protobuf, works on Python 3.14) ──
class GroqEmbeddings(Embeddings):
    def _get_embedding(self, text: str) -> List[float]:
        words = text.lower().split()
        vec = np.zeros(384)
        for i, word in enumerate(words):
            idx = hash(word) % 384
            vec[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)


# ── PDF Parsing ──────────────────────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# ── Chunking ─────────────────────────────────────────────────────────────────
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# ── Vector Store ──────────────────────────────────────────────────────────────
def get_vector_store(chunks):
    embeddings = GroqEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


# ── Conversation Chain ────────────────────────────────────────────────────────
def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        ),
        memory=memory
    )


# ── Handle User Input ─────────────────────────────────────────────────────────
def handle_user_input(question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your study materials first!")
        return
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content),
                     unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content),
                     unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="StudyMind — AI Study Assistant",
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("📚 StudyMind — Your AI Study Assistant")
    st.markdown("Upload your textbooks, lecture notes, or PDFs and ask any question!")

    user_question = st.text_input("💬 Ask a question about your study material:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("📂 Upload Study Materials")
        pdf_docs = st.file_uploader(
            "Upload PDFs (textbooks, notes, papers)",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("⚡ Process Documents", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF!")
            else:
                with st.spinner("Processing your study materials..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    st.info(f"Created {len(chunks)} text chunks")
                    vectorstore = get_vector_store(chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success(f"✅ {len(pdf_docs)} file(s) processed! Start asking questions.")

        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        st.markdown("- What is the full form of BDA?")
        st.markdown("- Summarize chapter 1")
        st.markdown("- Explain MapReduce")
        st.markdown("- Difference between OLAP and OLTP")
        st.markdown("- Give me 5 MCQs on Hadoop")
        st.markdown("---")
        st.markdown("**🔧 Stack:** LangChain · FAISS · Groq LLaMA-3")


if __name__ == "__main__":
    main()
