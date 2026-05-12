import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template

load_dotenv()

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
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# ── Vector Store (FAISS + HuggingFace Embeddings — FREE) ─────────────────────
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# ── Conversational Chain (Groq LLM — FREE) ───────────────────────────────────
def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr",       # Max Marginal Relevance — diverse results
            search_kwargs={"k": 5}
        ),
        memory=memory,
        verbose=False
    )
    return chain

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

    user_question = st.text_input("Ask a question about your study material:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("📂 Upload Study Materials")
        pdf_docs = st.file_uploader(
            "Upload PDFs (textbooks, notes, papers)",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process Documents", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF!")
            else:
                with st.spinner("Processing your study materials..."):
                    # Step 1: Extract text
                    raw_text = get_pdf_text(pdf_docs)

                    # Step 2: Chunk
                    chunks = get_text_chunks(raw_text)
                    st.info(f"Created {len(chunks)} text chunks")

                    # Step 3: Embed + Vector store
                    vectorstore = get_vector_store(chunks)

                    # Step 4: Build chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success(f"✅ {len(pdf_docs)} document(s) processed! Start asking questions.")

        st.markdown("---")
        st.markdown("**💡 Sample Questions:**")
        st.markdown("- Explain the concept of X")
        st.markdown("- Summarize chapter 3")
        st.markdown("- What are the key differences between A and B?")
        st.markdown("- Give me 5 MCQs on this topic")

if __name__ == "__main__":
    main()
