import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.embeddings import FakeEmbeddings
from htmlTemplates import css, bot_template, user_template

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def get_vector_store(chunks):
    # FakeEmbeddings works locally without torch/sentence-transformers
    embeddings = FakeEmbeddings(size=768)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="llama3-8b-8192",
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

def handle_user_input(question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your study materials first!")
        return
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="StudyMind", page_icon="📚", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("📚 StudyMind — Your AI Study Assistant")
    st.markdown("Upload your textbooks, notes, or PDFs and ask any question!")

    user_question = st.text_input("Ask a question about your study material:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("📂 Upload Study Materials")
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Process Documents", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF!")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vectorstore = get_vector_store(chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success(f"✅ Done! {len(pdf_docs)} file(s) processed. Start asking!")

        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        st.markdown("- Explain the concept of X")
        st.markdown("- Summarize chapter 3")
        st.markdown("- What are the key differences between A and B?")
        st.markdown("- Give me 5 MCQs on this topic")

if __name__ == "__main__":
    main()
