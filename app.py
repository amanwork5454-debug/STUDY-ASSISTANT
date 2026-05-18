import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e !important;
}

/* ── Hide Streamlit chrome but keep sidebar toggle ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
header { visibility: hidden; }

/* ── Sidebar toggle button — always visible ── */
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] > div > button,
button[data-testid="baseButton-header"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: all !important;
    z-index: 9999 !important;
    background: #1e1e2e !important;
    border: 1px solid #4f46e5 !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 12px 6px !important;
    position: fixed !important;
    left: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
}
[data-testid="collapsedControl"] svg,
section[data-testid="stSidebar"] > div > button svg {
    fill: #a5b4fc !important;
    stroke: #a5b4fc !important;
    color: #a5b4fc !important;
    width: 16px !important;
    height: 16px !important;
}
[data-testid="collapsedControl"]:hover {
    background: #4f46e5 !important;
}

/* ── Main header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 500;
    color: #a5b4fc;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    line-height: 1.1;
}
.hero-sub {
    font-size: 15px;
    color: #6b7280;
    font-weight: 300;
    margin: 0;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e1e3a, transparent);
    margin: 1.5rem 0;
}

/* ── Chat container ── */
.chat-wrap {
    max-width: 780px;
    margin: 0 auto;
    padding: 0 1rem 6rem;
}

/* ── Message bubbles ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.3s ease;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.3s ease;
}
.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: #fff;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3);
}
.bubble-bot {
    background: #13131f;
    border: 1px solid #1e1e2e;
    color: #d1d5db;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    position: relative;
}
.bubble-bot::before {
    content: '🧠';
    position: absolute;
    left: -34px;
    top: 10px;
    font-size: 18px;
}
.bubble-user-label {
    font-size: 10px;
    color: #6366f1;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-align: right;
    margin-bottom: 4px;
    padding-right: 4px;
}
.bubble-bot-label {
    font-size: 10px;
    color: #4b5563;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
    padding-left: 4px;
}

/* ── Input styling ── */
[data-testid="stTextInput"] input {
    background: #13131f !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 12px !important;
    color: #e5e7eb !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #4b5563 !important; }
[data-testid="stTextInput"] label {
    color: #6b7280 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif !important;
    color: #e0e7ff !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] p {
    color: #6b7280 !important;
    font-size: 13px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 1px dashed #1e1e2e !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    color: #6b7280 !important;
    font-size: 13px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Clear button override ── */
.clear-btn .stButton > button {
    background: #1a1a2e !important;
    border: 1px solid #2d2d44 !important;
    color: #9ca3af !important;
    -webkit-text-fill-color: #9ca3af !important;
}
.clear-btn .stButton > button:hover {
    background: #1f0a0a !important;
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    -webkit-text-fill-color: #ef4444 !important;
}

/* ── Status messages ── */
[data-testid="stSuccess"] {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}
[data-testid="stInfo"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,0.08) !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 10px !important;
    color: #fcd34d !important;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 1.2rem 0 1.5rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 1.5rem;
}
.sidebar-logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #4f46e5, #818cf8);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.sidebar-logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #e0e7ff;
}
.sidebar-logo-version {
    font-size: 10px;
    color: #4b5563;
    font-weight: 400;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 8px;
    margin: 1rem 0;
}
.stat-box {
    flex: 1;
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #a5b4fc;
}
.stat-lbl {
    font-size: 10px;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #374151;
}
.empty-icon {
    font-size: 48px;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.empty-text {
    font-size: 15px;
    color: #374151;
    font-weight: 300;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #6366f1 !important; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.processing { animation: pulse 1.5s infinite; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0
if "files_count" not in st.session_state:
    st.session_state.files_count = 0
if "questions_count" not in st.session_state:
    st.session_state.questions_count = 0


# ── Core Functions ────────────────────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🧠</div>
        <div>
            <div class="sidebar-logo-text">StudyMind</div>
            <div class="sidebar-logo-version">Powered by Groq + Cohere</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Study Materials")
    pdf_docs = st.file_uploader(
        "Drop your PDFs here",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("⚡  Process Documents", type="primary"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Reading & indexing your PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vector_store(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.chunks_count = len(chunks)
                st.session_state.files_count = len(pdf_docs)
                st.session_state.questions_count = 0
                st.session_state.chat_history = None
            st.success(f"✓ Ready! {len(pdf_docs)} file(s) indexed.")

    # Stats
    if st.session_state.files_count > 0:
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-val">{st.session_state.files_count}</div>
                <div class="stat-lbl">Files</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{st.session_state.chunks_count}</div>
                <div class="stat-lbl">Chunks</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{st.session_state.questions_count}</div>
                <div class="stat-lbl">Asked</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Clear chat button
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation"):
        st.session_state.chat_history = None
        st.session_state.questions_count = 0
        if st.session_state.conversation:
            st.session_state.conversation.memory.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='border-top:1px solid #1e1e2e; padding-top:16px;'>
        <p style='font-size:11px; color:#374151; line-height:1.6;'>
            🔗 LangChain &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; Groq LLaMA-3.3<br>
            🎯 Cohere Embeddings &nbsp;·&nbsp; RAG Pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ GenAI + RAG</div>
    <h1 class="hero-title">StudyMind AI</h1>
    <p class="hero-sub">Upload your study materials. Ask anything. Get instant answers.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# Input
user_question = st.text_input(
    "question",
    placeholder="Ask anything about your uploaded study material...",
    label_visibility="collapsed"
)

# Handle question
if user_question and st.session_state.conversation:
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    st.session_state.questions_count += 1
elif user_question and not st.session_state.conversation:
    st.warning("⚠️ Please upload and process your PDFs first using the sidebar.")

# Display only latest Q&A
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if st.session_state.chat_history:
    last_two = st.session_state.chat_history[-2:]
    for i, msg in enumerate(last_two):
        if i % 2 == 0:
            st.markdown(f"""
            <div class="msg-user">
                <div>
                    <div class="bubble-user-label">You</div>
                    <div class="bubble-user">{msg.content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-bot">
                <div style="padding-left:42px">
                    <div class="bubble-bot-label">StudyMind</div>
                    <div class="bubble-bot">{msg.content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📖</div>
        <div class="empty-text">Upload a PDF and start asking questions<br>about your study material</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
