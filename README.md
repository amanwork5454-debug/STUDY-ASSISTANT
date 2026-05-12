# 📚 StudyMind — AI Study Assistant (GenAI + RAG)

A RAG-powered study assistant that lets you chat with your PDFs (textbooks, notes, research papers) using a 100% FREE AI stack.

## 🆓 Free Stack
| Component | Tool | Cost |
|-----------|------|------|
| LLM | Groq (llama-3-8b) | Free |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free (local) |
| Vector DB | FAISS | Free (local) |
| UI | Streamlit | Free |
| Hosting | Streamlit Cloud | Free |

## 🚀 Setup & Run

### 1. Clone / download the project
```bash
git clone <your-repo>
cd study_assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your FREE Groq API key
- Go to https://console.groq.com
- Sign up (free)
- Create an API key
- Paste it in `.env` file:
```
GROQ_API_KEY=your_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

## 🏗️ Architecture

```
PDF Upload
    ↓
PyPDF2 (text extraction)
    ↓
RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2) — runs locally
    ↓
FAISS Vector Store
    ↓
MMR Retrieval (top-5 diverse chunks)
    ↓
Groq LLM (llama-3-8b) — free API
    ↓
ConversationBufferMemory (multi-turn chat)
    ↓
Streamlit UI (chat interface)
```

## 💡 Features
- Upload multiple PDFs at once
- Ask natural language questions
- Multi-turn conversation with memory
- MMR retrieval for diverse, non-repetitive answers
- Sample question prompts in sidebar

## 📝 Resume Bullets
- Built a **multi-PDF RAG Study Assistant** using LangChain, FAISS, and Groq (llama-3); implemented MMR-based semantic retrieval over chunked academic documents with conversational memory.
- Designed end-to-end RAG ingestion pipeline — PDF parsing, recursive text chunking (500 tokens, 50 overlap), HuggingFace sentence-transformer embeddings, and FAISS vector indexing with zero API cost.
- Deployed interactive Streamlit UI supporting multi-turn Q&A over uploaded study materials using ConversationBufferMemory and ConversationalRetrievalChain.

## 🔧 Possible Extensions
- Add RAGAS evaluation (faithfulness, relevance scoring)
- Add page-level citations in answers
- Support DOCX, TXT files
- Add quiz/MCQ generation feature
- Deploy on Streamlit Cloud (free hosting)
