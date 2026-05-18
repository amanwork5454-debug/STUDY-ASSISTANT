# 🧠 StudyMind AI — Intelligent Study Assistant

> **GenAI + RAG powered app** that lets you chat with your PDFs — textbooks, notes, research papers — using a 100% free AI stack.

[![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.6-green?style=flat-square)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA--3.3--70b-orange?style=flat-square)](https://groq.com)
[![Cohere](https://img.shields.io/badge/Embeddings-Cohere-blueviolet?style=flat-square)](https://cohere.com)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-red?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=flat-square)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=flat-square)](https://study-assistant-aman.streamlit.app)

---

## 🚀 Live Demo

👉 **[study-assistant-aman.streamlit.app](https://study-assistant-aman.streamlit.app)**

---

## 📸 Preview

| Feature | Description |
|---|---|
| 📂 Upload PDFs | Upload multiple textbooks, notes, or papers |
| 💬 Ask Questions | Natural language Q&A over your documents |
| 🧠 Smart Retrieval | Cohere semantic embeddings find the right content |
| 🔁 Memory | Multi-turn conversation — follow-up questions work |
| 🗑️ Clear Chat | Reset conversation while keeping documents indexed |
| 📊 Live Stats | See files, chunks, and questions count in real time |

---

## 🏗️ Architecture

```
📄 PDF Upload
      ↓
📖 PyPDF2  (text extraction)
      ↓
✂️  RecursiveCharacterTextSplitter  (chunk_size=500, overlap=50)
      ↓
🔢 Cohere Embeddings  (embed-english-light-v3.0)
      ↓
🗄️  FAISS Vector Store  (local, in-memory)
      ↓
🔍 Similarity Retrieval  (top-5 relevant chunks)
      ↓
🤖 Groq LLM  (llama-3.3-70b-versatile)
      ↓
💬 ConversationBufferMemory  (multi-turn chat)
      ↓
🖥️  Streamlit UI  (dark theme, chat bubbles)
```

---

## 💸 Free Stack — Total Cost ₹0

| Component | Tool | Cost |
|---|---|---|
| LLM | Groq — llama-3.3-70b-versatile | ✅ Free |
| Embeddings | Cohere — embed-english-light-v3.0 | ✅ Free (trial) |
| Vector DB | FAISS (local) | ✅ Free |
| UI | Streamlit | ✅ Free |
| Hosting | Streamlit Cloud | ✅ Free |

---

## ⚙️ Tech Stack

- **LangChain** — RAG pipeline, memory, retrieval chain
- **FAISS** — Local vector store for fast similarity search
- **Cohere** — Real semantic embeddings (no torch required)
- **Groq** — Ultra-fast LLM inference (llama-3.3-70b)
- **PyPDF2** — PDF text extraction
- **Streamlit** — Interactive web UI with custom dark theme

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/amanwork5454-debug/study-assistant.git
cd study-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
```

Get your free keys:
- Groq → [console.groq.com](https://console.groq.com)
- Cohere → [dashboard.cohere.com](https://dashboard.cohere.com)

### 4. Run the app
```bash
streamlit run app.py
```

Opens at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
study-assistant/
├── app.py                  # Main application — full RAG pipeline + UI
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores .env and venv
└── README.md               # You are here
```

---

## 💡 How to Use

1. **Upload** your PDF(s) using the sidebar
2. Click **⚡ Process Documents** — chunks and indexes your PDFs
3. **Ask anything** in the text box
4. Get instant answers from your study material
5. Ask **follow-up questions** — memory keeps context
6. Click **🗑️ Clear Conversation** to start fresh

---

## 👤 Author

**Aman** — [github.com/amanwork5454-debug](https://github.com/amanwork5454-debug)

---


