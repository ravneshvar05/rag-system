---
title: Multimodal RAG System
emoji: 🏢
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
---

# 🤖 Multimodal RAG System

A powerful **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **Streamlit**, and **LangChain**. This application allows users to upload various document types (PDF, DOCX, TXT, CSV, MD, ZIP), ingest them into a vector database, and ask complex questions using **Llama-3.3-70B** (via Groq) and **Hugging Face** embeddings.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-🦜️🔗-green)

## ✨ Features

- **📂 Multi-Format Ingestion**: Supports PDF, DOCX, TXT, CSV, Markdown, and Images.
- **📦 ZIP Archive Support**: Automatically extracts and processes files from uploaded ZIPs.
- **🔍 Vector Search**: Uses **FAISS** for efficient similarity search and retrieval.
- **🧠 Advanced LLM**: Powered by **Llama-3.3-70B** via Groq for high-quality answers.
- **💬 Interactive Chat UI**: Clean, responsive Streamlit interface with chat history.
- **🚀 FASTApi Backend**: Robust backend for handling ingestion and query processing.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Orchestration**: LangChain
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace (`sentence-transformers`)
- **LLM**: Groq API (Llama-3.3-70B)
- **PDF Processing**: PyMuPDF, pdfplumber

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [Groq API Key](https://console.groq.com/)
- [Hugging Face Token](https://huggingface.co/settings/tokens)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multimodal-rag-system.git
   cd multimodal-rag-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory (copy from `.env.example` if available):
   ```bash
   cp .exampleenv .env
   ```
   Add your API keys:
   ```ini
   HF_TOKEN=your_huggingface_token
   GROQ_API_KEY=your_groq_api_key
   ```

## 🏃‍♂️ Usage

### 1. Start the Backend API
Run the FastAPI server to handle ingestion and retrieval logic.
```bash
uvicorn app:app --reload
```
*Server will start at `http://127.0.0.1:8000`*

### 2. Launch the User Interface
Open a new terminal and run the Streamlit app.
```bash
streamlit run ui.py
```
*The UI will open in your browser at `http://localhost:8501`*

## 📁 Project Structure

```
multimodal-rag-system/
├── app.py                 # FastAPI backend entry point
├── ui.py                  # Streamlit frontend application
├── src/                   # Source code modules
│   ├── ingest.py          # File loading and processing logic
│   ├── vector_store.py    # FAISS vector database management
│   ├── rag.py             # RAG pipeline and LLM interaction
│   ├── config.py          # Configuration settings
│   └── logger.py          # Logging setup
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (do not commit)
└── README.md              # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
