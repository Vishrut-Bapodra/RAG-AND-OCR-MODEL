# RAG Model — Local Retrieval-Augmented Generation System

This project implements a **local Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload documents and query them using a large language model, with answers grounded strictly in the uploaded content.

The system is designed to handle **real-world documents**, including scanned PDFs, and emphasizes **controlled, context-only generation** rather than open-ended LLM responses.

---

## 🔍 Core Capabilities

### 1. Multi-Format Document Ingestion
- Supports **PDF, TXT, DOCX, and PPTX** files
- Files are stored locally with unique identifiers
- Multiple documents can be uploaded and indexed together

---

### 2. Robust PDF Text Extraction
- Uses a **hybrid extraction strategy**:
  - Digital text extraction via PyPDF2
  - **OCR fallback using PyMuPDF + Tesseract** for scanned PDFs
- Supports multilingual OCR (English + Gujarati)

---

### 3. Sentence-Aware Chunking
- Documents are split using **sentence boundaries**
- Chunk size capped to maintain semantic coherence
- Overlapping sentences ensure context continuity across chunks

---

### 4. Vector-Based Semantic Search
- Uses **SentenceTransformers (multilingual-e5-base)** for embeddings
- Embeddings are **L2-normalized** for cosine similarity
- Stored in a **FAISS IndexFlatIP** vector index
- Lazy-loaded model and index to optimize memory usage

---

### 5. Context-Grounded Answer Generation
- User queries retrieve top-k relevant chunks
- Retrieved chunks are injected as structured context
- LLM is explicitly instructed to:
  - Use **only the provided context**
  - Avoid hallucination or external knowledge

---

### 6. LLM Integration via OpenRouter
- Uses **OpenRouter API** for model access
- Default model: `openrouter/free`
- Secure API key handling via environment variables
- Configurable temperature and retrieval depth

---

### 7. Interactive Streamlit Chat Interface
- Real-time chat UI built with Streamlit
- Session-based chat memory
- Styled user/assistant messages
- Document upload and indexing from sidebar

---

## 🛠️ Tech Stack

- Python
- Streamlit (UI)
- SentenceTransformers
- FAISS
- PyPDF2, PyMuPDF, Tesseract OCR
- OpenRouter API
- FAISS Vector Store

---

## ▶️ Setup & Usage

1. Clone the repository
   ```bash
   git clone <repo-url>
   cd rag-model

2. Install dependencies
   ```bash
    pip install -r requirements.txt
   
3. Set OpenRouter API key
    ```bash
    export OPENROUTER_API_KEY="your_api_key"

4. Run the app
    ```bash
    streamlit run app.py
