# 🔬 AI Research Assistant

An advanced **RAG-based AI system** that enables academic research, document analysis, and real-time AI responses using live arXiv papers and uploaded PDFs.

---

## 🚀 Features

### 🧠 Multi-Mode Query System

- **📊 Analysis Mode** → Compare uploaded PDFs using Retrieval + Gemini  
- **🔬 Research Mode** → Full pipeline:
- Query → arXiv Search → PDF Download → Chunking → Temp Vector DB → Retrieval → LLM
- 
- **⚡ Simple Mode** → Fast responses via Groq (Llama 3.1)

---

### 🔍 Retrieval-Augmented Generation (RAG)

- Context-aware answers using:
- Uploaded documents
- Real-time arXiv research papers
- Semantic retrieval using embeddings
- Source attribution with:
- Paper titles
- arXiv links
- Relevance scores

---

### 📄 Document Processing

- Upload research papers (PDF)
- Automatic chunking & embedding
- Stored in ChromaDB vector database

---

### ⚡ Optimized Research Pipeline (NEW 🔥)

- Top-2 relevant papers selection
- Limited PDF parsing (first ~10–12 pages)
- Parallel PDF processing
- Temporary vector DB per query (no data pollution)
- Caching for faster repeated queries

---

### 🤖 Multi-Model Orchestration

- **Groq (Llama 3.1)** → fast responses
- **Gemini Pro** → document analysis
- **Gemini Advanced** → deep research synthesis

---

## 🏗️ Architecture
User Query
↓
Mode Selection (UI)
↓
Router
↓
[Analysis] → Local Retrieval → Gemini
[Research] → arXiv → PDF → Chunk → Temp DB → Retrieval → Gemini
[Simple] → Groq
↓
Final Answer + Sources


---
## 🧪 Tech Stack

- Python
- Streamlit
- ChromaDB (Vector Database)
- HuggingFace Embeddings
- arXiv API
- Groq API (Llama 3.1)
- Google Gemini API

---

## ⚡ Key Improvements

- Removed unreliable auto-routing → replaced with user-controlled modes
- Implemented optimized arXiv RAG pipeline
- Added caching and parallel processing
- Reduced latency by limiting PDFs and chunking
- Improved source attribution with real paper links
- Clean and structured UI with expandable sources

---

## 📸 Demo
  
## 📸 Demo

### 📝 Answer Output
![Answer Output](working_demo/main.png)

### 📚 Sources with Citations
![Sources](working_demo/sources.png)

---

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-research-assistant.git
cd ai-research-assistant
```
### 2.installing dependencies
  pip install -r requirements.txt
### 3.Set up environment variables
 GROQ_API_KEY=your_key
 GEMINI_API_KEY=your_key
### 4.Run the application
streamlit run app/main.py

📖 Usage
  Upload Papers
  Go to Upload Papers
  Upload PDF
  Process into vector DB
  Ask Questions
  Enter query
  Select mode:
📊 Analysis (your docs)
🔬 Research (arXiv)
⚡ Simple (fast answer)

📌 Example Queries
1. What is a transformer?
2. Compare BERT and GPT architectures
3. Latest research on RAG systems
4. Explain multimodal learning advancements


🧠 Author
   AKASH

## 📖 Usage

### Upload Papers

* Go to **Upload Papers** tab
* Upload a PDF
* Process and store in vector database

### Ask Questions

* Enter your query
* System automatically:

  * Classifies query
  * Selects pipeline
  * Returns answer with sources

---

## 📌 Example Queries

* What is a transformer?
* Compare BERT and GPT architectures
* Find papers on retrieval-augmented generation
* State of research on multimodal learning

---

## 📄 License

MIT License
=======

## 🧪 Tech Stack

* Python, Streamlit
* ChromaDB (Vector DB)
* Groq API (Llama 3.1)
* Google Gemini API
* arXiv API

---

## ✨ Recent Improvements

* ❌ Removed unreliable smart routing
* ✅ Added explicit user-controlled modes
* ✅ Implemented streaming responses
* ✅ Added stop-response functionality
* ✅ Improved system reliability & UX

---

## 📌 Future Work

* Streaming with full RAG pipeline
* Chat history memory
* PDF-specific Q&A mode
* Deployment (Docker / Cloud)

---

## ⚙️ Setup

```bash
git clone <repo>
cd AI-RESEARCH-ASSISTANT
pip install -r requirements.txt
streamlit run app/main.py
```

---

## 🧠 Author

Akash
