# 🔬 AI Research Assistant


An advanced **RAG-based AI system** that supports document analysis, academic research, and real-time streaming responses with user-controlled interruption.


---

## 🚀 Features

### 🧠 Multi-Mode Query System

* **📊 Analysis Mode** → Compare uploaded PDFs using Retrieval + Gemini
* **🔬 Research Mode** → Full pipeline:

  ```
  Query → arXiv Search → Retriever → Context → LLM
  ```
* **⚡ Simple Mode** → Fast responses via Groq (Llama 3.1)

---

### ⚡ Real-Time Streaming + Stop Control (NEW 🔥)

* Token-by-token response streaming
* ⛔ Stop response mid-generation
* Partial output retained
* ChatGPT-like interactive experience

---

### 📄 Document Processing

* Upload research papers (PDF)
* Automatic chunking & embeddings
* Stored in vector DB (ChromaDB)

---

### 🔍 Retrieval-Augmented Generation (RAG)

* Context-aware answers from:

  * Uploaded documents
  * arXiv research papers
* Source attribution with relevance scores

---

### 🤖 Multi-Model Orchestration

* **Groq (Llama 3.1)** → fast responses & streaming
* **Gemini Pro** → document analysis
* **Gemini Advanced** → deep research synthesis

---

## 🏗️ Architecture

```
User Query
   ↓
Mode Selection (UI buttons)
   ↓
Router
   ↓
[Analysis] → Retriever → Gemini  
[Research] → arXiv → Retriever → Context → LLM  
[Simple] → Groq  
   ↓
Streaming Output (with Stop Control)
```

---

<<<<<<< HEAD
## ⚙️ Tech Stack

* Python
* Streamlit
* ChromaDB (Vector Database)
* HuggingFace Embeddings
* Groq API
* Google Gemini API

---

## ▶️ Getting Started

### 1. Clone the repository

```id="z4h8qm"
git clone https://github.com/YOUR_USERNAME/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Install dependencies

```id="p3k1vn"
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file:

```id="v6r2lb"
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
```

### 4. Run the application

```id="k8m1xc"
streamlit run app/main.py
```

Open in browser: http://localhost:8501

---

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
>>>>>>> 63c75e3 (feat: add streaming responses with stop control and remove smart routing)
