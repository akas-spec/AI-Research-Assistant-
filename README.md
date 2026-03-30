# 🔬 AI Research Assistant

An intelligent AI system that answers research queries using dynamic routing, retrieval-augmented generation (RAG), and multi-model reasoning.

---

## 🚀 Features

* **Smart Query Routing**: Automatically classifies queries and selects the best pipeline
* **Multiple Pipelines**:

  * Simple → Direct LLM response
  * Retrieval → RAG using vector database
  * Analytical → Deep reasoning with Gemini
  * Research → arXiv search + synthesis
* **RAG System**: Uses embeddings + ChromaDB for context-aware answers
* **Multi-LLM Support**: Groq (fast responses) + Gemini (advanced reasoning)
* **PDF Processing**: Upload and query your own research papers
* **Interactive UI**: Built using Streamlit

---

## 🧠 System Architecture

```id="7x9q2k"
User Query
   ↓
Query Router (classification)
   ↓
Pipeline Selection
   ↓
[LLM / RAG / arXiv Search]
   ↓
Final Response + Sources
```

---

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
