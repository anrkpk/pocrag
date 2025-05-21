# 🤖 Customer Care AI Assistant - RAG Implementation

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **LLMs**, and **Streamlit** for building an AI assistant that can answer user questions based on uploaded documents and web sources.

---

## 📂 Project Structure

src/
├── pages/
│ ├── chatbot.py # Streamlit chatbot UI
│ ├── upload.py # Upload interface for PDFs/URLs
│
├── utils/embeddings/
│ ├── chunking.py # Chunking strategies (recursive, token-based, etc.)
│ ├── generator.py # LLM generation logic
│ ├── retriever.py # Document retrieval from vector store
│ ├── vector_store.py # FAISS or Chroma vector DB handling
│ └── init.py
├── main.py # Streamlit app entry point
├── .env # API keys and environment variables
├── requirements.txt # Python dependencies
├── Dockerfile # Docker image definition
├── docker-compose.yml # For container orchestration
├── README.md # You are here!

## 🧠 RAG Pipeline Overview

```text
Document Upload → Chunking → Embedding → Vector DB
           ↓
     User Query → Retrieve → LLM → Answer

Document Ingestion: PDFs and URLs are uploaded and split into manageable chunks.

Embedding: Documents are converted into vectors using embedding models.

Vector Store: Stored in FAISS/Chroma for fast retrieval.

Retriever: Finds relevant chunks based on user queries.

Generator: Uses an LLM (like ChatGroq) to generate answers.


1.Clone the Repository :
https://github.com/anrkpk/pocrag/tree/phase2

2. Setup Environment

.emv
GROQ_API_KEY=your-groq-api-key

3. Install Python Dependencies
pip install -r requirements.txt


4. Run the Streamlit App
cd src
streamlit run main.py

5. Dockerized Setup (Optional)
docker build -t rag-assistant .

6. Run Using Docker Compose
docker-compose up
