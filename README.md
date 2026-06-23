# 🧠 Conversational RAG Assistant with Gemma

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)

A highly advanced, full-stack **Retrieval-Augmented Generation (RAG)** application demonstrating the core capabilities of large language models (LLMs), semantic search, and stateful conversational memory.

This project was built to showcase a deep, practical understanding of the modern AI engineering stack, specifically leveraging **LangChain** and **Transformer** architectures to build autonomous, context-aware AI assistants.

## 🌟 Key Technical Features

1. **Retrieval-Augmented Generation (RAG):**
   - Implements semantic search using **HuggingFace Transformers** (`all-MiniLM-L6-v2`) to embed document chunks.
   - Utilizes **FAISS** (Facebook AI Similarity Search) for blazingly fast, high-dimensional vector retrieval.
   - Accurately grounds the LLM's responses in user-uploaded PDF data, virtually eliminating hallucinations.

2. **Advanced LangChain Architectures:**
   - Designed using modern **LCEL (LangChain Expression Language)** for robust and streaming-friendly data pipelines.
   - Implements a **History-Aware Retriever**: The system dynamically rephrases user queries based on the chat history to maintain strict contextual relevance across multi-turn conversations.

3. **Intelligent Fallback Mechanism:**
   - Features a dynamic routing system: if the retrieved context does not contain the answer, the LLM seamlessly falls back on its own pre-trained knowledge base to assist the user.

4. **Stateful Streamlit UI:**
   - A fully interactive frontend built with Streamlit, featuring real-time conversational memory, session state management, and an intuitive document processing pipeline.

## 🛠️ Technology Stack

- **Framework:** LangChain (Core, Community, Classic)
- **LLM:** Google Gemma (via Ollama)
- **Embeddings:** HuggingFace Sentence Transformers
- **Vector Database:** FAISS
- **Frontend:** Streamlit
- **Document Processing:** PyPDF

## 🚀 Getting Started

### Prerequisites
1. Install Python 3.9+
2. Install [Ollama](https://ollama.com/) and pull the Gemma model:
   ```bash
   ollama pull gemma:2b
   ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/conversational-rag-gemma.git
   cd conversational-rag-gemma
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🏗️ System Architecture Flow
1. **Document Ingestion:** User uploads a PDF. `PyPDFLoader` extracts the text.
2. **Chunking & Embedding:** The text is split using `RecursiveCharacterTextSplitter` and embedded using HuggingFace Transformers.
3. **Storage:** The high-dimensional vectors are stored in a local FAISS index.
4. **Query Processing:** When a user asks a question, the `history_aware_retriever` combines the chat history and new question to form a standalone query.
5. **Generation:** The context is pulled from FAISS and fed into the Gemma LLM alongside the strict prompt instructions. The answer is streamed back to the UI.
