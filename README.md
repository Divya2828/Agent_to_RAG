# Business Model Generator + RAG QA (Streamlit)

A small Streamlit app that uses LangChain, OpenAI-compatible models, and a simple RAG (Retrieval-Augmented Generation) pipeline.

## 🚀 What it does

- **Business Model Generator**: Type a business idea and get a structured, document-style business model output.
- **RAG Q&A**: Paste the generated text, build a vector store, and ask questions about it.

## 📁 Project Structure

- `app.py` – Streamlit UI, tabs for Business Model Generator + RAG Q&A
- `agent.py` – LangChain agent that generates a business model from a prompt
- `rag.py` – Builds embeddings, stores them in an in-memory vector store, and answers user queries using a retrieval tool
- `requirements.txt` – Python dependencies
- `.env` – Environment variables (API keys + model settings)

## 🧰 Requirements

- Python 3.10+ (recommended)
- A working OpenAI-compatible API key + endpoint

## ✅ Setup

1. **Create / activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Create a `.env` file** (or update the existing one)

Example `.env`:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1

CHAT_MODEL=stepfun/step-3.5-flash:free
EMBEDDING_MODEL=openai/text-embedding-3-small
```

> ⚠️ Do not commit your `.env` file (esp. secrets) into source control.

## ▶️ Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## 🧩 How it works (high level)

### Business Model Generator
- `app.py` calls `run_agent()` from `agent.py`.
- `agent.py` creates a LangChain agent using `langchain_openai.ChatOpenAI` and two toy tools.
- The agent generates a structured business model response based on your input.

### RAG Q&A
- `app.py` sends the generated document to `rag.generate_embeddings()`.
- `rag.py` uses `OpenAIEmbeddings` to create embeddings and stores them in an in-memory vector store.
- A LangChain agent uses the `retrieve_context` tool to pull relevant chunks and answer questions.

## 🧠 Notes / Next steps

- The RAG system uses an in-memory store, so rebuilding is required each session.
- You can replace with a persistent vector store (e.g., Chroma, Pinecone) for long-term use.
- Consider adding prompt templates and better validation for improved UX.

---

© 2026