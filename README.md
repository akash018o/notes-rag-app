# Ask My Notes — Local RAG App

A fully local, offline question-answering app that reads your own documents and answers questions grounded in them, with source citations. Built from scratch to learn how Retrieval-Augmented Generation (RAG) actually works, one layer at a time.

No API keys, no cloud costs, no data leaving your machine.

## Features

- **Sentence-aware chunking** — documents are split at sentence boundaries (not blind character cuts), with overlap so context isn't lost at chunk edges
- **Semantic search** — finds relevant text by meaning, not just keyword matching
- **Source citations** — every answer shows which file(s) it came from
- **Relevance filtering** — drops weak/irrelevant matches before they ever reach the LLM
- **File upload** — add `.txt` documents directly from the browser, with automatic encoding detection (handles UTF-8, UTF-16, etc.)
- **Chat interface** — multi-turn conversation with history, built with Streamlit
- **Debug panel** — inspect retrieval distances for every answer, to see exactly what was retrieved and why

## How it works

```
Your documents (.txt)
        │
        ▼
 1. Chunk        →  split into ~500-character, sentence-aligned pieces
        │
        ▼
 2. Embed        →  each chunk becomes a 384-number vector (sentence-transformers)
        │
        ▼
 3. Store        →  vectors saved in a local Chroma database
        │
        ▼
   [ user asks a question ]
        │
        ▼
 4. Retrieve     →  question is embedded, closest chunks are found by vector similarity
        │
        ▼
 5. Filter       →  chunks with weak similarity are discarded
        │
        ▼
 6. Augment      →  surviving chunks are inserted into a prompt as context
        │
        ▼
 7. Generate     →  a local LLM (Llama 3.1 8B via Ollama) answers using only that context
        │
        ▼
   Answer + source file names
```

If no retrieved chunk is relevant enough, the app says so instead of guessing — this is the main defense against hallucination.

## Tech stack

| Component | Tool |
| --- | --- |
| LLM (generation) | [Ollama](https://ollama.com) running Llama 3.1 8B, fully local |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector database | [Chroma](https://www.trychroma.com/) |
| Interface | [Streamlit](https://streamlit.io/) |
| Language | Python 3.10+ |

## Setup

1. **Install [Ollama](https://ollama.com/download)** and pull the model:

   ```bash
   ollama pull llama3.1:8b
   ```

2. **Clone this repo and set up a virtual environment:**

   ```bash
   git clone <your-repo-url>
   cd rag-app
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your documents** to the `docs/` folder (`.txt` files), then build the vector database:

   ```bash
   python ingest.py
   ```

5. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   It opens at `http://localhost:8501`. You can also upload additional `.txt` files directly from the sidebar.

## Project structure

```
rag-app/
├── docs/              # your source documents (.txt)
├── chroma_db/         # vector database (auto-generated, gitignored)
├── utils.py           # shared chunking + encoding-detection logic
├── ingest.py          # bulk-loads docs/ into the vector database
├── app.py             # the Streamlit chat application
├── requirements.txt
└── README.md
```

## Known limitations

This was built to learn the fundamentals, not to be production software:

- No authentication — anyone with access to the running app can use it
- Retrieval quality depends on chunk size/overlap tuning; no re-ranking step yet
- Local 8B models are less capable than large hosted models
- Only `.txt` files are supported (no PDF/DOCX parsing yet)
- Not tested at scale with large document collections

## What I learned building this

Tokens and how LLMs generate text, why hallucination happens, embeddings and cosine similarity, why chunking strategy matters, vector databases, prompt design for grounding model answers, and how to debug a multi-stage pipeline by checking each stage (retrieval vs. generation) independently.
