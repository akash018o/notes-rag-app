import os
import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from utils import chunk_text, decode_bytes

DISTANCE_THRESHOLD = 1.2  # lower = stricter; tune this using the debug panel below

st.set_page_config(page_title="My Notes Assistant")
st.title("Ask My Notes")


@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("notes")
    return model, collection


model, collection = load_resources()

# --- File upload ---
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

uploaded_files = st.sidebar.file_uploader("Add .txt documents", type=["txt"], accept_multiple_files=True)
new_files = [f for f in uploaded_files or [] if f.name not in st.session_state.processed_files]

for uploaded in new_files:
    content = decode_bytes(uploaded.read())
    os.makedirs("docs", exist_ok=True)
    with open(os.path.join("docs", uploaded.name), "w", encoding="utf-8") as f:
        f.write(content)  # also save so a future full rebuild includes it

    chunks = chunk_text(content)
    ids = [f"{uploaded.name}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": uploaded.name} for _ in chunks]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=model.encode(chunks).tolist())
    st.session_state.processed_files.add(uploaded.name)

if new_files:
    st.sidebar.success(f"Added {len(new_files)} file(s).")

st.caption(f"{collection.count()} chunks loaded")

# --- Chat ---
if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your documents")

if question:
    results = collection.query(
        query_embeddings=model.encode([question]).tolist(),
        n_results=3,
    )
    candidates = list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))
    kept = [(doc, meta) for doc, meta, dist in candidates if dist <= DISTANCE_THRESHOLD]

    if not kept:
        answer = "I don't know based on the documents (nothing close enough was found)."
        sources = []
    else:
        context = "\n".join(doc for doc, _ in kept)
        sources = sorted(set(meta["source"] for _, meta in kept))
        prompt = f"""You are a helpful assistant. Use the context below to answer the question.
The context may use different words than the question, so think about meaning.
Only if the context has nothing relevant, say "I don't know based on the documents."

Context:
{context}

Question: {question}
Answer:"""
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        answer = response["message"]["content"]

    st.session_state.history.append((question, answer, sources, candidates))

for q, a, sources, candidates in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        if sources:
            st.caption("Sources: " + ", ".join(sources))
        with st.expander("Debug: retrieval distances"):
            for doc, meta, dist in candidates:
                mark = "kept" if dist <= DISTANCE_THRESHOLD else "dropped"
                st.text(f"[{mark}] {dist:.3f}  {meta['source']}  {doc[:80]}...")