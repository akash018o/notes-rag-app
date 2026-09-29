from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent

print("Loading models...")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(PROJECT_DIR / "chroma_db"))
collection = client.get_or_create_collection("notes")
print(f"Ready. {collection.count()} chunks loaded. Type 'exit' to quit.\n")

while True:
    question = input("Ask a question: ")
    if question.lower() in ("exit", "quit"):
        break

    results = collection.query(
        query_embeddings=model.encode([question]).tolist(),
        n_results=2,
    )

    documents = results.get("documents", [[]])[0]
    metadata = results.get("metadatas", [[]])[0]

    if not documents:
        print("I don't know based on the documents.\n")
        continue

    context = "\n".join(documents)
    sources = sorted(set(m["source"] for m in metadata))

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

    print("\nSources:", ", ".join(sources))
    print(response["message"]["content"] + "\n")