from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

database_path = Path(__file__).resolve().parent / "chroma_db"
client = chromadb.PersistentClient(path=str(database_path))
collection = client.get_or_create_collection("my_docs")

docs = [
    "Our refund policy allows returns within 30 days of purchase.",
    "The cat sat on the sofa and watched the rain.",
    "You can reset your password from the account settings page.",
    "Shipping usually takes 3 to 5 business days within India.",
    "To contact support, email help@example.com or call during office hours.",
    "Premium members get free delivery on all orders.",
]
ids = [f"doc{i}" for i in range(len(docs))]

collection.upsert(
    ids=ids,
    documents=docs,
    embeddings=model.encode(docs).tolist(),
)

query = "How long until my order arrives?"
results = collection.query(
    query_embeddings=model.encode([query]).tolist(),
    n_results=2,
)

for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"{dist:.3f}  {doc}")