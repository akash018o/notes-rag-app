from fileinput import filename
import shutil
import os
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MAX_CHARS = 500
PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "docs"


def chunk_text(text, max_chars=MAX_CHARS):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = [current[-1]]  # carry the last sentence over as overlap
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


# 1. Read and chunk all files first
ids, docs, metadatas = [], [], []
for path in sorted(DOCS_DIR.iterdir()):
    if path.is_file() and path.suffix.lower() == ".txt":
        with open(os.path.join("docs", filename), encoding="utf-8") as f:
            text = f.read()
        file_chunks = chunk_text(text)
        print(f"{path.name}: {len(file_chunks)} chunks")
        for i, chunk in enumerate(file_chunks):
            ids.append(f"{path.name}-{i}")
            docs.append(chunk)
            metadatas.append({"source": path.name})

if not docs:
    raise SystemExit("No .txt files found in the docs folder, nothing stored.")

# 2. Only now replace the old data
database_path = PROJECT_DIR / "chroma_db"
shutil.rmtree(database_path, ignore_errors=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(database_path))
collection = client.get_or_create_collection("notes")

collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=model.encode(docs).tolist())
print(f"Stored {collection.count()} chunks from the docs folder")