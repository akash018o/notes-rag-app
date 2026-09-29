from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, free, runs locally

query = "How do I get my money back?"
docs = [
    "Our refund policy allows returns within 30 days of purchase.",
    "The cat sat on the sofa and watched the rain.",
    "You can reset your password from the account settings page.",
]

query_vec = model.encode(query)
doc_vecs = model.encode(docs)

print("One embedding has", len(query_vec), "numbers")
print("First 5 numbers:", query_vec[:5])

scores = util.cos_sim(query_vec, doc_vecs)[0]
for doc, score in zip(docs, scores):
    print(f"{float(score):.3f}  {doc}")