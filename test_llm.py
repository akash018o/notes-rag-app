import ollama

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Give me one interesting fact about black holes."}],
)
print(response["message"]["content"])