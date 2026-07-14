from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded!")

embedding = model.encode("Online Banking transfer")

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])