import json
from sentence_transformers import SentenceTransformer

print("Loading chunks...")
with open('data/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f" Loaded {len(chunks)} chunks")

print("Loading Hugging Face Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding chunks (this will take a few minutes)...")
texts = [chunk['text'] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

print(f" Embeddings created: {embeddings.shape}")

for i, chunk in enumerate(chunks):
    chunk['embedding'] = embeddings[i].tolist()

print("Saving embedded chunks...")
with open('data/chunks_embedded.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False)

print(f" Saved {len(chunks)} embedded chunks!")