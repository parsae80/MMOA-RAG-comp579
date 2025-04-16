import faiss
import numpy as np

# Load FAISS index
index = faiss.read_index("wikipedia.contriever")

# Load passages from TSV
with open("psgs_w100.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Show first few vectors and corresponding passages
for i in range(min(3, index.ntotal)):
    vector = index.reconstruct(i)
    passage = lines[i].strip()
    print(f"🔢 Vector {i} (len={len(vector)}):")
    print(f"📖 Passage: {passage}")
    print("-" * 50)

