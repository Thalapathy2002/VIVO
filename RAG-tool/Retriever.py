import faiss
import numpy as np
from embeddings import get_embedding
from github_loader import load_catalog_from_github


docs = load_catalog_from_github()


corpus_embeddings = np.array([get_embedding(doc["content"]) for doc in docs])
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

def get_answer(question):
    q_embedding = np.array([get_embedding(question)])
    D, I = index.search(q_embedding, k=1)
    doc = docs[I[0][0]]
    return f"Based on our catalog: {doc['content']}"