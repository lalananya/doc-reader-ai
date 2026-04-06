import faiss
import numpy as np
from PyPDF2 import PdfReader
import ollama
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def get_embeddings(chunks):
    return embedding_model.encode(chunks)


def create_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index


def ask_question(query, chunks, index):
    question_embedding = embedding_model.encode([query])[0]

    distances, indices = index.search(
        np.array([question_embedding]).astype("float32"), k=3
    )
    context = [chunks[i] for i in indices[0]]

    response = ollama.chat(
        model='llama3',
        messages=[
            {"role": "system", "content": "You are answering questions about a book. Use ONLY the given context. If unsure, say I dont know. Be precise."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )

    return response['message']['content']