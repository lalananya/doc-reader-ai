import faiss
import numpy as np
# from PyPDF2 import PdfReader
import ollama

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def load_pdf(file_path):
    # reader = PdfReader(file_path)
    # text = ""
    # for page in reader.pages:
    #     text += page.extract_text()
    # return text
    return extract_text(file_path)


def chunk_text(text, chunk_size=500, overlap = 100):
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
        return chunks


def get_embeddings(chunks):
    return embedding_model.encode(chunks)


def create_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def rerank_chunks(query, chunks):
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, chunk_vectors)[0]
    ranked_indices = scores.argsort()[::-1]
    return [chunks[i] for i in ranked_indices]

def ask_question(query, chunks, index):
    question_embedding = embedding_model.encode([query])[0]

    distances, indices = index.search(
        np.array([question_embedding]).astype("float32"), k=5
    )
    retrieved_chunks = [chunks[i] for i in indices[0]]
    ranked_chunks = rerank_chunks(query, retrieved_chunks)
    top_chunks = ranked_chunks[:3]
    context = "\n\n".join(top_chunks)
    response = ollama.chat(
        model='llama3',
        messages=[
            {   "role": "system", "content": (
                "You are precise assistantanswering questions from a PDF.\n" 
                "Use ONLY the provided context.\n "
                "If the answer is not in the context, say I dont know. \n"
                "Do not guess \n"
                "Keep the answers the short and factual \n")
            },
            {   "role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )

    return response['message']['content']