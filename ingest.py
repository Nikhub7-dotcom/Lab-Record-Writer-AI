from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks

def get_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def build_and_save_index(chunks, model):
    print("Creating embeddings for all chunks...")
    vectors = model.encode(chunks, show_progress_bar=True)
    
    # convert to float32 — FAISS requires this specific number format
    vectors = np.array(vectors).astype("float32")
    
    # create FAISS index
    dimension = vectors.shape[1]  # 384 for our model
    index = faiss.IndexFlatL2(dimension)
    
    # add all vectors into the index
    index.add(vectors)
    
    # save the index file to disk
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")
    
    # save the chunks separately so we can retrieve original text later
    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Done! Saved {index.ntotal} vectors to faiss_index/")

if __name__ == "__main__":
    # Step 1 — load
    text = load_pdf_text("data/SSMDA_Lab.pdf")
    print(f"Extracted {len(text)} characters from PDF")
    
    # Step 2 — chunk
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3 — load model
    model = get_embedding_model()
    
    # Step 4 — build and save index
    build_and_save_index(chunks, model)