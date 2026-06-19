import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from save_docx import save_as_docx



load_dotenv()

def load_index_and_chunks():
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve_relevant_chunks(query, index, chunks, model, top_k=5):
    # convert the query into a vector
    query_vector = model.encode([query]).astype("float32")
    
    # search FAISS for top_k most similar vectors
    distances, indices = index.search(query_vector, top_k)
    
    # fetch the actual text chunks using the indices
    results = [chunks[i] for i in indices[0]]
    return results

def generate_lab_record(experiment_name, subject, retrieved_chunks):
    # join all retrieved chunks into one context block
    context = "\n\n".join(retrieved_chunks)

    # initialize the Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3  # low temperature = more focused, less random output
    )

    # system message tells the LLM what role to play
    system_message = SystemMessage(content="""You are an expert academic assistant 
that writes professional lab records for engineering students. 
You write in a clear, structured format suitable for university submissions.
Always include: AIM, THEORY, PROCEDURE, CODE, OUTPUT DESCRIPTION, and CONCLUSION.""")

    # human message is the actual request with context injected
    human_message = HumanMessage(content=f"""
Using the following reference material from existing lab records:

{context}

Write a complete and detailed lab record for the following:

Subject: {subject}
Experiment: {experiment_name}

Format it with these sections:
1. AIM
2. THEORY (explain the concept clearly, minimum 150 words)
3. PROCEDURE (step by step)
4. CODE (write working code)
5. EXPECTED OUTPUT (describe what the output looks like)
6. CONCLUSION (2-3 lines summarizing what was demonstrated)
""")

    print("Calling Groq LLM...")
    response = llm.invoke([system_message, human_message])
    return response.content


#Quick test
if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_index_and_chunks()

    experiment = "Eigenvalues and Eigenvectors in Scilab"
    subject = "SSMDA (Statistical and Scientific Methods for Data Analysis)"

    print(f"Retrieving context for: {experiment}")
    results = retrieve_relevant_chunks(experiment, index, chunks, model)

    print("Generating lab record...")
    lab_record = generate_lab_record(experiment, subject, results)

    # save to docx
    os.makedirs("output", exist_ok=True)
    output_file = "output/Eigenvalues_Lab_Record.docx"
    save_as_docx(lab_record, experiment, output_file)

    print("\nDone! Check your output/ folder.")