# Lab Record Auto-Writer

An AI-powered tool that generates complete lab records for engineering students using RAG (Retrieval Augmented Generation).

## How it works
1. Enter your experiment name and subject
2. The system retrieves relevant context from indexed lab records
3. Groq's LLaMA 3.1 generates a complete structured lab record
4. Download the output as a Word document (.docx)

## Tech Stack
- LangChain — RAG pipeline
- FAISS — vector similarity search
- sentence-transformers — text embeddings (all-MiniLM-L6-v2)
- Groq API — LLM inference (LLaMA 3.1 8B)
- Gradio — web UI
- python-docx — Word document generation

## Built by
Nikhil Kumar — B.Tech CSE, MSIT GGSIPU