import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from generate import load_index_and_chunks, retrieve_relevant_chunks, generate_lab_record
from save_docx import save_as_docx

# --- Load model and index once when app starts ---
# We do this outside the function so it doesn't reload on every request
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index, chunks = load_index_and_chunks()

print("Ready!")

# --- Core function that Gradio will call ---
def create_lab_record(experiment_name, subject):
    if not experiment_name.strip():
        return None, "Please enter an experiment name."
    
    if not subject.strip():
        return None, "Please enter a subject name."

    try:
        # Step 1 — retrieve relevant chunks
        results = retrieve_relevant_chunks(
            experiment_name, index, chunks, model
        )

        # Step 2 — generate lab record using Groq
        lab_record = generate_lab_record(
            experiment_name, subject, results
        )

        # Step 3 — save as docx
        os.makedirs("output", exist_ok=True)
        
        # clean filename — remove spaces and special characters
        safe_name = experiment_name.replace(" ", "_").replace("/", "_")
        output_path = f"output/{safe_name}_Lab_Record.docx"
        
        save_as_docx(lab_record, experiment_name, output_path)

        return output_path, "Lab record generated successfully!"

    except Exception as e:
        return None, f"Error: {str(e)}"


# --- Build the Gradio UI ---
with gr.Blocks(title="Lab Record Auto-Writer") as app:
    
    gr.Markdown("# Lab Record Auto-Writer")
    gr.Markdown("Enter your experiment details and get a complete lab record as a Word document.")
    
    with gr.Row():
        with gr.Column():
            experiment_input = gr.Textbox(
                label="Experiment Name",
                placeholder="e.g. Eigenvalues and Eigenvectors in Scilab",
                lines=2
            )
            subject_input = gr.Textbox(
                label="Subject Name",
                placeholder="e.g. SSMDA",
                lines=1
            )
            generate_btn = gr.Button("Generate Lab Record", variant="primary")

        with gr.Column():
            status_output = gr.Textbox(
                label="Status",
                interactive=False
            )
            file_output = gr.File(
                label="Download Lab Record"
            )

    # connect button to function
    generate_btn.click(
        fn=create_lab_record,
        inputs=[experiment_input, subject_input],
        outputs=[file_output, status_output]
    )

    gr.Markdown("### Try these experiments:")
    gr.Markdown("""
- Matrix Operations in Scilab  
- Eigenvalues and Eigenvectors in Scilab  
- Gauss Elimination Method  
- Correlation using Scilab  
- Frequency Table in SPSS  
    """)

if __name__ == "__main__":
    app.launch(share=True)