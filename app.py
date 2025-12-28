"""
Gradio Demo App for Drug Discovery Model
Predicts drug approval likelihood from SMILES molecular strings

Usage:
    pip install gradio transformers torch
    python app.py
"""

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model configuration
MODEL_ID = "kprsnt/drug-discovery-qwen-14b"

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded!")

# Example SMILES strings for common drugs
EXAMPLES = [
    ["CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"],
    ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen"],
    ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"],
    ["CC(=O)NC1=CC=C(C=C1)O", "Paracetamol"],
    ["CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "Testosterone"],
]


def predict_drug(smiles: str) -> dict:
    """Predict drug approval likelihood from SMILES."""
    if not smiles or not smiles.strip():
        return {"Error": "Please enter a SMILES string"}
    
    try:
        # Tokenize
        inputs = tokenizer(
            smiles.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        failed_prob = probs[0].item()
        approved_prob = probs[1].item()
        
        return {
            "❌ Failed": failed_prob,
            "✅ Approved": approved_prob
        }
    
    except Exception as e:
        return {"Error": str(e)}


# Create Gradio interface
with gr.Blocks(
    title="Drug Discovery AI",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple"
    )
) as demo:
    
    gr.Markdown("""
    # 🧬 Drug Discovery AI
    ### Predict Drug Approval Likelihood from SMILES Molecular Strings
    
    This model is a fine-tuned **Qwen 2.5 14B** for drug discovery. 
    Enter a SMILES string to predict whether the drug is likely to be approved or fail in clinical trials.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            smiles_input = gr.Textbox(
                label="SMILES String",
                placeholder="Enter SMILES molecular string (e.g., CC(=O)OC1=CC=CC=C1C(=O)O for Aspirin)",
                lines=2
            )
            
            predict_btn = gr.Button("🔬 Predict", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output = gr.Label(
                label="Prediction",
                num_top_classes=2
            )
    
    gr.Markdown("### Example Molecules")
    
    with gr.Row():
        for smiles, name in EXAMPLES[:5]:
            gr.Button(f"{name}", size="sm").click(
                lambda s=smiles: s,
                outputs=smiles_input
            )
    
    # Connect prediction
    predict_btn.click(
        fn=predict_drug,
        inputs=smiles_input,
        outputs=output
    )
    
    smiles_input.submit(
        fn=predict_drug,
        inputs=smiles_input,
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### About
    - **Model:** [kprsnt/drug-discovery-qwen-14b](https://huggingface.co/kprsnt/drug-discovery-qwen-14b)
    - **Base Model:** Qwen 2.5 14B
    - **Task:** Binary classification (Drug Approval Prediction)
    - **Training:** Fine-tuned on AMD MI300X with 4-bit quantization
    
    ⚠️ **Disclaimer:** This is a demo model. Predictions should not replace expert judgment in drug development.
    """)


if __name__ == "__main__":
    demo.launch(share=True)
