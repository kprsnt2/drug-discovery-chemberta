"""
Gradio Demo App for Drug Discovery Text Generation Model

Multi-mode interface for drug discovery assistance:
- Drug Analysis: Predict approval likelihood with explanations
- Failure Analysis: Understand why drugs failed
- Drug Comparison: Compare two drug candidates
- Improvement Suggestions: Get structural modification recommendations
- Free Chat: Open-ended drug discovery discussions

Usage:
    pip install gradio transformers torch
    python app.py
"""

import os
import sys
import torch
import gradio as gr
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import MODEL_CONFIG, GENERATION_CONFIG, CHECKPOINT_DIR

# Try to import model
try:
    from src.model import DrugDiscoveryLLM
except ImportError as e:
    print(f"Warning: Could not import DrugDiscoveryLLM: {e}")
    DrugDiscoveryLLM = None

# Global model instance
model = None

# Example SMILES for common drugs
APPROVED_EXAMPLES = [
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", "Pain/Inflammation"),
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen", "Pain/Inflammation"),
    ("CC(C)NCC(O)C1=CC=C(O)C(O)=C1", "Isoproterenol", "Cardiovascular"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", "CNS Stimulant"),
    ("CC1=C(C=C(C=C1)S(=O)(=O)N)NC(=O)C2=CC=CC=C2", "Tolbutamide", "Diabetes"),
]

FAILED_EXAMPLES = [
    ("CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3", "Rofecoxib (Vioxx)", "Cardiovascular toxicity"),
    ("CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4", "Terfenadine", "QT prolongation"),
    ("CC1=C(C2=CC=CC=C2O1)CC(C)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4", "Troglitazone", "Hepatotoxicity"),
    ("COC1=C(C=C(C=C1)C=CC(CC(CC(=O)O)O)O)C2=NC(=C(C=C2)F)C(C)C", "Cerivastatin", "Rhabdomyolysis"),
]


def load_model(model_path: str = None):
    """Load the drug discovery model."""
    global model
    
    if model is not None:
        return "Model already loaded!"
    
    try:
        # Look for fine-tuned model first
        if model_path is None:
            # Check for latest checkpoint
            checkpoints = list(CHECKPOINT_DIR.glob("run_*/final"))
            if checkpoints:
                model_path = str(sorted(checkpoints)[-1])
                print(f"Found fine-tuned model: {model_path}")
            else:
                model_path = MODEL_CONFIG['model_name']
                print(f"Using base model: {model_path}")
        
        print(f"Loading model from: {model_path}")
        model = DrugDiscoveryLLM(
            model_name=model_path,
            load_in_4bit=False,  # Not needed with 192GB VRAM
            use_flash_attention=True,
        )
        return f"Model loaded successfully from: {model_path}"
    except Exception as e:
        return f"Error loading model: {e}"


def analyze_drug(smiles: str, name: str = "", temperature: float = 0.7) -> str:
    """Analyze a drug candidate."""
    if model is None:
        return "⚠️ Model not loaded. Please click 'Load Model' first."
    
    if not smiles or not smiles.strip():
        return "⚠️ Please enter a SMILES string."
    
    try:
        prompt = f"Analyze this drug candidate and predict its approval likelihood:\nSMILES: {smiles.strip()}"
        if name:
            prompt += f"\nDrug Name: {name}"
        
        output = model.generate(prompt, temperature=temperature, max_new_tokens=1024)
        return output.generated_text
    except Exception as e:
        return f"❌ Error during analysis: {e}"


def analyze_failure(smiles: str, name: str, known_reason: str = "", temperature: float = 0.7) -> str:
    """Analyze why a drug failed."""
    if model is None:
        return "⚠️ Model not loaded. Please click 'Load Model' first."
    
    if not smiles or not smiles.strip():
        return "⚠️ Please enter a SMILES string."
    
    try:
        prompt = f"This drug failed in clinical development. Explain why:\nDrug: {name}\nSMILES: {smiles.strip()}"
        if known_reason:
            prompt += f"\nKnown issue: {known_reason}"
        
        output = model.generate(prompt, temperature=temperature, max_new_tokens=1024)
        return output.generated_text
    except Exception as e:
        return f"❌ Error during analysis: {e}"


def compare_drugs(smiles1: str, name1: str, smiles2: str, name2: str, temperature: float = 0.7) -> str:
    """Compare two drug candidates."""
    if model is None:
        return "⚠️ Model not loaded. Please click 'Load Model' first."
    
    if not smiles1.strip() or not smiles2.strip():
        return "⚠️ Please enter SMILES strings for both drugs."
    
    try:
        prompt = f"""Compare the safety profiles of these two drugs:
Drug 1: {smiles1.strip()} ({name1})
Drug 2: {smiles2.strip()} ({name2})"""
        
        output = model.generate(prompt, temperature=temperature, max_new_tokens=1024)
        return output.generated_text
    except Exception as e:
        return f"❌ Error during comparison: {e}"


def suggest_improvements(smiles: str, name: str, problem: str, temperature: float = 0.7) -> str:
    """Suggest structural improvements."""
    if model is None:
        return "⚠️ Model not loaded. Please click 'Load Model' first."
    
    if not smiles.strip() or not problem.strip():
        return "⚠️ Please enter both a SMILES string and the problem to address."
    
    try:
        prompt = f"This drug failed due to {problem}. Suggest structural modifications to improve safety:\n{smiles.strip()}"
        if name:
            prompt += f"\nName: {name}"
        
        output = model.generate(prompt, temperature=temperature, max_new_tokens=1024)
        return output.generated_text
    except Exception as e:
        return f"❌ Error generating suggestions: {e}"


def chat(message: str, history: list, temperature: float = 0.7) -> str:
    """Free-form drug discovery chat."""
    if model is None:
        return "⚠️ Model not loaded. Please click 'Load Model' first."
    
    if not message.strip():
        return "⚠️ Please enter a message."
    
    try:
        # Build conversation context
        context = ""
        for user_msg, assistant_msg in history[-3:]:  # Last 3 turns
            context += f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
            context += f"<|im_start|>assistant\n{assistant_msg}\n<|im_end|>\n"
        
        prompt = context + f"<|im_start|>user\n{message}\n<|im_end|>\n<|im_start|>assistant\n"
        
        output = model.generate(prompt, temperature=temperature, max_new_tokens=1024)
        return output.generated_text
    except Exception as e:
        return f"❌ Error: {e}"


# Create Gradio interface
with gr.Blocks(
    title="Drug Discovery AI",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .tab-content { padding: 20px; }
        .example-btn { margin: 5px; }
    """
) as demo:
    
    # Header
    gr.Markdown("""
    # 🧬 Drug Discovery AI
    ### AI-Powered Drug Development Assistant
    
    This model helps pharmaceutical researchers analyze drug candidates, understand failures, 
    and get suggestions for safer drug development. Powered by **Qwen2.5-14B** fine-tuned on 
    drug discovery data.
    """, elem_classes="main-header")
    
    # Model loading section
    with gr.Row():
        with gr.Column(scale=3):
            model_status = gr.Textbox(
                label="Model Status",
                value="Model not loaded. Click 'Load Model' to start.",
                interactive=False,
            )
        with gr.Column(scale=1):
            load_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
    
    load_btn.click(fn=load_model, outputs=model_status)
    
    # Main tabs
    with gr.Tabs():
        
        # Tab 1: Drug Analysis
        with gr.TabItem("🔬 Drug Analysis"):
            gr.Markdown("### Analyze a drug candidate and predict approval likelihood")
            
            with gr.Row():
                with gr.Column(scale=2):
                    analysis_smiles = gr.Textbox(
                        label="SMILES String",
                        placeholder="Enter SMILES molecular string (e.g., CC(=O)OC1=CC=CC=C1C(=O)O for Aspirin)",
                        lines=2,
                    )
                    analysis_name = gr.Textbox(
                        label="Drug Name (optional)",
                        placeholder="e.g., Aspirin, Candidate-123",
                    )
                    analysis_temp = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature (creativity)",
                    )
                    analyze_btn = gr.Button("🔬 Analyze Drug", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    analysis_output = gr.Markdown(label="Analysis Result")
            
            # Examples
            gr.Markdown("### Quick Examples (Approved Drugs)")
            with gr.Row():
                for smiles, name, category in APPROVED_EXAMPLES[:5]:
                    gr.Button(f"✅ {name}", size="sm").click(
                        lambda s=smiles, n=name: (s, n),
                        outputs=[analysis_smiles, analysis_name],
                    )
            
            analyze_btn.click(
                fn=analyze_drug,
                inputs=[analysis_smiles, analysis_name, analysis_temp],
                outputs=analysis_output,
            )
        
        # Tab 2: Failure Analysis
        with gr.TabItem("⚠️ Failure Analysis"):
            gr.Markdown("### Understand why a drug failed in development")
            
            with gr.Row():
                with gr.Column(scale=2):
                    failure_smiles = gr.Textbox(
                        label="SMILES String",
                        placeholder="Enter SMILES of the failed drug",
                        lines=2,
                    )
                    failure_name = gr.Textbox(
                        label="Drug Name",
                        placeholder="e.g., Vioxx",
                    )
                    failure_reason = gr.Textbox(
                        label="Known Failure Reason (optional)",
                        placeholder="e.g., Cardiovascular toxicity",
                    )
                    failure_temp = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature",
                    )
                    failure_btn = gr.Button("🔍 Analyze Failure", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    failure_output = gr.Markdown(label="Failure Analysis")
            
            # Examples
            gr.Markdown("### Quick Examples (Failed/Withdrawn Drugs)")
            with gr.Row():
                for smiles, name, reason in FAILED_EXAMPLES:
                    gr.Button(f"❌ {name}", size="sm").click(
                        lambda s=smiles, n=name, r=reason: (s, n, r),
                        outputs=[failure_smiles, failure_name, failure_reason],
                    )
            
            failure_btn.click(
                fn=analyze_failure,
                inputs=[failure_smiles, failure_name, failure_reason, failure_temp],
                outputs=failure_output,
            )
        
        # Tab 3: Comparison
        with gr.TabItem("⚖️ Drug Comparison"):
            gr.Markdown("### Compare two drug candidates")
            
            with gr.Row():
                with gr.Column():
                    comp_smiles1 = gr.Textbox(label="Drug 1 SMILES", placeholder="First drug SMILES")
                    comp_name1 = gr.Textbox(label="Drug 1 Name", placeholder="First drug name")
                
                with gr.Column():
                    comp_smiles2 = gr.Textbox(label="Drug 2 SMILES", placeholder="Second drug SMILES")
                    comp_name2 = gr.Textbox(label="Drug 2 Name", placeholder="Second drug name")
            
            comp_temp = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
            comp_btn = gr.Button("⚖️ Compare Drugs", variant="primary", size="lg")
            comp_output = gr.Markdown(label="Comparison Result")
            
            # Pre-load example
            gr.Button("Load Example: Aspirin vs Vioxx", size="sm").click(
                lambda: (
                    "CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin",
                    "CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3", "Rofecoxib (Vioxx)"
                ),
                outputs=[comp_smiles1, comp_name1, comp_smiles2, comp_name2],
            )
            
            comp_btn.click(
                fn=compare_drugs,
                inputs=[comp_smiles1, comp_name1, comp_smiles2, comp_name2, comp_temp],
                outputs=comp_output,
            )
        
        # Tab 4: Improvement Suggestions
        with gr.TabItem("💡 Suggestions"):
            gr.Markdown("### Get suggestions to improve a failed drug")
            
            with gr.Row():
                with gr.Column(scale=2):
                    sugg_smiles = gr.Textbox(
                        label="SMILES String",
                        placeholder="Enter SMILES of the problematic drug",
                        lines=2,
                    )
                    sugg_name = gr.Textbox(
                        label="Drug Name",
                        placeholder="Drug name",
                    )
                    sugg_problem = gr.Textbox(
                        label="Problem to Address",
                        placeholder="e.g., hepatotoxicity, QT prolongation, poor bioavailability",
                    )
                    sugg_temp = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature",
                    )
                    sugg_btn = gr.Button("💡 Get Suggestions", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    sugg_output = gr.Markdown(label="Improvement Suggestions")
            
            sugg_btn.click(
                fn=suggest_improvements,
                inputs=[sugg_smiles, sugg_name, sugg_problem, sugg_temp],
                outputs=sugg_output,
            )
        
        # Tab 5: Free Chat
        with gr.TabItem("💬 Chat"):
            gr.Markdown("### Open-ended drug discovery discussion")
            
            chatbot = gr.Chatbot(height=400, label="Conversation")
            chat_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about drug discovery, molecular properties, safety...",
                lines=2,
            )
            chat_temp = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
            chat_btn = gr.Button("💬 Send", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Chat")
            
            def respond(message, history, temperature):
                response = chat(message, history, temperature)
                history.append((message, response))
                return history, ""
            
            chat_btn.click(
                fn=respond,
                inputs=[chat_input, chatbot, chat_temp],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                fn=respond,
                inputs=[chat_input, chatbot, chat_temp],
                outputs=[chatbot, chat_input],
            )
            clear_btn.click(lambda: [], outputs=chatbot)
    
    # Footer
    gr.Markdown("""
    ---
    ### About
    - **Model:** Fine-tuned Qwen2.5-14B on drug discovery data
    - **Training:** Full fine-tuning on AMD MI300X 192GB
    - **Data:** ChEMBL, FDA, DrugBank, clinical trial failures
    
    ⚠️ **Disclaimer:** This is an AI research tool. Predictions should not replace expert judgment 
    in drug development. Always consult with qualified pharmaceutical scientists.
    """)


if __name__ == "__main__":
    # Auto-load model if available
    print("Starting Drug Discovery AI...")
    
    demo.launch(
        share=True,  # Create public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
    )
