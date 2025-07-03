import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import pickle

# Load model and tokenizer from the Hugging Face Hub
model_name = "Sarthak279/Disease-symptom-prediction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)



# Load label encoder from uploaded pickle file
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define prediction logic
def predict_disease(note):
    inputs = tokenizer(
        note,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# Define Gradio UI
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Textbox(lines=4, placeholder="e.g. Patient complains of chest pain and breathlessness", label="📝 Enter Clinical Note or Symptoms"),
    outputs=gr.Textbox(label="🧠 Predicted Disease"),
    title="🩺 Sarthak's Disease Predictor",
    description="Enter symptoms or patient notes to predict a disease using a fine-tuned transformer model.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()