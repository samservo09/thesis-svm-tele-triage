import gradio as gr
import torch
import joblib
import pickle
from transformers import RobertaModel, RobertaTokenizer

# Load RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

# Load PCA and SVM models
pca = joblib.load("pca.pkl")  # Ensure PCA was saved with joblib

with open("svm_roberta_model.pkl", "rb") as f:
    svm = joblib.load(f)  # Ensure SVM was saved with joblib

# Label mappings
label_mapping = {0: "Behavior", 1: "Supportive", 2: "Indicator", 3: "Attempt", 4: "Ideation"}
severity_mapping = {
    "Behavior": "🟢 No immediate distress detected. Stay mindful of your emotions.",
    "Supportive": "🟡 You may be seeking support. It's good to talk to someone you trust.",
    "Indicator": "🟠 Potential signs of distress detected. Consider reaching out for guidance.",
    "Attempt": "🔴 High distress detected! Please reach out for professional help.",
    "Ideation": "🔴 Critical risk detected! Please seek emergency support immediately."
}

# Function to get RoBERTa embeddings
def get_roberta_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract CLS token embeddings

# Function to predict label
def predict_label(user_input):
    try:
        # Generate RoBERTa embeddings
        user_input_embedding = get_roberta_embeddings([user_input])

        # Apply PCA
        user_input_pca = pca.transform(user_input_embedding)

        # Make prediction using SVM
        prediction = svm.predict(user_input_pca)

        # Debugging: Print the raw prediction
        print(f"Raw model output: {prediction}")

        # Ensure prediction is mapped correctly
        predicted_label = label_mapping.get(int(prediction[0]), "Unknown")

        # Debugging: Print mapped label
        print(f"Mapped Label: {predicted_label}")

        # Get severity message, ensuring it does not throw KeyError
        severity_message = severity_mapping.get(predicted_label, "⚠️ Unable to determine severity.")

        return f"Prediction: {predicted_label}\nSeverity: {severity_message}"

    except Exception as e:
        print(f"Error in prediction: {e}")
        return "⚠️ An error occurred while processing your input. Please try again."

# Create Gradio UI
iface = gr.Interface(
    fn=predict_label,
    inputs="text",
    outputs="text",
    title="SVM-RoBERTa Chatbot",
    description="Enter a message describing your emotions, and the model will assess the severity.",
)

# Run the chatbot
if __name__ == "__main__":
    iface.launch()
