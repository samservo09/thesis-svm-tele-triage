import streamlit as st
import pandas as pd
import random
import streamlit as st
from PIL import Image

# Load dataset
df = pd.read_csv("combined-predictions.csv")

# Load dataset
df = pd.read_csv("combined-predictions.csv")

# Replace NaN vector entries with string representations of empty lists
df["tf-idf"] = df["tf-idf"].fillna("[]")
df["roberta_emb"] = df["roberta_emb"].fillna("[]")

# Optionally: Replace empty vector strings with actual zero-vectors of fixed length
df["tf-idf"] = df["tf-idf"].apply(lambda x: x if x != "[]" else str([0.0]*20))        # Adjust 20 if needed
df["roberta_emb"] = df["roberta_emb"].apply(lambda x: x if x != "[]" else str([0.0]*768))  # RoBERTa typically has 768-dim

# Define exact color hex codes for labels
label_colors = {
    "Attempt": "#D32F2F",       # Deep Red
    "Behavior": "#F44336",      # Light Red
    "Ideation": "#FB8C00",      # Deep Orange
    "Indicator": "#FFB74D",     # Light Orange
    "Supportive": "#388E3C",    # Green
}

# Function to get a random statement
def get_random_statement():
    random_row = df.sample(n=1).iloc[0]
    return random_row["Post"], random_row["Actual_Label"], random_row["S_Predicted_Label"], random_row["R_Predicted_Label"]

# Function to color-code labels using hex colors
def color_label(label):
    color = label_colors.get(label, "black")
    return f'<span style="color:{color}; font-weight:bold;">{label}</span>'

# Initialize session state
if "post" not in st.session_state:
    st.session_state.post, st.session_state.actual, st.session_state.svm_pred, st.session_state.roberta_pred = get_random_statement()

# Streamlit UI
st.title("🔍 Suicide Severity Risk Classification - Model Comparison")

st.markdown("### Simulate SOP 1 (Feature Loss): ")
st.markdown("##### Support Vector Machine (SVM) algorithm faces a limitation in text classification tasks due to their tendency to discard essential features of textual data.")

import ast

# Section for SOP 1 Feature Representation Comparison
st.markdown("##### Feature Representation Comparison for Current Post:")

# Find the current row
current_row = df[df["Post"] == st.session_state.post]

if not current_row.empty:
    tfidf_vector = ast.literal_eval(current_row["tf-idf"].values[0])
    roberta_vector = ast.literal_eval(current_row["roberta_emb"].values[0])

    # Format TF-IDF into a dataframe and show only non-zero top weights
    tfidf_df = pd.DataFrame(tfidf_vector, columns=["Weight"]).reset_index(names=["Feature Index"])
    tfidf_df_nonzero = tfidf_df[tfidf_df["Weight"] > 0].sort_values(by="Weight", ascending=False).head(20)

    st.markdown("###### Top TF-IDF Feature Weights (Non-zero, Top 20):")
    st.dataframe(tfidf_df_nonzero, use_container_width=True)

    # Format RoBERTa vector (first 20 values for brevity)
    roberta_df = pd.DataFrame(roberta_vector, columns=["Embedding Value"]).reset_index(names=["Index"])
    
    st.markdown("###### RoBERTa Embedding Vector (First 20 Dimensions):")
    st.dataframe(roberta_df.head(20), use_container_width=True)
else:
    st.warning("Current post not found in the dataframe.")

# Show label distribution table
st.markdown("### Simulate SOP 2 (Class Imbalance):")
st.markdown("##### The Support Vector Machine (SVM) algorithm inherently biases classification toward the majority class, resulting in poor performance on the minority class in imbalanced datasets.")

# Display the current statement
st.write(f"### 📝 Statement:")
st.info(st.session_state.post)

# Display labels with color formatting
st.markdown(f"**✅ Actual Label:** {color_label(st.session_state.actual)}", unsafe_allow_html=True)
st.markdown(f"**🤖 SVM Prediction:** {color_label(st.session_state.svm_pred)}", unsafe_allow_html=True)
st.markdown(f"**🦾 RoBERTa-SVM Prediction:** {color_label(st.session_state.roberta_pred)}", unsafe_allow_html=True)

st.write("Click the button below to see a **random statement** with predictions from the models.")

# Refresh to get a new statement
if st.button("🔄 Show Another"):
    st.session_state.post, st.session_state.actual, st.session_state.svm_pred, st.session_state.roberta_pred = get_random_statement()
    st.rerun()
    
label_counts = df["Actual_Label"].value_counts().reset_index()
label_counts.columns = ["Label", "Number of Posts"]

st.dataframe(label_counts.reset_index(drop=True), use_container_width=True)

st.markdown("### Simulate SOP 3 (HDLSS Data):")
st.markdown("##### The Support Vector Machine (SVM) algorithm struggles with high-dimensional, low-sample-size (HDLSS) textual data, needing improvements for better predictive performance.")

image1 = Image.open('svm-no-pca.png')
st.image(image1, caption='This is a visualization of the dataset of SVM with NO PCA applied.', use_column_width=True)

image2 = Image.open('svm-pca.png')
st.image(image2, caption='This is a visualization of the dataset of SVM with PCA applied.', use_column_width=True)

# Color-coded Legend with background highlights
st.markdown("""
### **📝 Legend: Suicide Risk Severity Labels**

<div style="padding:10px; border-radius:8px; margin-bottom:10px;">
<span style="color:#D32F2F; font-weight:bold;">Attempt (AT):</span> Any deliberate action that could result in death, whether completed or not. <br>
🔹 <i>Next Steps:</i> <b>Seek emergency assistance immediately.</b>
</div>

<div style="padding:10px; border-radius:8px; margin-bottom:10px;">
<span style="color:#F44336; font-weight:bold;">Behavior (BR):</span> Actions with higher risk, including self-harm, active planning, or past hospitalization.  <br>
🔹 <i>Next Steps:</i> <b>Immediate intervention is necessary. Contact crisis support services.</b>
</div>

<div style="padding:10px; border-radius:8px; margin-bottom:10px;">
<span style="color:#FB8C00; font-weight:bold;">Ideation (ID):</span> Thoughts of suicide, preoccupation with risk factors (e.g., job loss, mental illness, substance abuse).  <br>
🔹 <i>Next Steps:</i> <b>Monitor closely, offer support, and encourage professional help.</b>
</div>

<div style="padding:10px; border-radius:8px; margin-bottom:10px;">
<span style="color:#FFB74D; font-weight:bold;">Indicator (IN):</span> Mentions of at-risk factors (e.g., divorce, illness, loss of a loved one) without personal intent.  <br>
🔹 <i>Next Steps:</i> <b>Keep an eye on risk factors, encourage conversations, and offer support.</b>
</div>

<div style="padding:10px; border-radius:8px;">
<span style="color:#388E3C; font-weight:bold;">Supportive (SU):</span> Engaging in discussion with no signs of personal risk, often offering help to others.  <br>
🔹 <i>Next Steps:</i> <b>Encourage continued support and positive engagement.</b>
</div>
""", unsafe_allow_html=True)

