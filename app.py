import streamlit as st
import pandas as pd
import random

# Load dataset
df = pd.read_csv("combined-predictions.csv")

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

st.write("Click the button below to see a **random statement** with predictions from the models.")

# Display the current statement
st.write(f"### 📝 Statement:")
st.info(st.session_state.post)

# Display labels with color formatting
st.markdown(f"**✅ Actual Label:** {color_label(st.session_state.actual)}", unsafe_allow_html=True)
st.markdown(f"**🤖 SVM Prediction:** {color_label(st.session_state.svm_pred)}", unsafe_allow_html=True)
st.markdown(f"**🦾 RoBERTa-SVM Prediction:** {color_label(st.session_state.roberta_pred)}", unsafe_allow_html=True)

# Refresh to get a new statement
if st.button("🔄 Show Another"):
    st.session_state.post, st.session_state.actual, st.session_state.svm_pred, st.session_state.roberta_pred = get_random_statement()
    st.rerun()

# Color-coded Legend with background highlights
st.markdown("""
### **📝 Legend: Suicide Risk Severity Labels**

<div style="padding:10px; border-radius:8px; margin-bottom:10px;">
<span style="color:#D32F2F; font-weight:bold;">Attempt (AT):</span> Any deliberate action that could result in death, whether completed or not.  
🔹 <i>Next Steps:</i> <b>Seek emergency assistance immediately.</b>
</div>

<div style="padding:10px; border-radius:8px; margin-bottom:10px;">
<span style="color:#F44336; font-weight:bold;">Behavior (BR):</span> Actions with higher risk, including self-harm, active planning, or past hospitalization.  
🔹 <i>Next Steps:</i> <b>Immediate intervention is necessary. Contact crisis support services.</b>
</div>

<div style="padding:10px; border-radius:8px; margin-bottom:10px;">
<span style="color:#FB8C00; font-weight:bold;">Ideation (ID):</span> Thoughts of suicide, preoccupation with risk factors (e.g., job loss, mental illness, substance abuse).  
🔹 <i>Next Steps:</i> <b>Monitor closely, offer support, and encourage professional help.</b>
</div>

<div style="padding:10px; border-radius:8px; margin-bottom:10px;">
<span style="color:#FFB74D; font-weight:bold;">Indicator (IN):</span> Mentions of at-risk factors (e.g., divorce, illness, loss of a loved one) without personal intent.  
🔹 <i>Next Steps:</i> <b>Keep an eye on risk factors, encourage conversations, and offer support.</b>
</div>

<div style="padding:10px; border-radius:8px;">
<span style="color:#388E3C; font-weight:bold;">Supportive (SU):</span> Engaging in discussion with no signs of personal risk, often offering help to others.  
🔹 <i>Next Steps:</i> <b>Encourage continued support and positive engagement.</b>
</div>
""", unsafe_allow_html=True)
