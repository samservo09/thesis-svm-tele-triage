import streamlit as st
import pandas as pd
import random

# Load dataset
df = pd.read_csv("combined-predictions.csv")

# Function to get a random statement
def get_random_statement():
    random_row = df.sample(n=1).iloc[0]  # Pick a random row
    return random_row["Post"], random_row["Actual_Label"], random_row["S_Predicted_Label"], random_row["R_Predicted_Label"]

# Function to color-code labels
def color_label(label):
    colors = {
        "Attempt": "red",  # Actual Attempt (Severe)
        "Behavior": "red",  # Suicidal Behavior (Severe)
        "Ideation": "orange",  # Suicidal Ideation (Moderate)
        "Indicator": "orange",  # Suicide Indicator (Moderate)
        "Supportive": "green",  # Supportive (Low Risk)
    }
    return f'<span style="color:{colors.get(label, "black")}; font-weight:bold;">{label}</span>'

# Initialize session state
if "post" not in st.session_state:
    st.session_state.post, st.session_state.actual, st.session_state.svm_pred, st.session_state.roberta_pred = get_random_statement()

# Streamlit UI
st.title("🔍 Suicide Severity Risk Classification - Model Comparison")

st.write("Click the button below to see a **random statement** with predictions from the models.")

# Display the current statement
st.write(f"### 📝 Statement: ")
st.info(st.session_state.post)

# Display actual and predicted labels with colors
st.markdown(f"**✅ Actual Label:** {color_label(st.session_state.actual)}", unsafe_allow_html=True)
st.markdown(f"**🤖 SVM Prediction:** {color_label(st.session_state.svm_pred)}", unsafe_allow_html=True)
st.markdown(f"**🦾 RoBERTa-SVM Prediction:** {color_label(st.session_state.roberta_pred)}", unsafe_allow_html=True)

# Button to fetch a new random statement
if st.button("🔄 Show Another"):
    st.session_state.post, st.session_state.actual, st.session_state.svm_pred, st.session_state.roberta_pred = get_random_statement()
    st.rerun()  # Refresh the UI to update with new statement

# Legend for label meanings with color-coded text
st.markdown("""
### **📝 Legend: Suicide Risk Severity Labels**
- <span style="color:red; font-weight:bold;">🟥 Actual Attempt (AT)</span> – Any deliberate action that could result in death, whether completed or not.  
  *🔹 Next Steps:* **Seek emergency assistance immediately.**  

- <span style="color:red; font-weight:bold;">🟥 Suicidal Behavior (BR)</span> – Actions with higher risk, including self-harm, active planning, or past hospitalization.  
  *🔹 Next Steps:* **Immediate intervention is necessary. Contact crisis support services.**  

- <span style="color:orange; font-weight:bold;">🟧 Suicidal Ideation (ID)</span> – Thoughts of suicide, preoccupation with risk factors (e.g., job loss, mental illness, substance abuse).  
  *🔹 Next Steps:* **Monitor closely, offer support, and encourage professional help.**  

- <span style="color:orange; font-weight:bold;">🟧 Suicide Indicator (IN)</span> – Mentions of at-risk factors (e.g., divorce, illness, loss of a loved one) without personal intent.  
  *🔹 Next Steps:* **Keep an eye on risk factors, encourage conversations, and offer support.**  

- <span style="color:green; font-weight:bold;">🟩 Supportive (SU)</span> – Engaging in discussion with no signs of personal risk, often offering help to others.  
  *🔹 Next Steps:* **Encourage continued support and positive engagement.**  
""", unsafe_allow_html=True)
