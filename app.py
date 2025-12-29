import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="centered"
)

# -------------------------------
# Load Model & Tokenizer
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("spam_lstm_model.h5", compile=False)
    tokenizer = joblib.load("spam_tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_artifacts()

MAX_LEN = 100

# -------------------------------
# App UI
# -------------------------------
st.title("📧 Spam Detection using Deep Learning")
st.markdown("""
This app uses a **Bidirectional LSTM model** to classify messages as  
**Spam ❌ or Not Spam ✅** — similar to real-world email/SMS filters.
""")

st.divider()

# User Input
user_message = st.text_area(
    "✍️ Enter your message below:",
    height=150,
    placeholder="e.g. Congratulations! You won a free lottery prize..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Check Message"):
    if user_message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([user_message])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Predict
        prob = model.predict(pad)[0][0]

        if prob >= 0.5:
            st.error("❌ **SPAM MESSAGE DETECTED**")
        else:
            st.success("✅ **NOT SPAM (HAM) MESSAGE**")

        st.info(f"📊 **Confidence Score:** {prob:.2f}")

st.divider()

# -------------------------------
# Example Messages
# -------------------------------
st.subheader("🧪 Try Example Messages")

col1, col2 = st.columns(2)

with col1:
    if st.button("Spam Example"):
        example = "Win Rs 10,000 cash prize now! Click link"
        st.write(example)

with col2:
    if st.button("Ham Example"):
        example = "Hey, are we meeting today?"
        st.write(example)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
---
👨‍💻 **Built by Vaibhav Singh**  
🎓 BSc Data Science | Aspiring ML & NLP Engineer

""")


