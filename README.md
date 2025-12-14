# spam-detection-lstm
A real-world spam detection web app using Bidirectional LSTM and Streamlit to classify messages as Spam or Not Spam with confidence scores.

# 📧 Spam Detection using Deep Learning (LSTM + Streamlit)

A **real-world Spam Detection Web App** built using **Deep Learning (Bidirectional LSTM)** and deployed with **Streamlit**.  
This project classifies messages as **Spam ❌** or **Not Spam (Ham) ✅**, similar to Gmail/SMS spam filters.

---

## 🚀 Live Features
- 🔍 Detects spam messages in real time
- 🧠 Deep Learning model (Bidirectional LSTM)
- 📊 Shows confidence score (prediction probability)
- 🌐 Interactive Streamlit web interface
- 💾 Trained model & tokenizer reused (no retraining)

---

## 🧠 Model Details
- **Architecture:** Bidirectional LSTM
- **Embedding Dimension:** 128
- **Max Sequence Length:** 100
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Output:** Spam (1) / Ham (0)

---

## 📂 Dataset Format
The dataset contains two columns:

| Column | Description |
|------|-------------|
| Category | `spam` or `ham` |
| Message | Text message content |

**Label Encoding:**
- `ham → 0`
- `spam → 1`

Example:
```text
Category   Message
ham        Hey, are we meeting today?
spam       Win a free iPhone now! Click here.

🛠️ Tech Stack

Python
TensorFlow / Keras
LSTM (Deep Learning)
Pandas & NumPy
Scikit-learn
Joblib
Streamlit

📁 Project Structure
📦 spam-detection-lstm
 ┣ 📜 app.py
 ┣ 📜 spam_lstm_model.h5
 ┣ 📜 spam_tokenizer.pkl
 ┣ 📜 README.md
 ┗ 📜 requirements.txt

🌍 Real-World Use Cases

Email spam filtering
SMS spam detection
Customer support automation
Fraud & scam detection
NLP-based text classification systems

👨‍💻 Author

Vaibhav Singh
🎓 BSc Data Science
💡 Aspiring Machine Learning & NLP Engineer

⭐ Future Improvements

Transformer-based model (BERT)
Email header analysis
Multilingual spam detection
Deployment on Streamlit Cloud
Mobile-friendly UI.

