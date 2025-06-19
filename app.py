import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fuzzywuzzy import process

# ===== Load and Train =====
df = pd.read_csv("Ananth.csv")  # ✅ Replace with your CSV name
X = df['input']
y = df['chatbot']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

chat_dict = dict(zip(df['input'].str.lower(), df['chatbot']))

# ===== Prediction Function =====
def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]

    matches = process.extract(user_input.lower(), chat_dict.keys(), limit=1)
    if matches and matches[0][1] >= 70:
        return chat_dict[matches[0][0]]
    else:
        return prediction

# ===== Streamlit Page Config =====
st.set_page_config("Tanglish Chatbot", layout="centered")
st.markdown("<h2 style='text-align: center;'>🤖 Tanglish Friendly Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Chat with your virtual friend in Tanglish!</p>", unsafe_allow_html=True)

# ===== Chat History =====
if "history" not in st.session_state:
    st.session_state.history = []

# ===== User Input =====
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Type your message:", placeholder="Ex: hi da, enna da sollu...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    if user_input.lower() == "exit":
        bot_reply = "Paakalam da! 👋"
    else:
        bot_reply = get_response(user_input)

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_reply))

# ===== Display Chat History (Classic Bubble Style) =====
for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(
            f"<div style='background-color:#DCF8C6;padding:10px;border-radius:10px;margin:5px 30% 5px 5%;'><b>You:</b> {message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#F1F0F0;padding:10px;border-radius:10px;margin:5px 5% 5px 30%;'><b>Bot:</b> {message}</div>",
            unsafe_allow_html=True
        )

# ===== Footer =====
st.markdown("---")
st.markdown("<center><small>Made with ❤️ using Streamlit • Tanglish Style</small></center>", unsafe_allow_html=True)
