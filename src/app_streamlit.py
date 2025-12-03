import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Fake News Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 10px;
    }
    .sub {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 25px;
        text-align: center;
        font-size: 24px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------ #
st.markdown("<div class='title'>Fake News Classifier 📰</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Detect whether a news article is Real or Fake using Machine Learning</div>", unsafe_allow_html=True)

# ------------------ TEXT INPUT ------------------ #
news_text = st.text_area(
    "Paste a news article here:",
    height=200,
    placeholder="Type or paste news article content..."
)

# ------------------ BUTTON ------------------ #
if st.button("Analyze News 🔍"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        transformed = vectorizer.transform([news_text])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.success("✔ REAL NEWS")
            st.markdown("<div class='result-box' style='background-color:#d4edda; color:#155724;'>This news appears to be REAL ✔</div>", unsafe_allow_html=True)
        else:
            st.error("❌ FAKE NEWS")
            st.markdown("<div class='result-box' style='background-color:#f8d7da; color:#721c24;'>This news appears to be FAKE ❌</div>", unsafe_allow_html=True)



