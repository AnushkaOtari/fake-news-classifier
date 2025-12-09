import streamlit as st
import pickle
import sklearn

# ------------------ Load model & vectorizer ------------------ #
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ------------------ Page Config ------------------ #
st.set_page_config(
    page_title="Fake News Classifier",
    layout="centered"
)

# ------------------ Custom CSS for UI ------------------ #
st.markdown("""
    <style>
    .title-container {
        background: linear-gradient(90deg, #1a73e8, #4285f4);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .textbox {
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .footer {
        margin-top: 40px;
        text-align: center;
        color: #777;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------ #
st.markdown("<div class='title-container'> Fake News Classifier</div>", unsafe_allow_html=True)
st.write("### Detect whether news is **REAL** or **FAKE** : ")

# ------------------ User Input ------------------ #
news_text = st.text_area(
    "Paste news article text below:",
    height=200,
    placeholder="Type or paste article content here...",
)

col1, col2 = st.columns(2)

# ------------------ Predict Button ------------------ #
with col1:
    predict_btn = st.button("🔍Analyze")

# ------------------ Clear Button ------------------ #
with col2:
    clear_btn = st.button("🧹 Clear")

if clear_btn:
    st.experimental_rerun()

# ------------------ Background Wallpaper ------------------ #
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.freepik.com/free-photo/top-view-old-french-newspaper-pieces_23994233.htm#fromView=keyword&page=1&position=0&uuid=682000ef-a601-45de-90f6-6064ae0abbba&query=Newspaper+background");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)


# ------------------ Prediction Logic ------------------ #
if predict_btn:
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        transformed = vectorizer.transform([news_text])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.markdown("<div class='result-card' style='background:#d4edda; color:#155724;'>✔ REAL NEWS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-card' style='background:#f8d7da; color:#721c24;'>❌ FAKE NEWS</div>", unsafe_allow_html=True)

# ------------------ Footer ------------------ #
st.markdown("<div class='footer'>Built with ❤️ by Anushka • Python | Machine Learning | NLP</div>", unsafe_allow_html=True)
