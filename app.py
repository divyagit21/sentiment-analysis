import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pickled tokenizer and model
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4B8BBE;
        margin-bottom: 20px;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
    }
    .confidence {
        font-size: 18px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# app Title
st.markdown('<div class="main-title">Real-Time Text Sentiment Analyzer</div>', unsafe_allow_html=True)

# user Input
user_input = st.text_area("Enter your text below to analyze sentiment:", "", height=150)

# action button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Tokenize and pad the input
        tokenized_input = tokenizer.texts_to_sequences([user_input])
        max_len = 200
        padded_input = pad_sequences(tokenized_input, maxlen=max_len)

        # prediction
        prediction = model.predict(padded_input)
        prob = prediction[0][0]

        # sentiment determining
        sentiment = "Positive" if prob > 0.5 else "Negative"
        css_class = "positive" if prob > 0.5 else "negative"

        # result
        st.markdown(f'<div class="result-box {css_class}">Predicted Sentiment: {sentiment}<br>'
                    f'<div class="confidence">Confidence: {prob:.2%}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")
