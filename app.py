import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download stopwords once
nltk.download('stopwords')

# Constants (should match training config)
VOCAB_SIZE = 5000
SEQ_LEN = 20

try:
# Load model
    model = load_model('models/model.keras',compile=False)  # Update path if needed
except Exception as e:
    model = load_model('models/model.keras',compile=False)
    print("Error loading model:", e)

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove non-alphabet characters
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    return review

def vectorize_text(text):
    processed = preprocess_text(text)
    onehot_repr = one_hot(processed, VOCAB_SIZE)
    padded = pad_sequences([onehot_repr], padding='post', maxlen=SEQ_LEN)
    return np.array(padded)

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter the news headline below to check whether it's fake or real.")

# Input box
user_input = st.text_input("News Headline", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a news headline.")
    else:
        vectorized = vectorize_text(user_input)
        prediction = model.predict(vectorized)[0][0]
        label = "🟢 Real News" if prediction < 0.6 else "🔴 Fake News"
        st.markdown(f"### Prediction: {label}")
        st.write(f"Confidence: {prediction:.4f}")
