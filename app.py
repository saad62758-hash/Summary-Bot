import streamlit as st
import numpy as np
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Download NLP resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="ANN Complaint Summarizer", layout="centered")
st.title("🧠 AI Complaint Summarizer using ANN")
st.write("Paste any complaint text below and the ANN will generate a short summary.")

# Text Input
user_text = st.text_area("Enter your complaint text:", height=200)

if st.button("Generate Summary"):
    if user_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # ============ PREPROCESSING ============
        stop_words = set(stopwords.words("english"))

        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            return text

        sentences = sent_tokenize(user_text)
        cleaned_sentences = [clean_text(s) for s in sentences]

        tokenized = []
        for s in cleaned_sentences:
            words = word_tokenize(s)
            words = [w for w in words if w not in stop_words]
            tokenized.append(words)

        # ============ WORD EMBEDDINGS ============
        w2v = Word2Vec(tokenized, vector_size=100, window=5, min_count=1)

        def sentence_vector(sentence):
            vectors = [w2v.wv[word] for word in sentence]
            return np.mean(vectors, axis=0)

        sentence_vectors = np.array([sentence_vector(s) for s in tokenized])

        # ============ ANN MODEL ============
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(100,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse')

        # Fake training labels (importance scores for demo)
        importance = np.array([len(s) for s in tokenized]).reshape(-1, 1)
        importance = importance / importance.max()

        model.fit(sentence_vectors, importance, epochs=25, verbose=0)

        # ============ PREDICTION ============
        scores = model.predict(sentence_vectors)

        # ============ SUMMARY ============
        def summarize(sentences, scores, top_n=2):
            ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            summary = " ".join([s for _, s in ranked[:top_n]])
            return summary

        summary = summarize(sentences, scores, top_n=2)

        # ============ OUTPUT ============
        st.subheader("📄 Original Text")
        st.write(user_text)

        st.subheader("✂️ Generated Summary (ANN Output)")
        st.success(summary)

        st.subheader("📊 Sentence Importance Scores")
        for i, score in enumerate(scores):
            st.write(f"Sentence {i+1}: {sentences[i]}  →  Score: {float(score[0]):.3f}")
