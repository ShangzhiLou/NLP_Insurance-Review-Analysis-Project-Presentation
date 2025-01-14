import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from openai import OpenAI

# Load models
tfidf_model = joblib.load('tfidf_model.pkl')
lstm_model = load_model('lstm_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

# Initialize LLM client
llm_client = OpenAI(
    api_key="sk-b49b64cec01542e88b88c4520c065628",
    base_url="https://api.deepseek.com"
)

# Streamlit app
st.title("Insurance Review Analysis")

# Prediction Section
st.header("Review Prediction")
review_text = st.text_area("Enter your insurance review:")

if review_text:
    # TF-IDF Prediction
    tfidf_pred = tfidf_model.predict([review_text])[0]
    tfidf_sentiment = "Positive" if tfidf_pred == 1 else "Negative"
    
    # LSTM Prediction
    seq = tokenizer.texts_to_sequences([review_text])
    padded = pad_sequences(seq, maxlen=100)
    lstm_pred = lstm_model.predict(padded, verbose=0)[0][0]
    lstm_sentiment = "Positive" if lstm_pred > 0.5 else "Negative"
    
    # LLM Prediction
    llm_response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an insurance review classifier. Classify the text as positive or negative and explain why."},
            {"role": "user", "content": review_text}
        ]
    )
    llm_output = llm_response.choices[0].message.content
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("TF-IDF Model", tfidf_sentiment)
    with col2:
        st.metric("LSTM Model", lstm_sentiment)
    with col3:
        st.metric("LLM Analysis", llm_output.split(".")[0])
    
    st.subheader("LLM Explanation")
    st.write(llm_output)

# Information Retrieval Section
st.header("Review Search")
search_query = st.text_input("Search reviews:")
if search_query:
    # Load data
    df = pd.read_csv('cleaned_data.csv')
    
    # Simple search
    results = df[df['avis'].str.contains(search_query, case=False)]
    
    if len(results) > 0:
        st.write(f"Found {len(results)} matching reviews:")
        for i, row in results.iterrows():
            with st.expander(f"Review {i+1} - {row['note']} stars"):
                st.write(row['avis'])
                st.write(f"Sentiment: {'Positive' if row['note'] > 3 else 'Negative'}")
    else:
        st.warning("No matching reviews found")

# RAG Section
st.header("Ask Questions")
question = st.text_input("Ask a question about insurance:")
if question:
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an insurance expert. Answer the user's question based on your knowledge."},
            {"role": "user", "content": question}
        ]
    )
    st.write(response.choices[0].message.content)
