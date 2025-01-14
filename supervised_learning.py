import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import matplotlib.pyplot as plt
import shap
from openai import OpenAI

# Load cleaned data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    return df

# Text preprocessing
def preprocess_text(df):
    # Basic text cleaning
    df['avis'] = df['avis'].str.lower()
    df['avis'] = df['avis'].str.replace(r'[^\w\s]', '', regex=True)
    return df

# TF-IDF with Logistic Regression
def train_tfidf_model(X_train, y_train):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression())
    ])
    model.fit(X_train, y_train)
    return model

# LSTM Model
def train_lstm_model(X_train, y_train, max_words=5000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        LSTM(64, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    
    history = model.fit(X_train_pad, y_train,
                       epochs=5,
                       batch_size=32,
                       validation_split=0.2)
    
    return model, tokenizer, history

# LLM-based classification
def llm_classification(text):
    client = OpenAI(
        api_key="sk-b49b64cec01542e88b88c4520c065628",
        base_url="https://api.deepseek.com"
    )
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an insurance review classifier. Classify the text as positive or negative."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def explain_tfidf_model(model, X_test):
    """Explain TF-IDF model using SHAP"""
    explainer = shap.LinearExplainer(model.named_steps['clf'], 
                                   model.named_steps['tfidf'].transform(X_test))
    shap_values = explainer.shap_values(model.named_steps['tfidf'].transform(X_test))
    
    # Visualize explanations
    shap.summary_plot(shap_values, 
                     model.named_steps['tfidf'].transform(X_test),
                     feature_names=model.named_steps['tfidf'].get_feature_names_out(),
                     show=False)
    plt.savefig('visualizations/tfidf_shap_summary.png')
    plt.close()

def explain_lstm_model(model, tokenizer, X_test, max_len=100):
    """Explain LSTM model using SHAP"""
    try:
        # Prepare test data
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
        
        # Create background data
        background = X_test_pad[np.random.choice(X_test_pad.shape[0], 100, replace=False)]
        
        # Create SHAP explainer
        explainer = shap.DeepExplainer(model, background)
        
        # Get SHAP values for first 10 samples
        shap_values = explainer.shap_values(X_test_pad[:10])
        
        # Visualize explanations
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first output if multiple
            
        plt.figure()
        shap.summary_plot(shap_values, 
                         X_test_pad[:10],
                         feature_names=list(tokenizer.index_word.values()),
                         show=False)
        plt.savefig('visualizations/lstm_shap_summary.png')
        plt.close()
        
    except Exception as e:
        print(f"Error explaining LSTM model: {str(e)}")
        print("Falling back to simpler SHAP explainer")
        
        # Try KernelExplainer as fallback
        try:
            def predict_wrapper(x):
                return model.predict(pad_sequences(x, maxlen=max_len))
                
            explainer = shap.KernelExplainer(predict_wrapper, 
                                           np.zeros((1, max_len)))
            shap_values = explainer.shap_values(X_test_pad[:10])
            
            plt.figure()
            shap.summary_plot(shap_values, 
                             X_test_pad[:10],
                             feature_names=list(tokenizer.index_word.values()),
                             show=False)
            plt.savefig('visualizations/lstm_shap_summary.png')
            plt.close()
            
        except Exception as e2:
            print(f"Fallback explanation failed: {str(e2)}")

# Main function
def main():
    # Load and preprocess data
    df = load_data()
    df = preprocess_text(df)
    
    # Prepare data
    df['sentiment'] = df['note'].apply(lambda x: 1 if x > 3 else 0)
    X = df['avis']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train TF-IDF model
    print("Training TF-IDF model...")
    tfidf_model = train_tfidf_model(X_train, y_train)
    y_pred = tfidf_model.predict(X_test)
    print("TF-IDF Model Results:")
    print(classification_report(y_test, y_pred))
    
    # Explain TF-IDF model
    print("\nExplaining TF-IDF model with SHAP...")
    explain_tfidf_model(tfidf_model, X_test)
    
    # Save TF-IDF model
    joblib.dump(tfidf_model, 'tfidf_model.pkl')
    
    # Train LSTM model
    print("\nTraining LSTM model...")
    lstm_model, tokenizer, history = train_lstm_model(X_train, y_train)
    
    # Explain LSTM model
    print("\nExplaining LSTM model with SHAP...")
    explain_lstm_model(lstm_model, tokenizer, X_test)
    
    # Save LSTM model
    lstm_model.save('lstm_model.h5')
    joblib.dump(tokenizer, 'tokenizer.pkl')
    
    # Plot training history
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('LSTM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('lstm_accuracy.png')
    
    # Test LLM classification
    sample_text = "The insurance process was smooth and efficient"
    print("\nLLM Classification Example:")
    print(f"Text: {sample_text}")
    print(f"Classification: {llm_classification(sample_text)}")

if __name__ == '__main__':
    main()
