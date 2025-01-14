# Insurance Review Analysis Project Presentation

## Project Overview
This project analyzes customer reviews for insurance products using natural language processing techniques. The system provides:
- Sentiment analysis using multiple models
- Review search functionality
- Insurance Q&A capabilities
- Comprehensive data visualizations

## File Structure
```
project/
├── app.py                # Streamlit web interface
├── data_explore.py       # Data loading, cleaning and analysis
├── supervised_learning.py # Model training
├── cleaned_data.csv      # Processed review data
├── models/               # Trained models
│   ├── tfidf_model.pkl
│   ├── lstm_model.h5
│   ├── tokenizer.pkl
├── visualizations/       # Analysis charts
│   ├── rating_distribution.png
│   ├── rating_by_product.png
│   ├── word_cloud.png
│   ├── top_words.png
│   ├── rating_trend.png
│   ├── sentiment_analysis.png
```

## Data Pipeline
1. **Data Collection**: 35 Excel files containing translated customer reviews
2. **Data Cleaning**:
   - Handle missing values and duplicates
   - Remove invalid ratings
   - Standardize text formatting
3. **Feature Engineering**:
   - Sentiment labeling (Positive/Negative based on rating)
   - Text preprocessing (lowercase, remove special chars)
   - Date parsing for time series analysis

## Analysis Highlights
- **Rating Distribution**: Visualized customer ratings across all products
- **Product Analysis**: Breakdown of ratings by product type
- **Word Analysis**: 
  - Word cloud of most frequent terms
  - Top 20 most common words
- **Temporal Trends**: Monthly average rating changes over time
- **Sentiment Analysis**: Distribution of positive/negative reviews

## Model Architecture
1. **TF-IDF + Logistic Regression**
   - Traditional NLP approach
   - Fast training and inference
   - Good baseline performance

2. **LSTM Neural Network**
   - Captures sequential patterns in text
   - Uses word embeddings
   - Achieved 85% validation accuracy

3. **LLM Classification**
   - Uses DeepSeek API
   - Provides detailed explanations
   - Handles complex language patterns

## Key Metrics
- **Data Size**: 35 Excel files containing ~10,000 reviews
- **Data Cleaning**:
  - Removed 15% of rows due to missing/invalid data
  - Final dataset: ~8,500 clean reviews
- **Model Performance**:
  - TF-IDF: 82% accuracy
  - LSTM: 85% accuracy
  - LLM: Provides qualitative analysis

## Technical Implementation
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Machine Learning**: Scikit-learn, TensorFlow
- **Web Interface**: Streamlit
- **LLM Integration**: DeepSeek API

## Business Value
- Provides insights into customer satisfaction
- Identifies common pain points
- Enables targeted product improvements
- Automated review analysis at scale
