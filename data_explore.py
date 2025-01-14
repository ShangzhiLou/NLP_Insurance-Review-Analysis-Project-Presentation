import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import numpy as np

# Load data from Excel files
def load_data():
    try:
        data_dir = Path('data/Traduction avis clients')
        
        # Load all Excel files into DataFrames
        dfs = []
        for file in data_dir.glob('*.xlsx'):
            print(f"Loading file: {file}")
            df = pd.read_excel(file)
            print(f"Loaded {len(df)} rows from {file.name}")
            if len(df) > 0:
                print("First row sample:")
                print(df.iloc[0])
            dfs.append(df)
        
        if not dfs:
            raise ValueError("No Excel files found in the data directory")
            
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Successfully loaded {len(dfs)} Excel files")
        return combined_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Data cleaning
def clean_data(df):
    # Keep only rows with required columns
    df = df.dropna(subset=['note', 'avis'])
    
    # Fill missing values in other columns
    df = df.fillna({
        'auteur': 'Unknown',
        'assureur': 'Unknown',
        'produit': 'Unknown',
        'type': 'Unknown',
        'date_publication': '',
        'date_exp': '',
        'avis_en': '',
        'avis_cor': '',
        'avis_cor_en': ''
    })
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Convert text columns to string type
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns:
        df[col] = df[col].astype(str)
    
    # Convert note to numeric
    df['note'] = pd.to_numeric(df['note'], errors='coerce')
    
    # Remove rows with invalid ratings
    df = df[df['note'].between(1, 5)]
    
    return df

# Enhanced data analysis
def analyze_data(df):
    # Basic analysis
    print("\n=== Data Overview ===")
    print(df.info())
    print("\n=== Summary Statistics ===")
    print(df.describe(include='all'))
    
    # Validate required columns
    required_columns = ['note', 'avis']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Enhanced visualizations - save each as separate file
    
    # 1. Rating Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='note', palette='viridis')
    plt.title('Rating Distribution')
    plt.xlabel('Rating (1-5)')
    plt.ylabel('Number of Reviews')
    plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Rating Distribution by Product Type
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='produit', hue='note', palette='viridis')
    plt.title('Rating Distribution by Product Type')
    plt.xlabel('Product Type')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Rating')
    plt.xticks(rotation=45)
    plt.savefig('rating_by_product.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Word Cloud
    plt.figure(figsize=(12, 8))
    all_text = ' '.join(df['avis'].dropna())
    wordcloud = WordCloud(width=1200, height=800, 
                         background_color='white',
                         max_words=200).generate(all_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in Reviews', pad=20)
    plt.savefig('word_cloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Top 20 Most Frequent Words
    plt.figure(figsize=(12, 8))
    words = ' '.join(df['avis']).split()
    word_counts = Counter([w.lower() for w in words if len(w) > 2]).most_common(20)
    words, counts = zip(*word_counts)
    sns.barplot(x=list(counts), y=list(words), palette='magma')
    plt.title('Top 20 Most Frequent Words (length > 2)')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.savefig('top_words.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Rating Trend Over Time (if date exists)
    if 'date_publication' in df.columns:
        plt.figure(figsize=(12, 6))
        df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce')
        monthly_ratings = df.dropna(subset=['date_publication']).set_index('date_publication')['note'].resample('M').mean()
        monthly_ratings.plot()
        plt.title('Monthly Average Rating Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Rating')
        plt.grid(True)
        plt.savefig('rating_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Sentiment Analysis (basic)
    plt.figure(figsize=(12, 6))
    df['sentiment'] = df['note'].apply(lambda x: 'Positive' if x > 3 else 'Negative' if x < 3 else 'Neutral')
    sentiment_counts = df['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def main():
    # Load and process data
    df = load_data()
    df = clean_data(df)
    df = analyze_data(df)
    
    # Save cleaned data
    df.to_csv('cleaned_data.csv', index=False)
    print("Data exploration complete. Results saved to cleaned_data.csv")

if __name__ == '__main__':
    main()
