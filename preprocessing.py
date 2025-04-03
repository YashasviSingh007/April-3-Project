import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import re
import os

def download_nltk_data():
    """
    Download required NLTK data with error handling
    """
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")
        print("Using basic tokenization instead...")

def preprocess_text(text):
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing special characters and numbers
    3. Basic tokenization
    4. Removing stopwords
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Basic tokenization (split by whitespace)
    tokens = text.split()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    except:
        # If stopwords are not available, just use the tokens as is
        pass
    
    return ' '.join(tokens)

def get_sentiment(text):
    """
    Get sentiment polarity and subjectivity using TextBlob
    """
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def main():
    # Create processed data directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Download NLTK data
    download_nltk_data()
    
    # Read the dataset
    print("Reading dataset...")
    df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
    
    # Get the text column name (assuming it's the first column)
    text_column = df.columns[0]
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Get sentiment analysis
    print("Performing sentiment analysis...")
    sentiment_results = df['processed_text'].apply(get_sentiment)
    df['sentiment_polarity'] = sentiment_results.apply(lambda x: x['polarity'])
    df['sentiment_subjectivity'] = sentiment_results.apply(lambda x: x['subjectivity'])
    
    # Save processed data
    print("Saving processed data...")
    df.to_csv('data/processed/processed_data.csv', index=False)
    print("Preprocessing completed!")

if __name__ == "__main__":
    main() 