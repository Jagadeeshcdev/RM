import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from dateutil.relativedelta import relativedelta
from google.oauth2 import service_account
import gspread

def convert_relative_time_to_date(relative_time):
    try:
        if 'months' in relative_time:
            delta = relativedelta(months=int(relative_time.split()[0]))
        else:
            delta = relativedelta(years=int(relative_time.split()[0]))

        return pd.to_datetime('today') - delta
    except ValueError:
        return pd.to_datetime('today')  # Default to today's date for non-numeric values

def perform_sentiment_analysis(df):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    df['compound_sentiment'] = df['review_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    zero_rating_mask = (df['rating'] == 0) & (df['compound_sentiment'].between(-0.05, 0.05))
    df.loc[zero_rating_mask, 'compound_sentiment'] = 0.0
    df['rating'] = np.ceil(df['rating']).astype(int)
    df['review_length'] = df['review_text'].apply(len)
    positive_keywords = ['good', 'excellent', 'positive', 'satisfactory', 'commendable']
    negative_keywords = ['bad', 'poor', 'worst', 'negative', 'unsatisfactory', 'disappointing']
    suggestion_keywords = ['suggestion', 'improvement', 'recommendation']
    df['feedback_category'] = 'review_text'
    df.loc[df['review_text'].str.contains('|'.join(positive_keywords), case=False), 'feedback_category'] = 'Positive'
    df.loc[df['review_text'].str.contains('|'.join(negative_keywords), case=False), 'feedback_category'] = 'Negative'
    df.loc[df['review_text'].str.contains('|'.join(suggestion_keywords), case=False), 'feedback_category'] = 'Suggestion'
    return df

def run():
    st.title("Sentiment Analysis Home")
    st.write("Upload your data and perform sentiment analysis")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data_frame = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data_frame.head())
        
        processed_data = perform_sentiment_analysis(data_frame)
        st.write("Processed Data:")
        st.write(processed_data.head())
        
        # Save the processed data to session state
        st.session_state['processed_data'] = processed_data

