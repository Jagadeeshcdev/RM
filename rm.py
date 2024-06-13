import streamlit as st
import pandas as pd
import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import gspread
from google.auth import default

# Google Authentication
from google.colab import auth
auth.authenticate_user()

creds, _ = default()
gc = gspread.authorize(creds)

# Load NLTK VADER lexicon
nltk.download('vader_lexicon')

# Functions
def convert_relative_time_to_date(relative_time):
    try:
        if 'months' in relative_time:
            delta = relativedelta(months=int(relative_time.split()[0]))
        else:
            delta = relativedelta(years=int(relative_time.split()[0]))
        return pd.to_datetime('today') - delta
    except ValueError:
        return pd.to_datetime('today')

def perform_sentiment_analysis(df):
    sns.set(style="whitegrid")
    df['review_text'].fillna('', inplace=True)
    df['website content'].fillna('', inplace=True)
    sia = SentimentIntensityAnalyzer()
    df['compound_sentiment'] = df['review_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    zero_rating_mask = (df['rating'] == 0) & (df['compound_sentiment'].between(-0.05, 0.05))
    df.loc[zero_rating_mask, 'compound_sentiment'] = 0.0
    df['rating'] = np.ceil(df['rating']).astype(int)
    df['review_length'] = df['review_text'].apply(len)
    positive_keywords = ['good', 'excellent', 'positive', 'satisfactory', 'commendable']
    negative_keywords = ['bad', 'poor', 'worst', 'negative', 'unsatisfactory', 'disappointing']
    keyword_feedback = ['in little rock']
    df['feedback_category'] = 'review_text'
    df.loc[df['review_text'].str.contains('|'.join(positive_keywords), case=False), 'feedback_category'] = 'Positive'
    df.loc[df['review_text'].str.contains('|'.join(negative_keywords), case=False), 'feedback_category'] = 'Negative'
    df.loc[df['review_text'].str.contains('|'.join(keyword_feedback), case=False), 'feedback_category'] = 'Keywords Frequency'

    def is_spam(review_text, website_content):
        website_words = set(re.split(r'\W+', website_content.lower()))
        review_words = set(re.split(r'\W+', review_text.lower()))
        return not website_words.intersection(review_words)

    df['spam'] = df.apply(lambda row: is_spam(row['review_text'], row['website content']), axis=1)
    df.loc[df['spam'], 'feedback_category'] = 'Spam'

    return df

def render_charts(df):
    st.info(f"Initial DataFrame shape: {df.shape}")
    df['business_column'].fillna('Unknown', inplace=True)
    avg_sentiment_per_business = df.groupby('business_column')['compound_sentiment'].mean().reset_index()
    st.info(f"Chart 2 - Average Sentiment Per Product OR Service: {avg_sentiment_per_business}")
    fig2 = go.Figure(data=[go.Bar(x=avg_sentiment_per_business['business_column'],
                                   y=avg_sentiment_per_business['compound_sentiment'],
                                   marker=dict(color=avg_sentiment_per_business['compound_sentiment'],
                                               colorscale='Viridis',
                                               colorbar=dict(title='Sentiment')))])
    fig2.update_layout(xaxis=dict(title='Business'), yaxis=dict(title='Average Sentiment'),
                       title='Average Sentiment Per Product OR Service')
    fig2.update_layout(width=1080, height=720)
    st.plotly_chart(fig2)
    df['atmosphere_compound'].fillna('Unknown', inplace=True)
    df_filtered = df[df['review_text'].notnull()][['atmosphere_compound', 'review_text', 'compound_sentiment']]
    df_filtered['Aspect'] = df_filtered['atmosphere_compound']
    custom_colors = {
        'aspect1': 'blue',
        'aspect2': 'green',
        'aspect3': 'red',
    }
    df_filtered['Color'] = df_filtered['Aspect'].map(custom_colors).fillna('gray')
    avg_sentiment_by_aspect = df_filtered.groupby('Aspect')['compound_sentiment'].mean().reset_index()
    st.info(f"Chart 4 - Average Sentiment On Business: {avg_sentiment_by_aspect}")
    fig4 = px.bar(avg_sentiment_by_aspect, x='Aspect', y='compound_sentiment',
                  title='Average Sentiment On Business',
                  labels={'Aspect': 'Aspect', 'compound_sentiment': 'Average Business Sentiment'},
                  color='Aspect', color_discrete_map=custom_colors)
    fig4.update_layout(width=1080, height=720)
    st.plotly_chart(fig4)
  
    fig1 = px.scatter(df, x='rating', y='compound_sentiment', color='rating',
                      labels={'rating': 'Rating', 'compound_sentiment': 'Rating Sentiment'},
                      title='Sentiment Analysis Based On Ratings',
                      color_continuous_scale='viridis')
    fig1.update_layout(width=1080, height=720)
    st.plotly_chart(fig1)
    positive_feedback = df[df['feedback_category'] == 'Positive']
    negative_feedback = df[df['feedback_category'] == 'Negative']
    keyword_feedback = df[df['feedback_category'] == 'Keywords Frequency']
    spam_feedback = df[df['feedback_category'] == 'Spam']
    if not positive_feedback.empty:
        positive_feedback_sample = positive_feedback.sample(n=min(65, len(positive_feedback)), replace=True)
        positive_feedback_sample['feedback_category'] = 'Positive'
    else:
        positive_feedback_sample = pd.DataFrame(columns=['review_text', 'compound_sentiment', 'feedback_category'])
    if not negative_feedback.empty:
        negative_feedback_sample = negative_feedback.sample(n=min(65, len(negative_feedback)), replace=True)
        negative_feedback_sample['feedback_category'] = 'Negative'
    else:
        negative_feedback_sample = pd.DataFrame(columns=['review_text', 'compound_sentiment', 'feedback_category'])
    if not keyword_feedback.empty:
        keyword_feedback_sample = keyword_feedback.sample(n=min(65, len(keyword_feedback)), replace=True)
        keyword_feedback_sample['feedback_category'] = 'Keywords Frequency'
    else:
        keyword_feedback_sample = pd.DataFrame(columns=['review_text', 'compound_sentiment', 'feedback_category'])
    if not spam_feedback.empty:
        spam_feedback_sample = spam_feedback.sample(n=min(65, len(spam_feedback)), replace=True)
        spam_feedback_sample['feedback_category'] = 'Spam'
    else:
        spam_feedback_sample = pd.DataFrame(columns=['review_text', 'compound_sentiment', 'feedback_category'])
    combined_feedback = pd.concat([positive_feedback_sample, negative_feedback_sample, keyword_feedback_sample, spam_feedback_sample])
    custom_palette = {'Positive': 'green', 'Negative': 'red', 'Keywords Frequency': 'orange', 'Spam': 'purple'}
    fig = px.bar(combined_feedback, x='feedback_category', y='compound_sentiment', color='feedback_category',
                 labels={'feedback_category': 'Feedback Category', 'compound_sentiment': 'Average Feedback Sentiment'},
                 color_discrete_map=custom_palette)
    fig.update_layout(width=1080, height=720)
    st.plotly_chart(fig)
    st.subheader("Positive Feedback")
    for feedback in positive_feedback_sample['review_text']:
        st.write(feedback)
    st.subheader("Negative Feedback")
    for feedback in negative_feedback_sample['review_text']:
        st.write(feedback)
    st.subheader("Keyword Frequency")
    for feedback in keyword_feedback_sample['review_text']:
        st.write(feedback)
    st.subheader("Spam Feedback")
    for feedback in spam_feedback_sample['review_text']:
        st.write(feedback)

sheet_url = 'https://docs.google.com/spreadsheets/d/1FX6KF-rX-GGPrkhaZPmjAQ1S9TEGc5yCExWGiAHFpWY/'
sheet = gc.open_by_url(sheet_url)
worksheet = sheet.worksheet('Sheet1')
data_frame = pd.DataFrame(worksheet.get_all_records())
processed_data = perform_sentiment_analysis(data_frame)
render_charts(processed_data)
