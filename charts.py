import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def render_charts(df):
    # Chart 1: Sentiment Analysis Based on Ratings
    fig1 = px.scatter(df, x='rating', y='compound_sentiment', color='rating',
                      labels={'rating': 'Rating', 'compound_sentiment': 'Rating Sentiment'},
                      title='Sentiment Analysis Based On Ratings',
                      color_continuous_scale='viridis')
    st.plotly_chart(fig1)

    # Chart 2: Average Sentiment Per Product OR Service
    avg_sentiment_per_business = df.groupby('business_column')['compound_sentiment'].mean().reset_index()
    fig2 = go.Figure(data=[go.Bar(x=avg_sentiment_per_business['business_column'],
                                   y=avg_sentiment_per_business['compound_sentiment'],
                                   marker=dict(color=avg_sentiment_per_business['compound_sentiment'],
                                               colorscale='Viridis',
                                               colorbar=dict(title='Sentiment')))])
    fig2.update_layout(xaxis=dict(title='Business'), yaxis=dict(title='Average Sentiment'),
                       title='Average Sentiment Per Product OR Service')
    st.plotly_chart(fig2)

    # Chart 4: Average Sentiment On Business
    df_filtered = df[df['review_text'].notnull()][['atmosphere_compound', 'review_text', 'compound_sentiment']]
    df_filtered['Aspect'] = df_filtered['atmosphere_compound']
    custom_colors = {'aspect1': 'blue', 'aspect2': 'green', 'aspect3': 'red'}
    df_filtered['Color'] = df_filtered['Aspect'].map(custom_colors)
    avg_sentiment_by_aspect = df_filtered.groupby('Aspect')['compound_sentiment'].mean().reset_index()
    fig4 = px.bar(avg_sentiment_by_aspect, x='Aspect', y='compound_sentiment',
                  title='Average Sentiment On Business',
                  labels={'Aspect': 'Aspect', 'compound_sentiment': 'Average Business Sentiment'},
                  color='Aspect', color_discrete_map=custom_colors)
    st.plotly_chart(fig4)

    # Chart 7: Effective Positive and Negative Feedback
    positive_feedback = df[df['feedback_category'] == 'Positive'].sample(n=50, replace=True)
    negative_feedback = df[df['feedback_category'] == 'Negative'].sample(n=50, replace=True)
    combined_feedback = pd.concat([positive_feedback, negative_feedback])
    combined_feedback['feedback_category'] = combined_feedback['feedback_category'].astype(str)
    custom_palette = {'Positive': 'green', 'Negative': 'red'}
    fig7 = px.bar(combined_feedback, x='review_text', y='compound_sentiment', color='feedback_category',
                  labels={'review_text': 'Review Text', 'compound_sentiment': 'Average Feedback Sentiment'},
                  color_discrete_map=custom_palette)
    fig7.update_xaxes(tickangle=90)
    fig7.update_xaxes(showticklabels=False)
    st.plotly_chart(fig7)

    positive_feedback = df[df['feedback_category'] == 'Positive']['review_text']
    negative_feedback = df[df['feedback_category'] == 'Negative']['review_text']
    st.write("Positive Feedback Examples:")
    for feedback in positive_feedback:
        st.write("Positive Feedback:", feedback)
    st.write("Negative Feedback Examples:")
    for feedback in negative_feedback:
        st.write("Negative Feedback:", feedback)

def run():
    st.title("Charts Page")
    st.write("All chart reports and text output will be generated here.")

    if 'processed_data' in st.session_state:
        df = st.session_state['processed_data']
        render_charts(df)
    else:
        st.write("No data available. Please upload data from the Home page.")

