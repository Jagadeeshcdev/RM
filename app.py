import streamlit as st

# Main app file
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon=":bar_chart:",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Charts"])

if page == "Home":
    import Home
    Home.run()
elif page == "Charts":
    import Charts
    Charts.run()
