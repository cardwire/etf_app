import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="ETF App", page_icon=":chart_with_upwards_trend:")


st.sidebar.success("Select a page below:")
st.markdown(" ### Navigation Menu")
page = st.sidebar.radio("Go to", ["ETF App", "ETF Statistics", "ETF Inspector", "ETF 3D Visualizer", "ETF Forecast Tool"])

if page == "ETF App":
    st.markdown("# Welcome to the ETF App")
    st.markdown("Welcome to the ETF Finder App. Navigate using the sidebar.")
    
    # Load ETF data
    data = pd.read_excel("database/df.xlsx")
    st.dataframe(data)

elif page == "ETF Inspector":
    st.markdown("# ETF Selection and Candlestick Chart")
    symbol = st.selectbox("Choose an ETF symbol:", data['symbol'].unique())
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='1d', interval='1m')
    
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                         open=hist['Open'],
                                         high=hist['High'],
                                         low=hist['Low'],
                                         close=hist['Close'])])
    fig.update_layout(title=f'{symbol} Price Chart', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)
    
elif page == "ETF Statistics":
    st.markdown("# ETF Statistics - Content Placeholder")
    st.write("This page provides an exploratory data analysis of our ETF-Database.")

elif page == "ETF 3D-Visualizer":
    st.markdown("# ETF 3D-Visualizer - Content Placeholder")
    st.write("This page provides 3D-representation of our ETF Database based on UMAP dimension reduction technique.")

elif page == "ETF Forecast Tool":
    st.markdown("# ETF Forecast Tool - Content Placeholder")
    st.write("This page provides a set of tools to get predictions concerning future performance of ETFs.")




