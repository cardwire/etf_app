import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="ETF App", page_icon=":chart_with_upwards_trend:")

st.sidebar.title("Navigation")
st.sidebar.success("Select a page below:")
page = st.sidebar.radio("Go to", ["Homepage", "ETF Inspector", "Page 2", "Page 3"])

if page == "Homepage":
    st.markdown("# Homepage")
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
    
elif page == "Page 2":
    st.markdown("# Page 2 - Content Placeholder")
    st.write("This page will contain additional ETF insights.")

elif page == "Page 3":
    st.markdown("# Page 3 - Content Placeholder")
    st.write("This page will contain ETF risk analysis.")

