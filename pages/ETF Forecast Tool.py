# [Theme] = .streamlit/config.toml

import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
from database.operations import prophet_forecast, ar_forecast, arima_forecast, sarima_forecast, es_forecast, xgb_forecast, lstm_forecast
import numpy as np
import sklearn
import datetime as dt

# Set page configuration
st.set_page_config(page_title="ETF Forecast Tool", page_icon=":chart_with_upwards_trend:")

# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

# Title
st.markdown("### ETF Selection")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")
data = data[["symbol", "full_name", "type", "total_assets", "ytd_return"]]
data = data.rename(columns={"full_name": "funds name", "type": "funds type", "total_assets": "AUM"})

# Add a column for checkbox selection
data['Select'] = False

# Display dataframe with checkboxes
edited_data = st.data_editor(data, column_config={
    "Select": st.column_config.CheckboxColumn("Select", help="Select an ETF to get a forecast")
}, hide_index=True)

# Update selected ETFs in session state
selected_etfs = edited_data[edited_data['Select']]['symbol'].tolist()

# Limit selection to 1 ETF
if len(selected_etfs) > 1:
    st.warning("You can select only one ETF to forecast its performance.")
    selected_etfs = selected_etfs[:1]

# Store selected ETFs in session state
st.session_state.selected_etfs = selected_etfs

# Check if an ETF is selected
if selected_etfs:
    # get symbol and ticker from selected ETF
    symbol = selected_etfs[0]
    ticker = yf.Ticker(symbol)

    # get long business summary
    long_sum = ticker.info['longBusinessSummary']

    st.markdown(f"## You selected {symbol}")
    st.markdown(f"### Read this general information on your chosen ETF: {long_sum}")

    st.divider()

    st.markdown("### Select your forecast period and a forecast algorithm of choice here")

    algorithm = st.selectbox("Select your forecast algorithm", options=[
        "prophet", "ar", "arima", "sarima", "es", "xgb", "lstm"
    ])
    period = st.slider("Choose a forecast period in days", min_value=30, max_value=1825)

    def on_click_forecast():
        if algorithm == "prophet":
            prophet_forecast(ticker, period)
        elif algorithm == "ar":
            ar_forecast(ticker, period)
        elif algorithm == "arima":
            arima_forecast(ticker, period)
        elif algorithm == "sarima":
            sarima_forecast(ticker, period)
        elif algorithm == "es":
            es_forecast(ticker, period)
        elif algorithm == "xgb":
            xgb_forecast(ticker, period)
        elif algorithm == "lstm":
            lstm_forecast(ticker, period)

    if st.button("Click to forecast"):
        on_click_forecast()
else:
    st.warning("Please select an ETF to forecast.")
