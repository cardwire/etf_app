import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
from database.operations import prophet_forecast, ada_forecast
import numpy as np
import sklearn
import datetime as dt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# Set page configuration
st.set_page_config(page_title="ETF Forecast Tool", page_icon=":chart_with_upwards_trend:")

# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

# Title
st.markdown("# ETF Selection")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Add a column for checkbox selection
data['Select'] = False

# Display dataframe with checkboxes
edited_data = st.data_editor(data, column_config={
    "Select": st.column_config.CheckboxColumn("Select", help="Select an ETF to get a forecast")
}, hide_index=True)

# Update selected ETFs in session state
selected_etfs = edited_data[edited_data['Select']]['symbol'].tolist()

# Limit selection to 1 ETFs
if len(selected_etfs) > 1:
    st.warning("You can select only one ETF to forecast its performance.")
    selected_etfs = selected_etfs[:1]

# Store selected ETFs in session state
st.session_state.selected_etfs = selected_etfs

# get symbol and ticker from selected ETF
symbol = selected_etfs[0]
ticker = yf.Ticker(symbol)

# get long business summary
long_sum = ticker.info['longBusinessSummary']

st.markdown(f" ## you selected {symbol}")
st.markdown(f" ###read this general information on your chosen ETF: {long_sum}")

st.divider()

st.markdown(" ### select your forecast period and a forecast algorithm of choice here")

algorithm = st.select_slider("select your forecast algorithm", options=["prophet", "adaboost", "random forest", "naive bayes"])
period = st.slider("chose a forecast period in days", min_value=1, max_value=365)

def on_click_forecast():
    if algorithm == "prophet":
        prophet_forecast(ticker, period)
    elif algorithm == "adaboost":
        ada_forecast(ticker, period)
    elif algorithm == "random forest":
        rf_forecast(ticker, period)
    elif algorithm == "naive bayes":
        naiveb_forecast(ticker, period)

if st.button("click to forecast"):
    on_click_forecast()
