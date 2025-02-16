import streamlit as st
import pandas as pd
import plotly.express as px
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf



st.set_page_config(page_title="Latest Actions and current Performance", page_icon="📊")

st.markdown("# Select an ETF from the interactive Table. Or directly select your ETF of choice via searching its Tickersymbol")


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

# Check if an ETF is selected
if selected_etfs:
    # get symbol and ticker from selected ETF
    symbol = selected_etfs[0]
    ticker = yf.Ticker(symbol)

    # get long business summary
    long_sum = ticker.info['longBusinessSummary']

    st.markdown(f"## {symbol}:")
    st.markdown(f"### {long_sum}")

    st.divider()
