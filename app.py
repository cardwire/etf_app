
import streamlit as st
import pandas as pd
import yfinance as yf


st.markdown("# ETF Finder")

etf_df = pd.read_csv("database/etf_df.csv")

# Function to get day low for a given symbol
def get_daylow(checkbox):
    ticker = yf.Ticker(symbol)
    return ticker.info['dayLow']

# Apply the function to get day low values
etf_df['daylow'] = etf_df['symbol'].apply(get_daylow)

# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

# Function to toggle selection
def toggle_selection(symbol):
    if symbol in st.session_state.selected_etfs:
        st.session_state.selected_etfs.remove(symbol)
    else:
        st.session_state.selected_etfs.append(symbol)

# Display the dataframe with checkboxes
for i, row in etf_df.iterrows():
    checkbox = st.checkbox(row['symbol'], key=row['symbol'], value=row['symbol'] in st.session_state.selected_etfs)
    if checkbox and row['symbol'] not in st.session_state.selected_etfs:
        toggle_selection(row['symbol'])
    elif not checkbox and row['symbol'] in st.session_state.selected_etfs:
        toggle_selection(row['symbol'])

st.write("Selected ETFs:", st.session_state.selected_etfs)
