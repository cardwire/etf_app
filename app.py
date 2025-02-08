import streamlit as st
import pandas as pd
import yfinance as yf
import time

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")




# Fetch the dayLow values in batches
symbols = etf_df["symbol"].tolist()
day_low_values = get_daylow_batch(symbols)

# Add dayLow column to etf_df
etf_df["dayLow"] = etf_df["symbol"].map(day_low_values)

# Display dataframe
st.write("ETF Data:", etf_df.head())

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
    checkbox = st.checkbox(
        f"{row['symbol']} - {row['dayLow']}", 
        key=row['symbol'], 
        value=row['symbol'] in st.session_state.selected_etfs
    )

    # Update session state based on the checkbox state
    if checkbox:
        if row['symbol'] not in st.session_state.selected_etfs:
            toggle_selection(row['symbol'])
    else:
        if row['symbol'] in st.session_state.selected_etfs:
            toggle_selection(row['symbol'])

# Display the selected ETFs
st.write("Selected ETFs:", st.session_state.selected_etfs)

