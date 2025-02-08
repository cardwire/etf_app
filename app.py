import streamlit as st
import pandas as pd
import yfinance as yf
import time

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")

# Function to get day low for a batch of symbols
def get_daylow_batch(symbols):
    day_low_dict = {}
    
    for i in range(0, len(symbols), 50):  # Process in chunks of 50
        batch = symbols[i:i+50]
        tickers = yf.Tickers(" ".join(batch))  # Fetch multiple tickers at once

        for symbol in batch:
            try:
                day_low = tickers.tickers[symbol].info.get('dayLow', None)
                day_low_dict[symbol] = day_low
            except Exception as e:
                st.warning(f"Failed to get data for {symbol}: {e}")
                day_low_dict[symbol] = None
        
        time.sleep(1)  # Pause for 1 second before processing the next batch

    return day_low_dict

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

