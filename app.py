import streamlit as st
import pandas as pd
import yfinance as yf

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")

# Function to get day low for a given symbol
def get_daylow(symbol):
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
    # Create a checkbox for each ETF symbol
    checkbox = st.checkbox(
        f"{row['symbol']} - {row['daylow']}", 
        key=row['symbol'], 
        value=row['symbol'] in st.session_state.selected_etfs
    )
    
    # Update the session state based on the checkbox state
    if checkbox:
        if row['symbol'] not in st.session_state.selected_etfs:
            toggle_selection(row['symbol'])
    else:
        if row['symbol'] in st.session_state.selected_etfs:
            toggle_selection(row['symbol'])

# Display the selected ETFs
st.write("Selected ETFs:", st.session_state.selected_etfs)
