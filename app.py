import streamlit as st
import pandas as pd
import yfinance as yf

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")


The code snippet you provided looks correct for adding the `dayLow` column to the `etf_df` DataFrame. However, the error might be due to missing or incorrect data in the CSV file or an issue with the `yfinance` library.

Here are a few steps to help debug and potentially fix the issue:

1. **Check if `symbol` column exists**:
   Make sure the `symbol` column exists in your CSV file.

2. **Check for missing data**:
   Ensure there are no missing or invalid symbols in your `etf_df`.

3. **Update `yfinance`**:
   Make sure you are using the latest version of the `yfinance` library.

You can add some debug prints to check the data:

```python
# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")

print(etf_df.head())  # Check the first few rows of the DataFrame

# Function to get day low for a given symbol
def get_daylow(symbol):
    try:
        return yf.Ticker(symbol).info['dayLow']
    except KeyError:
        print(f"Failed to get data for {symbol}")
        return None

# Add dayLow column to etf_df
etf_df["dayLow"] = etf_df["symbol"].apply(get_daylow)

print(etf_df.head())  # Check the DataFrame after adding the dayLow column
```

This will help identify if there are any issues with the data or fetching the dayLow values.

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
